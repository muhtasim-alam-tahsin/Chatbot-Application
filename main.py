import sqlite3
import os
import uuid

from typing import TypedDict,Annotated,Literal
from pydantic import Field,BaseModel
from dotenv import load_dotenv

load_dotenv()
from langgraph.types import Command,interrupt
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState,StateGraph,START,END,state
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition,ToolNode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
SQLITE_PATH = os.path.join(BASE_DIR, "checkpoints.sqlite")

llm = init_chat_model(
    model = 'llama-3.1-8b-instant',
    model_provider='groq',
    temperature=0.1
)
vision_llm = init_chat_model(
    model="qwen/qwen3.6-27b",
    model_provider="groq",
    temperature=0.1,
)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = Chroma(
    collection_name = "long_term_knowledge",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)
title_llm = init_chat_model(
    model="llama-3.1-8b-instant",
    model_provider="groq",
    temperature=0.3,
)

def generate_title(first_message: str) -> str:
    prompt = (
        "Summarize the following user message into a short chat title, "
        "3 to 6 words, no punctuation, no quotes. Just the title text.\n\n"
        f"Message: {first_message}"
    )
    response = title_llm.invoke([{"role": "user", "content": prompt}])
    return response.content.strip().strip('"')
def init_titles_table():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS thread_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT
        )
    """)
    # migration: add created_at if it doesn't exist yet (older DBs won't have it)
    existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(thread_titles)").fetchall()]
    if "created_at" not in existing_cols:
        conn.execute("ALTER TABLE thread_titles ADD COLUMN created_at TEXT DEFAULT (datetime('now'))")
    conn.commit()
    conn.close()

def save_thread_title(thread_id: str, title: str):
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO thread_titles (thread_id, title, created_at)
        VALUES (?, ?, COALESCE(
            (SELECT created_at FROM thread_titles WHERE thread_id = ?),
            datetime('now')
        ))
    """, (thread_id, title, thread_id))
    conn.commit()
    conn.close()

def list_threads_with_titles():
    """Newest thread first — this gives you the stack/LIFO ordering."""
    conn = sqlite3.connect(SQLITE_PATH)
    rows = conn.execute(
        "SELECT thread_id, title, created_at FROM thread_titles ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return rows

def get_thread_title(thread_id: str) -> str | None:
    conn = sqlite3.connect(SQLITE_PATH)
    row = conn.execute(
        "SELECT title FROM thread_titles WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else None

def delete_thread(thread_id: str):
    """Remove a thread from titles AND from the checkpointer's tables."""
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("DELETE FROM thread_titles WHERE thread_id = ?", (thread_id,))
    for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
        try:
            conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
        except sqlite3.OperationalError:
            pass  # table name may differ by langgraph-checkpoint-sqlite version
    conn.commit()
    conn.close()

init_titles_table()  


def add_to_knowledge_base(text: str, source: str = "user_upload") -> int:
    chunk_size, overlap = 800, 100
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
 
    docs = [Document(page_content=c, metadata={"source": source}) for c in chunks]
    if docs:
        vector_store.add_documents(docs)
    return len(docs)

search_tool = DuckDuckGoSearchRun(
    description=(
        "Search the live web for current events, facts, or anything that may "
        "have changed recently. Use this when the user asks about something "
        "time-sensitive or you are not confident from your own knowledge or "
        "the retrieved context."
    )
)
tools = [search_tool]
llm_with_tools = llm.bind_tools(tools)

class state(TypedDict):
    messages : Annotated[list,add_messages]
    message_intent : str | None
    next_node : str| None

def prompt_llm(state: state):
    query = state["messages"][-1].content
    retrieved = vector_store.similarity_search(query, k=3)
    context = "\n".join(f"- {doc.page_content}" for doc in retrieved) or "(no relevant context found)"
 
    system = {
        "role": "system",
        "content": (
            "You are a helpful assistant with two sources of extra information: "
            "(1) a long-term knowledge base retrieved below, and (2) a live web "
            "search tool you can call for current events or anything you're unsure "
            "about. Use the retrieved context if it's relevant and cite it naturally. "
            "If it's not relevant, ignore it. Call the web search tool when the user "
            "asks about recent/current information you can't answer confidently.\n\n"
            f"Retrieved context:\n{context}"
        ),
    }
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}

def build_graph():
    """Compile the LangGraph app with a persistent SQLite checkpointer."""
    graph_builder = StateGraph(state)
 
    graph_builder.add_node("agent", prompt_llm)
    graph_builder.add_node("tools", ToolNode(tools))
 
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")
 
    # check_same_thread=False needed because Streamlit may call from different threads
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
 
    return graph_builder.compile(checkpointer=checkpointer)

# def get_graph():
#     return build_graph()
 
 
# graph = get_graph()

# config = {'configurable': {'thread_id': uuid.uuid4() }}
# while True:
#     user_message = input("Enter Message: ")
#     result = graph.invoke({'messages':[{'role': 'user','content': user_message}]}, config=config)

#     while '__interrupt__' in result:
#         prompt = result['__interrupt__'][0].value
#         decision = input(f"{prompt}\n>")
#         result = graph.invoke(Command(resume=decision), config=config) 

#     print(result['messages'][-1].content)







# def prompt_llm_chat(state:state):
#     messages = [{'role': 'system', 'content':'You are a talkative chatbot'}]+ state['messages']
#     response = llm.invoke(messages)

#     return {'messages': [{'role': 'assistant', 'content': response.content}]}
# def prompt_llm_rag(state:state):
#     query = state['messages'][-1].content
#     documents = vector_store.similarity_search(query,k=3)
#     context = '\n'.join(f'- {doc.page_content}' for doc in documents)

#     messages = [{'role': 'system', 'content':f"You are a RAG agent. Answer only this way: If you don't know the answer you should say that you don't know.\n\nContext:\n{context} "}] + state['messages']
#     response = llm.invoke(messages)

#     return {'messages': [{'role': 'assistant', 'content': response.content}]}
    

# graph_builder = StateGraph(state)

# graph_builder.add_node("chat_agent", prompt_llm_chat)
# graph_builder.add_node("rag_agent", prompt_llm_rag)

# graph_builder.add_edge('chat_agent',END)
# graph_builder.add_edge('rag_agent',END)


# def classify_intent(state:state):
#     last_intent = state.get('message_intent')
    
#     structured_llm = llm.with_structured_output(IntentClassifier)
#     result = structured_llm.invoke([
#         {'role': 'system', 'content': (
#             f"The previous message intent was '{last_intent}'. "
#             "Classify the user's latest message as 'chat', 'knowledge', or 'code'. "
#             "If it looks like a short follow-up answer (providing a detail, file name, "
#             "confirmation, or clarification) rather than a new independent request, "
#             f"keep the same intent as before: '{last_intent}'."
#         )},
#         {'role': 'user', 'content': state['messages'][-1].content}
#     ])
#     return {'message_intent': result.message_intent}

# class IntentClassifier(BaseModel):
#     message_intent: Literal['chat','knowledge','code'] = Field(..., description='' \
#     'classify whether the user wants to just chat, ask for knowledge or change in code for project')
