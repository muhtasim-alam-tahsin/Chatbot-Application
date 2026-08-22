import streamlit as st
import uuid
import sqlite3
from pypdf import PdfReader
import base64
from main import vision_llm
from main import add_to_knowledge_base, build_graph, generate_title, save_thread_title
from main import get_thread_title, list_threads_with_titles, delete_thread

st.set_page_config(page_title="AI Chatbot Platform", page_icon="🤖", layout="wide")

def get_graph():
    return build_graph()

graph = get_graph()

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'show_kb_panel' not in st.session_state:
    st.session_state.show_kb_panel = False


def list_threads():
    """Return every distinct thread_id ever saved to the checkpointer."""
    conn = sqlite3.connect("checkpoints.sqlite")
    cur = conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
    threads = [row[0] for row in cur.fetchall()]
    conn.close()
    return threads

past_threads = list_threads()


def load_thread_history(thread_id:str):
    config = {'configurable': {'thread_id': thread_id} }
    snapshot =  graph.get_state(config)
    msgs = snapshot.values.get("messages",[]) if snapshot and snapshot.values else []
    rendered = []
    for m in msgs:
        if getattr(m,"type",None) == 'human' and m.content:
            rendered.append({'role': 'user', 'content': m.content})
        elif getattr(m,"type",None) == 'ai' and m.content:
            rendered.append({'role':'assistant', 'content': m.content})
    return rendered


# Sidebar------   

with st.sidebar:

    if not st.session_state.show_kb_panel:
        st.title("🤖 Chatbot Platform")
 
        st.subheader("Conversation")
        if st.button("📚 Customize Knowledge Base", use_container_width=True):
                    st.session_state.show_kb_panel = True
                    st.rerun()
        current_title = get_thread_title(st.session_state.thread_id) or "Current Chat"
        st.markdown(f"**{current_title}**")
        st.divider()
        
        past_threads = list_threads()

        if past_threads:
            
            st.subheader("Previous Chats")

            if "confirm_delete" not in st.session_state:
                st.session_state.confirm_delete = None

            threads = list_threads_with_titles()   # already newest-first (stack order)

            for tid, title, created_at in threads:
                label = title or f"Untitled ({tid[:8]})"
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    if st.button(label, key=f"load_{tid}", use_container_width=True):
                        st.session_state.thread_id = tid
                        st.session_state.messages = load_thread_history(tid)
                        st.session_state.confirm_delete = None
                        st.rerun()
                with col_b:
                    if st.button("🗑", key=f"del_{tid}"):
                        st.session_state.confirm_delete = tid
                        st.rerun()
            if st.session_state.confirm_delete:
                tid = st.session_state.confirm_delete
                label = next((t or tid[:8] for i, t, _ in threads if i == tid), tid[:8])
                st.warning(f"Delete **{label}**? This can't be undone.")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Confirm delete", use_container_width=True):
                        delete_thread(tid)
                        if st.session_state.thread_id == tid:      # deleting the active chat
                            st.session_state.thread_id = str(uuid.uuid4())
                            st.session_state.messages = []
                        st.session_state.confirm_delete = None
                        st.rerun()
                with c2:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state.confirm_delete = None
                        st.rerun()
        else:
            st.caption("No previous conversations yet.")
        st.divider()
        # col1, col2 = st.columns(2)
        # with col1:
        # if st.button("🔄 Load", use_container_width=True):
        #     st.session_state.thread_id = current_title
        #     st.session_state.messages = load_thread_history(current_title)
        #     st.rerun()
        # with col2:
        if st.button("🆕 New chat", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
        st.caption(f"Active thread: `{st.session_state.thread_id}`")
        st.caption("Conversation history is persisted to disk (SQLite) — reopen this "
                    "thread ID any time, even after restarting the app.")

    else:
        if st.button("⬅ Back", use_container_width=True):
            st.session_state.show_kb_panel = False
            st.rerun()
    
        if st.button("Customize Knowlege Base", use_container_width=True):
            st.subheader("📚 Long-term Knowledge Base")
            st.caption("Text added here is embedded and stored in ChromaDB, and will be "
                        "retrieved automatically in future chats when relevant.")
            kb_text = st.text_area("Paste text to remember", height=120, key="kb_text")
            if st.button("➕ Add text", use_container_width=True):
                if kb_text.strip():
                    n = add_to_knowledge_base(kb_text)
                    st.success(f"Added chunk to the knowledge base")
                else:
                    st.warning("Nothing to add to the knowledge base")

        uploaded = st.file_uploader("or upload a .txt file", type=["txt"])
        if uploaded is not None and st.button("➕ Add file", use_container_width=True):
            content = uploaded.read().decode("utf-8", errors="ignore")
            n = add_to_knowledge_base(content, source=uploaded.name)
            st.success(f"Added {n} chunk(s) from {uploaded.name}.")
 
    

# Main Content----

st.title("AI Chatbot Platform")

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        for f in msg.get('files',[]):
            st.markdown(f"📎 *{f}*")

prompt = st.chat_input(
    "What do you want to know",
    accept_file="multiple",
    file_type=["jpg","png","jpeg","doc","txt","pdf"]
)

if prompt:
    user_text = prompt.text or ""
    user_files = prompt["files"] if prompt["files"] else []


    config = {'configurable': {'thread_id': st.session_state.thread_id}}
    is_first_message = len(st.session_state.messages) == 0 
    st.session_state.messages.append({'role': 'user', 'content': user_text, 'files': user_files})
    
    with st.chat_message('user'):
        st.markdown(user_text)
        for f in user_files:
            if f.type and f.type.startswith("image/"):
                st.image(f)
            else:
                st.markdown(f"📎 *{f.name}*")

# --- build file_context from non-image attachments ---
    file_context = ""
    for f in user_files:
        if f.type and f.type.startswith("image/"):
            continue  # handled separately below
        if f.name.endswith(".txt") or f.type == "text/plain":
            file_context += f"\n\n[Attached file: {f.name}]\n{f.read().decode('utf-8', errors='ignore')}"
        elif f.type == "application/pdf":
            from pypdf import PdfReader
            reader = PdfReader(f)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            file_context += f"\n\n[Attached file: {f.name}]\n{text}"

    full_message = user_text + file_context
    image_files = [f for f in user_files if f.type and f.type.startswith("image/")]

    with st.chat_message('assistant'):
        with st.spinner("Preparing..."):
            if image_files:
                import base64
                from main import vision_llm
                content = [{"type": "text", "text": user_text}]
                for img in image_files:
                    b64 = base64.b64encode(img.read()).decode("utf-8")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{img.type};base64,{b64}"}
                    })
                response = vision_llm.invoke([{"role": "user", "content": content}]).content
            else:
                result = graph.invoke(
                    {'messages': [{'role': 'user', 'content': full_message}]},
                    config=config,
                )
                response = result['messages'][-1].content

        st.markdown(response)

    st.session_state.messages.append({'role': 'assistant', 'content': response, 'files': []})
    if is_first_message:
        title_source = user_text if user_text.strip() else "image conversation"
        title = generate_title(title_source)
        save_thread_title(st.session_state.thread_id, title)
        st.rerun()

    