# 🤖 AI Chatbot Platform

A production-style conversational AI platform built with **LangGraph**, **Groq**, and **Streamlit** — combining short-term conversation memory, long-term semantic memory (RAG), real-time web search, and multimodal (text + image) input in a single agent.


## ✨ Features

- **💬 Multi-turn conversations with persistent memory** — chat history is saved to SQLite via a LangGraph checkpointer, so conversations survive app restarts, not just the current session.
- **🧵 ChatGPT/Claude-style conversation management** — a sidebar list of past chats (newest first), each auto-titled by summarizing the first message, with delete (with confirmation) and new-chat controls.
- **📚 Long-term knowledge base (RAG)** — paste text or upload files to a persistent ChromaDB vector store; relevant chunks are retrieved automatically and injected into context whenever they're relevant to a question.
- **🔎 Real-time web search tool** — the agent autonomously decides when to search the live web (e.g. for current events) using a standard ReAct-style tool-calling loop, rather than always/never searching.
- **🖼️ Multimodal input** — attach images or text/PDF files directly in the chat input; images are routed to a vision-capable model, text/PDF content is extracted and folded into the prompt.
- **⚡ Fast inference via Groq** — runs on Groq's LPU hardware for low-latency responses.

## 🏗️ Architecture

```
┌─────────────┐      ┌───────────────────┐      ┌──────────────┐
│  Streamlit   │ ───▶ │   LangGraph Agent   │ ───▶ │   Groq LLM    │
│   (UI layer) │ ◀─── │  (orchestration)    │ ◀─── │ (model layer) │
└─────────────┘      └─────────┬─────────┘      └──────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌───────────────┐ ┌──────────┐  ┌────────────────┐
        │   ChromaDB     │ │  Web      │  │  SQLite         │
        │ (long-term /   │ │  Search   │  │  (short-term /  │
        │  semantic RAG) │ │  Tool     │  │  chat history)  │
        └───────────────┘ └──────────┘  └────────────────┘
```

Each message flows through the graph, which decides — turn by turn — whether to pull context from the knowledge base, call the web search tool, or respond directly.

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangGraph |
| LLM | Groq (`openai/gpt-oss-120b`, with a separate vision model for images) |
| Long-term memory | ChromaDB |
| Short-term memory | SQLite (LangGraph checkpointer) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Web search | DuckDuckGo (no API key required) |

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Copy the example file and add your [Groq API key](https://console.groq.com):
```bash
cp .env.example .env
```
```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the app
```bash
streamlit run ui.py
```
Then open `http://localhost:8501`.

## 📁 Project Structure

```
.
├── ui.py               # Streamlit UI: chat interface, sidebar, conversation management
├── main.py              # LangGraph agent: LLM, tools, ChromaDB, checkpointing
├── requirements.txt      # Python dependencies
├── .env.example          # Template for required environment variables
└── .gitignore
```

