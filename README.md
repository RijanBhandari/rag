# 🤖 RAG Document Chat

A simple document chat app — upload a PDF, ask questions, get answers. Runs entirely on free Kaggle GPUs, no local machine required.

Built with LlamaIndex, ChromaDB, Ollama, and Streamlit.

> **Honest disclaimer:** works well for straightforward questions about your documents. Hallucinates on complex or multi-hop ones — it's a local 1–3B model, not GPT-4.

---

## What it does

- Upload a PDF from the chat box and start asking questions about it
- Retrieves relevant chunks with hybrid search (vector + keyword) and reranks them with FlashRank before generating an answer
- Falls back to plain chat (no retrieval) if no document has been uploaded yet
- Streams answers token by token
- Remembers conversation history within a session

## How it works

1. **Ingestion** — uploaded PDFs are parsed with `docling`, split into chunks (`SentenceSplitter`, 256 tokens, 20 overlap), embedded with Ollama's `nomic-embed-text`, and stored in a persistent **ChromaDB** collection.
2. **Retrieval** — queries run through a hybrid vector/keyword retriever (`alpha=0.4`, top 20) over the indexed chunks.
3. **Reranking** — the top candidates are filtered and reordered with **FlashRank**, keeping only the 4 best chunks.
4. **Generation** — Ollama serves `llama3.2:3b` locally, which streams the final answer using the reranked chunks (plus conversation history) as context.
5. **Interface** — a **Streamlit** chat UI handles the upload, the conversation, and token streaming.

## Project structure

```
rag/
├── Llama-index/   # PDF_Chat_using_LLamaIndex_and_Ollama.ipynb — early prototype notebook
├── Rag_app/       # the actual app: Rag_app.py (Streamlit UI) + rag_backend.py (RAG pipeline)
└── UI/            # earlier Streamlit experiment (a plain echo bot), kept for reference
```

👉 **`Rag_app/`** is the real app. See [`Rag_app/README.md`](./Rag_app/README.md) for the full Kaggle setup (installing Ollama, pulling models, setting up ngrok, and launching Streamlit) and a demo video.

## Stack

| Component | Library |
|---|---|
| LLM | Ollama (`llama3.2:3b`) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Vector store | ChromaDB (persistent) |
| PDF parsing | Docling |
| Indexing / retrieval | LlamaIndex (hybrid retrieval) |
| Reranking | FlashRank |
| Frontend | Streamlit |
| Tunnel | pyngrok |

## Why Kaggle

Kaggle notebooks give free access to T4 GPUs, which is enough to run `llama3.2:3b` through Ollama for a personal document Q&A tool — no local GPU or cloud billing needed.

## Limitations

- Answers degrade on questions that require reasoning across multiple document sections
- Session state resets if the Streamlit server restarts
- ChromaDB persists to disk within the Kaggle session but is wiped when the session ends
