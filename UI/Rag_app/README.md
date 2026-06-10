# 🤖 Local RAG Chatbot

A simple document chat app — upload a PDF, ask questions, get answers. Runs entirely on free Kaggle GPUs, no local machine required.

Built with LlamaIndex, ChromaDB, Ollama, and Streamlit.

> **Honest disclaimer:** Works well for straightforward questions about your documents. Hallucinates on complex or multi-hop ones — it's a local 1–3B model, not GPT-4.

---

## Demo

https://github.com/user-attachments/assets/a26da8c9-7a63-4a0f-b2bd-48530a7b6c17

---

## What it does

- Upload PDF documents and chat with them
- Retrieves relevant chunks using hybrid search + FlashRank reranking
- Falls back to plain chat (no RAG) if no document is uploaded
- Streams responses token by token
- Remembers conversation history within a session

---

## Running on Kaggle

No GPU? No problem. Kaggle gives you free compute. Follow the steps below in a Kaggle notebook.

### 1. Install dependencies

```python
!pip install -q streamlit pyngrok flashrank --quiet
!pip install llama_index-postprocessor-flashrank_rerank --quiet
!pip install pypdf chromadb llama-index llama-index-vector-stores-chroma llama-index-llms-ollama llama-index-embeddings-ollama docling llama-index-readers-docling --quiet
!sudo apt update --quiet
!sudo apt install -y pciutils --quiet
!sudo apt-get install zstd --quiet
!curl -fsSL https://ollama.com/install.sh | sh
!pip install llama_index-llms-base
```

### 2. Start Ollama in the background

```python
import os
import time
import subprocess

os.environ["OLLAMA_LOG_LEVEL"] = "error"
print("Starting Ollama in the system background...")
global_ollama_process = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    close_fds=True
)
time.sleep(5)
print("🟢 Ollama is running quietly in the background. You can now move to the next cell!")
```

### 3. Pull the required models

```python
%%capture
!ollama pull llama3.2:3b
!ollama pull nomic-embed-text
```

`llama3.2:3b` handles chat. `nomic-embed-text` handles embeddings.

### 4. Add your code files

Paste the contents of `rag_backend.py`:

```python
%%writefile rag_backend.py
# paste rag_backend.py contents here
```

Then paste `Rag_app.py`:

```python
%%writefile Rag_app.py
# paste Rag_app.py contents here
```

### 5. Set up ngrok

You need an ngrok account to expose the Streamlit app. Sign up free at [ngrok.com](https://ngrok.com), grab your auth token, and add it as a Kaggle secret named `NGROK_AUTH_TOKEN`.

Then import it:

```python
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
```

### 6. Launch the app

```python
from pyngrok import ngrok
import subprocess

NGROK_TOKEN = user_secrets.get_secret("NGROK_AUTH_TOKEN")
ngrok.set_auth_token(NGROK_TOKEN)

print("Starting Streamlit app...")
process = subprocess.Popen(["streamlit", "run", "Rag_app.py", "--server.port", "8501"])

public_url = ngrok.connect(8501)
print("\n" + "="*50)
print(f"👉 CLICK THIS URL TO OPEN YOUR APP: {public_url.public_url}")
print("="*50 + "\n")
```

Click the URL that appears and your app is live.

---

## Stack

| Component | Library |
|---|---|
| LLM | Ollama (`llama3.2:3b`) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Vector store | ChromaDB (persistent) |
| Indexing / RAG | LlamaIndex |
| Reranking | FlashRank |
| Frontend | Streamlit |
| Tunnel | pyngrok |

---

## Limitations

- Answers degrade on questions that require reasoning across multiple document sections
- Session state resets if the Streamlit server restarts
- ChromaDB persists to disk within the Kaggle session but is wiped when the session ends
