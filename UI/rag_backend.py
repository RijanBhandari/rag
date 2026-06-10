import os
import chromadb

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
    set_global_handler,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.postprocessor.flashrank_rerank import FlashRankRerank
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.chroma import ChromaVectorStore

set_global_handler("simple")


def initialize_models():
    llm = Ollama(model="llama3.2:3b", request_timeout=360.0)
    Settings.llm = llm
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    return llm


def embbed_and_store(files):
    initialize_models()
    file_storage = "./files"
    os.makedirs(file_storage, exist_ok=True)

    for file in files:
        with open(os.path.join(file_storage, file.name), "wb") as f:
            f.write(file.getbuffer())
        file.getbuffer().release()

    file_extractor = {".pdf": DoclingReader()}

    documents = SimpleDirectoryReader(
        input_dir=file_storage,
        file_extractor=file_extractor
    ).load_data(show_progress=False)

    db = chromadb.PersistentClient(path="chroma/chroma_db")
    chroma_collection = db.get_or_create_collection("rag_data_collection")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=20)
    current_count = chroma_collection.count()

    if os.path.exists(file_storage) and len(os.listdir(file_storage)) > 0:
        if current_count > 0:
            existing_metadatas = chroma_collection.get(include=["metadatas"])["metadatas"]
            existing_doc_ids = {m.get("doc_id") for m in existing_metadatas if m}

            new_documents = [doc for doc in documents if doc.doc_id not in existing_doc_ids]

            if new_documents:
                for doc in new_documents:
                    VectorStoreIndex.from_documents(
                        [doc],
                        storage_context=storage_context,
                        transformations=[text_splitter],
                    )
        else:
            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[text_splitter],
                show_progress=False,
            )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    return index


def generate_response(index, query, history=None):
    """
    index   : VectorStoreIndex or None
    query   : current user message string
    history : list of {"role": "user"/"assistant", "content": "..."} dicts
              (should NOT include the current user message)
    """
    llm = initialize_models()
    history = history or []

    if index is not None:
        # Build a context-aware prompt by prepending chat history
        history_text = ""
        for m in history:
            role = "User" if m["role"] == "user" else "Assistant"
            history_text += f"{role}: {m['content']}\n"

        full_prompt = f"{history_text}User: {query}" if history_text else query

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=20,
            vector_store_query_mode="hybrid",
            alpha=0.4,
        )
        response_synthesizer = get_response_synthesizer(streaming=True)
        similarity_filter = SimilarityPostprocessor()
        reranker = FlashRankRerank(
            top_n=4,
            providers=["CPUExecutionProvider"]
        )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[similarity_filter, reranker],
        )

        response = query_engine.query(full_prompt)
        return response.response_gen  # string generator

    else:
        # No documents — use proper chat API so conversation history is respected
        chat_messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant.")
        ]
        for m in history:
            role = MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT
            chat_messages.append(ChatMessage(role=role, content=m["content"]))

        chat_messages.append(ChatMessage(role=MessageRole.USER, content=query))

        return llm.stream_chat(chat_messages)  # ChatResponse generator (.delta)


def extract_text_from_chunk(chunk):
    if isinstance(chunk, str):
        return chunk
    elif hasattr(chunk, "delta"):
        return chunk.delta
    elif hasattr(chunk, "text"):
        return chunk.text
    else:
        return ""