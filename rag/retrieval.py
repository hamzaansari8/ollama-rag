from pathlib import Path
from typing import List, Tuple, Any, Dict

import chromadb
from rank_bm25 import BM25Okapi

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama


# ---------------------------
# PATHS
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
INDEX_DIR = STORAGE_DIR / "index_store"

COLLECTION_NAME = "rag_docs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

VECTOR_TOP_K = 3
BM25_CANDIDATE_K = 10
MAX_CONTEXT_CHARS_PER_CHUNK = 600   # 🔥 reduced


# ---------------------------
# HELPERS
# ---------------------------
def _get_node_text(item: Any) -> str:
    if hasattr(item, "text"):
        return item.text or ""
    if hasattr(item, "node") and hasattr(item.node, "text"):
        return item.node.text or ""
    return ""


def _get_node_metadata(item: Any) -> dict:
    if hasattr(item, "metadata") and item.metadata:
        return item.metadata
    if hasattr(item, "node") and hasattr(item.node, "metadata") and item.node.metadata:
        return item.node.metadata
    return {}


def _get_node_id(item: Any) -> str:
    if hasattr(item, "node_id"):
        return str(item.node_id)
    if hasattr(item, "node") and hasattr(item.node, "node_id"):
        return str(item.node.node_id)
    return str(id(item))


def _is_not_found_answer(answer: str) -> bool:
    return "not found" in answer.lower()


# ---------------------------
# LOAD INDEX (ONCE)
# ---------------------------
def load_index():
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)

    storage_context = StorageContext.from_defaults(
        persist_dir=str(INDEX_DIR),
        vector_store=vector_store,
    )

    return load_index_from_storage(storage_context)


# 🔥 GLOBAL CACHE
INDEX = load_index()
LLM = Ollama(
    model="phi3:Mini",
    temperature=0,
    request_timeout=600.0,
    context_window=2048,
)


# ---------------------------
# VECTOR RETRIEVAL
# ---------------------------
def vector_retrieve(query: str, top_k: int = VECTOR_TOP_K):
    retriever = INDEX.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)


# ---------------------------
# BM25 RETRIEVAL
# ---------------------------
def bm25_retrieve(query: str, top_k: int = VECTOR_TOP_K):
    base_nodes = vector_retrieve(query, top_k=BM25_CANDIDATE_K)

    if not base_nodes:
        return []

    texts = []
    valid_nodes = []

    for node in base_nodes:
        text = _get_node_text(node).strip()
        if text:
            texts.append(text)
            valid_nodes.append(node)

    if not texts:
        return []

    tokenized_corpus = [text.lower().split() for text in texts]
    tokenized_query = query.lower().split()

    if not tokenized_query:
        return []

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    return [valid_nodes[i] for i in ranked_indices]


# ---------------------------
# HYBRID RETRIEVAL
# ---------------------------
def hybrid_retrieve(query: str, top_k: int = VECTOR_TOP_K):
    vector_nodes = vector_retrieve(query, top_k)
    bm25_nodes = bm25_retrieve(query, top_k)

    combined = vector_nodes + bm25_nodes

    unique_nodes = {}
    for node in combined:
        unique_nodes[_get_node_id(node)] = node

    return list(unique_nodes.values())[:top_k]


# ---------------------------
# CONTEXT + SOURCES
# ---------------------------
def build_context(nodes) -> Tuple[str, List[str]]:
    context_parts = []
    sources = []

    for node in nodes:
        text = _get_node_text(node).strip()
        if not text:
            continue

        metadata = _get_node_metadata(node)

        source = metadata.get("source_file", "unknown")
        page = metadata.get("page_number", "unknown")

        trimmed_text = text[:MAX_CONTEXT_CHARS_PER_CHUNK]

        context_parts.append(
            f"Source: {source} | Page: {page}\n{trimmed_text}"
        )

        sources.append(f"{source} (Page {page})")

    context = "\n\n---\n\n".join(context_parts)
    return context, sorted(set(sources))


# ---------------------------
# LLM
# ---------------------------
def ask_llm(query: str, context: str) -> str:
    prompt = f"""
Answer ONLY exactly from context.
copy the exact format of context (bullet points,numbering,etc)
If not found, say: Not found.

{context}

Q: {query}
A:
""".strip()

    response = LLM.complete(prompt)
    return str(response).strip()


# ---------------------------
# MAIN PIPELINE (UI READY)
# ---------------------------
def query_rag(query: str) -> Dict:
    nodes = hybrid_retrieve(query)

    context, sources = build_context(nodes)

    if not context.strip():
        return {"answer": "Not found in documents", "sources": []}

    answer = ask_llm(query, context)

    if _is_not_found_answer(answer):
        return {"answer": "Not found in documents", "sources": []}

    return {"answer": answer, "sources": sources}


# ---------------------------
# CLI LOOP
# ---------------------------
if __name__ == "__main__":
    while True:
        q = input("\nEnter your question (or 'exit'): ").strip()

        if q.lower() == "exit":
            break

        result = query_rag(q)

        print("\n===== ANSWER =====\n")
        print(result["answer"])

        print("\n===== SOURCES =====\n")
        if result["sources"]:
            for s in result["sources"]:
                print(s)
        else:
            print("No relevant sources found")