from __future__ import annotations

import shutil
from pathlib import Path
from typing import List
import chromadb
import fitz  
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
INDEX_DIR = STORAGE_DIR / "index_store"

COLLECTION_NAME = "rag_docs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REBUILD = True

def load_pdf_as_documents(pdf_path: Path) -> List[Document]:
    docs: List[Document] = []

    with fitz.open(str(pdf_path)) as pdf:
        for page_index in range(len(pdf)):
            page = pdf.load_page(page_index)
            text = page.get_text("text").strip()

            if not text:
                continue

            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source_file": pdf_path.name,
                        "page_number": page_index + 1,
                    },
                )
            )

    return docs

def load_all_documents() -> List[Document]:
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {DATA_DIR}")

    all_docs: List[Document] = []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")
        docs = load_pdf_as_documents(pdf_file)
        print(f"  Pages with text: {len(docs)}")
        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("No text extracted from PDFs.")

    return all_docs

def build_index() -> None:
    if REBUILD and STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Embedding model
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    # Load docs
    documents = load_all_documents()
    print(f"Total loaded documents: {len(documents)}")

    # HYBRID CHUNKING START

    # Step 1: Semantic splitting
    semantic_splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=1,
        breakpoint_percentile_threshold=75,
    )

    semantic_nodes = semantic_splitter.get_nodes_from_documents(documents)

    # Step 2: Fallback splitting (for large chunks)
    fallback_splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )

    final_nodes = []

    for node in semantic_nodes:
        # If chunk is too large → split again
        if len(node.text) > 1000:
            smaller_nodes = fallback_splitter.get_nodes_from_documents([node])
            final_nodes.extend(smaller_nodes)
        else:
            final_nodes.append(node)

    nodes = final_nodes

    print(f"Total hybrid chunks created: {len(nodes)}")

    # Add metadata
    for i, node in enumerate(nodes):
        meta = dict(node.metadata or {})
        meta["chunk_id"] = f"chunk_{i:06d}"
        meta["source_file"] = meta.get("source_file", "unknown")
        meta["page_number"] = meta.get("page_number", "unknown")
        node.metadata = meta

    # ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index
    _ = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    storage_context.persist(persist_dir=str(INDEX_DIR))

    print("\nIngestion completed successfully.")
    print(f"Chroma stored in: {CHROMA_DIR}")
    print(f"Index stored in: {INDEX_DIR}")

if __name__ == "__main__":
    build_index()