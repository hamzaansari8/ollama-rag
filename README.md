# 📄 RAG Pipeline with Ollama + LlamaIndex + ChromaDB

## 🚀 Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline using:
- Local LLM via Ollama (phi3:mini)
- LlamaIndex for orchestration
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Hybrid Retrieval (Vector + BM25)

The system:
- Answers questions from PDFs
- Returns "Not Found" for irrelevant queries
- Uses efficient semantic chunking

---

## 🧠 Architecture
User Query → Hybrid Retriever (Vector + BM25) → Relevant Chunks → LLM (Ollama) → Final Answer

---

## 📦 Tech Stack
- LLM: Ollama (phi3:mini)
- Framework: LlamaIndex
- Vector DB: ChromaDB
- Embeddings: all-MiniLM-L6-v2
- Retrieval: Hybrid (Vector + BM25)
- UI: Streamlit

---

## ⚙️ Installation

1. Clone repo:
git clone https://github.com/HamzaAnsari8/Ollama-RAG.git
cd <project-folder>

2. Create environment:
python -m venv venv
source venv/bin/activate   (Linux/Mac)
venv\Scripts\activate      (Windows)

3. Install dependencies:
pip install -r requirements.txt

4. Install Ollama:
Download: https://ollama.com
ollama pull phi3:mini
ollama serve

---

## 📂 Project Structure
project/
│
├── data/           # PDFs
├── storage/        # ChromaDB
├── ingestion.py    # Ingestion pipeline
├── query.py        # Query pipeline
├── app.py          # Streamlit UI
├── requirements.txt
└── README.md

---

## 📥 Data Ingestion
Run:
python ingestion.py

This will:
- Load PDFs
- Perform semantic chunking
- Generate embeddings
- Store in ChromaDB

---

## 🔍 Query
Run:
python query.py

OR UI:
streamlit run app.py

---

## 🧪 Features
- Hybrid Retrieval (Vector + BM25)
- Semantic Chunking
- Local LLM (no API cost)
- "Not Found" fallback
- Modular pipeline

---

## ⚠️ Important Constraints

1. Ollama is LOCAL ONLY
- Runs on localhost
- Cannot be used in Colab or cloud directly

2. GPU not required
- phi3:mini runs on CPU

3. Do NOT randomly upgrade dependencies
- Can break torch / transformers compatibility

---

## 🔧 Troubleshooting

Import issues:
pip install --upgrade torch transformers

Reset DB:
delete storage/
run ingestion again

Check Ollama:
ollama list
ollama serve

---

## 📈 Future Improvements
- Cross-encoder reranking
- Metadata filtering
- Multi-doc querying
- FastAPI deployment
- Agent workflows

---

## 🧠 Key Learnings
- Hybrid retrieval > pure vector search
- Chunking quality = answer quality
- Local LLM = low cost but infra limits

---

## 📌 Reality Check
This is NOT production-ready.

Missing:
- Logging
- Monitoring
- Evaluation
- Scaling

---

## 👨‍💻 Author
Hamza Ansari