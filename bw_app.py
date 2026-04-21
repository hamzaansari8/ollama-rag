import streamlit as st
from rag.retrieval import query_rag

st.set_page_config(
    page_title="Ollama RAG",
    page_icon="📘",
    layout="centered",
)

st.markdown("""
<style>
    .stApp {
        background: #ffffff;
        color: #0f172a;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    .hero {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 45%, #3b82f6 100%);
        padding: 24px 24px;
        border-radius: 18px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.22);
    }

    .hero h1 {
        margin: 0;
        font-size: 30px;
        font-weight: 800;
    }

    .hero p {
        margin: 8px 0 0 0;
        font-size: 15px;
        opacity: 0.95;
    }

    .section-title {
        color: #1d4ed8;
        font-size: 18px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 8px;
    }

    .answer-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 6px solid #2563eb;
        padding: 16px 16px;
        border-radius: 14px;
        color: #0f172a;
        line-height: 1.7;
        white-space: pre-wrap;
    }

    .source-box {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 14px;
        padding: 14px 16px;
        margin-top: 8px;
    }

    .source-item {
        background: #f8fbff;
        border: 1px solid #dbeafe;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 8px;
        color: #1e3a8a;
    }

    .stTextInput > div > div > input {
        border: 2px solid #93c5fd !important;
        border-radius: 12px !important;
        padding: 12px 14px !important;
        box-shadow: none !important;
    }

    .stTextInput > div > div > input:focus {
        border: 2px solid #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }

    .hint {
        color: #64748b;
        font-size: 13px;
        margin-top: 6px;
    }

    hr {
        border: none;
        border-top: 1px solid #dbeafe;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📘Ollama RAG</h1>
    <p>Ask a question from the uploaded PDFs. Enter the question and press Enter.</p>
</div>
""", unsafe_allow_html=True)

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "result" not in st.session_state:
    st.session_state.result = None

if "input_box" not in st.session_state:
    st.session_state.input_box = ""

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

if st.session_state.clear_input:
    st.session_state.input_box = ""
    st.session_state.clear_input = False

query = st.text_input("Your question", key="input_box", placeholder="Type your question here and press Enter")

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Clear"):
        st.session_state.last_query = ""
        st.session_state.result = None
        st.session_state.clear_input = True
        st.rerun()

if query and query.strip() and query.strip() != st.session_state.last_query:
    st.session_state.last_query = query.strip()
    with st.spinner("Searching the PDFs..."):
        st.session_state.result = query_rag(query.strip())

result = st.session_state.result

if result:
    st.markdown("<div class='section-title'>Question</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='source-box'>{st.session_state.last_query}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Answer</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='answer-box'>{result['answer']}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Sources</div>", unsafe_allow_html=True)
    if result["sources"]:
        for s in result["sources"]:
            st.markdown(f"<div class='source-item'>{s}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='source-box'>No relevant sources found</div>", unsafe_allow_html=True)