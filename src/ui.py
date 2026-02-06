import streamlit as st

def setup_page():
    """Configures the Streamlit page settings."""
    st.set_page_config(
        page_title="Nemotron RAG Pipeline",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_sidebar():
    """Renders the sidebar for API key input and instructions."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "NVIDIA API Key",
            type="password",
            help="Get your key from https://build.nvidia.com/"
        )
        
        st.markdown("---")
        st.markdown("""
        ### 📚 About
        This app demonstrates a **RAG (Retrieval-Augmented Generation)** pipeline using:
        - **Extraction**: `pdfplumber` (Simulating nv-ingest)
        - **Embedding**: `nvidia/llama-nemotron-embed-vl-1b-v2`
        - **Reranking**: `nvidia/llama-nemotron-rerank-vl-1b-v2`
        - **Chat**: `nvidia/llama-3.3-nemotron-super-49b`
        """)
        
        return api_key

def render_chat_message(role, content):
    """Renders a chat message with appropriate styling."""
    with st.chat_message(role):
        st.markdown(content)

def render_docs_viewer(docs):
    """Renders retrieved documents for inspection."""
    if not docs:
        return
        
    with st.expander("🔍 Retrieved Context (Top Chunks)", expanded=False):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}** (Score: {doc.get('score', 0):.4f})")
            st.code(doc.get('content', ''), language='text')
            st.divider()
