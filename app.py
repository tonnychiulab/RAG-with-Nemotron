import streamlit as st
import tempfile
import os
from src.ui import setup_page, render_sidebar, render_chat_message, render_docs_viewer
from src.pipeline import RAGPipeline

# 1. Page Setup
setup_page()

# 2. Sidebar & Configuration
api_key = render_sidebar()

# 3. Main Interface
st.title("📄 Nemotron Document Intelligence")
st.caption("Powered by NVIDIA NIM & Streamlit")

if not api_key:
    st.warning("👈 Please enter your NVIDIA API Key in the sidebar to start.")
    st.stop()

# Initialize Pipeline
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = RAGPipeline(api_key)

# 4. File Upload
uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")

if uploaded_file:
    # Handle file processing
    if 'processed_file' not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        with st.spinner("🚀 Processing document (Extraction -> Embedding -> Indexing)..."):
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Run Pipeline Steps
                chunks = st.session_state.pipeline.extract_text(tmp_path)
                st.session_state.pipeline.build_index(chunks)
                st.session_state.processed_file = uploaded_file.name
                st.success(f"✅ Indexed {len(chunks)} chunks from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                os.remove(tmp_path)

# 5. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    render_chat_message(message["role"], message["content"])

# Handle Input
if prompt := st.chat_input("Ask about your document..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_chat_message("user", prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        if 'processed_file' in st.session_state:
            with st.spinner("🤖 Thinking..."):
                # 1. Retrieval
                retrieved_docs = st.session_state.pipeline.retrieve(prompt)
                
                # 2. Reranking
                reranked_docs = st.session_state.pipeline.rerank(prompt, retrieved_docs)
                
                # Optional: Show context in expander
                render_docs_viewer(reranked_docs)
                
                # 3. Generation
                stream = st.session_state.pipeline.generate_response(prompt, reranked_docs)
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
        else:
            full_response = "Please upload a document first!"
            message_placeholder.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
