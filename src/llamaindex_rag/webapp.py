"""Streamlit web UI for the LlamaIndex RAG system."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from streamlit.components.v1 import html
from streamlit.runtime.uploaded_file_manager import UploadedFile

from llamaindex_rag import RagSystem, RagConfig
from llamaindex_rag.utils import (
    load_dotenv, 
    human_readable_size, 
    check_api_keys,
    get_available_devices,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up page configuration
st.set_page_config(
    page_title="LlamaIndex RAG",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
    }
    .sources-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .source-item {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .highlighted {
        background-color: #ffff00;
        padding: 0px 2px;
        border-radius: 3px;
    }
    .stTextArea textarea {
        height: 200px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.rag_system = None
    st.session_state.query_history = []
    st.session_state.response_history = []
    st.session_state.tab = "query"
    st.session_state.active_index = "default"
    st.session_state.uploaded_files = []


def initialize_rag_system() -> RagSystem:
    """Initialize the RAG system."""
    with st.spinner("Initializing RAG system..."):
        config = RagConfig()
        rag_system = RagSystem(config)
        st.session_state.initialized = True
        st.session_state.rag_system = rag_system
        return rag_system


def get_rag_system() -> RagSystem:
    """Get or initialize the RAG system."""
    if st.session_state.initialized and st.session_state.rag_system is not None:
        return st.session_state.rag_system
    return initialize_rag_system()


def render_sidebar() -> None:
    """Render the sidebar."""
    st.sidebar.markdown("## 🦙 LlamaIndex RAG")
    st.sidebar.markdown("---")
    
    # API keys status
    api_keys = check_api_keys()
    st.sidebar.markdown("### API Keys")
    
    for service, available in api_keys.items():
        icon = "✅" if available else "❌"
        st.sidebar.markdown(f"{icon} {service.capitalize()}")
    
    if not api_keys["openai"]:
        st.sidebar.warning(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        )
    
    # Device information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Devices")
    
    devices = get_available_devices()
    for device, available in devices.items():
        icon = "✅" if available else "❌"
        st.sidebar.markdown(f"{icon} {device.upper()}")
    
    # Available indices
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Indices")
    
    rag_system = get_rag_system()
    indices = rag_system.list_indices()
    
    if not indices:
        st.sidebar.info("No indices available. Please create one first.")
    else:
        active_index = st.sidebar.selectbox(
            "Select active index",
            indices,
            index=indices.index(st.session_state.active_index) if st.session_state.active_index in indices else 0,
        )
        st.session_state.active_index = active_index
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    
    tabs = {
        "query": "🔍 Query",
        "upload": "📤 Upload Documents",
        "manage": "🛠️ Manage Indices",
        "settings": "⚙️ Settings",
    }
    
    for tab_id, tab_name in tabs.items():
        if st.sidebar.button(tab_name, key=f"tab_{tab_id}"):
            st.session_state.tab = tab_id
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "LlamaIndex RAG is a powerful Retrieval-Augmented Generation system "
        "built with LlamaIndex for enhancing large language model responses with external knowledge."
    )
    st.sidebar.markdown(
        "[GitHub](https://github.com/yourusername/llamaindex-rag) | "
        "[Documentation](https://github.com/yourusername/llamaindex-rag#readme)"
    )


def query_tab() -> None:
    """Render the query tab."""
    st.markdown("<h1 class='main-header'>🔍 Ask Your Documents</h1>", unsafe_allow_html=True)
    
    rag_system = get_rag_system()
    indices = rag_system.list_indices()
    
    if not indices:
        st.warning("No indices available. Please upload and index some documents first.")
        if st.button("Go to Upload"):
            st.session_state.tab = "upload"
        return
    
    # Query input
    query = st.text_input("Enter your question:", key="query_input")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        similarity_top_k = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=10,
            value=4,
        )
    
    with col2:
        mode = st.selectbox(
            "Query mode",
            ["default", "summarize", "tree"],
            index=0,
        )
    
    with col3:
        st.markdown("&nbsp;")  # Spacer
        submit = st.button("Submit", key="submit_query", use_container_width=True)
    
    # Process query
    if submit and query:
        with st.spinner("Processing query..."):
            try:
                response = rag_system.query(
                    query_text=query,
                    index_name=st.session_state.active_index,
                    mode=mode,
                    similarity_top_k=similarity_top_k,
                )
                
                # Add to history
                st.session_state.query_history.append(query)
                st.session_state.response_history.append(response)
                
                # Display response
                st.markdown("### Answer")
                st.markdown(str(response))
                
                # Display sources
                if hasattr(response, "source_nodes") and response.source_nodes:
                    st.markdown("<div class='sources-header'>Sources</div>", unsafe_allow_html=True)
                    
                    for i, node in enumerate(response.source_nodes):
                        metadata = node.node.metadata
                        source = metadata.get("file_name", "Unknown source")
                        text = node.node.text
                        
                        with st.expander(f"Source {i+1}: {source}"):
                            st.text(text[:1000] + ("..." if len(text) > 1000 else ""))
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Query history
    if st.session_state.query_history:
        st.markdown("---")
        st.markdown("### Recent Queries")
        
        for i, (q, r) in enumerate(zip(
            reversed(st.session_state.query_history[-5:]), 
            reversed(st.session_state.response_history[-5:])
        )):
            with st.expander(f"Q: {q}"):
                st.markdown("**Answer:**")
                st.markdown(str(r))


def upload_tab() -> None:
    """Render the upload tab."""
    st.markdown("<h1 class='main-header'>📤 Upload Documents</h1>", unsafe_allow_html=True)
    
    rag_system = get_rag_system()
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload documents to index",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "docx", "csv", "html"],
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        
        # Display uploaded files
        st.markdown("### Uploaded Files")
        
        files_info = []
        for file in uploaded_files:
            files_info.append({
                "name": file.name,
                "type": file.type,
                "size": human_readable_size(file.size),
            })
        
        # Create a temp directory for uploaded files
        import tempfile
        import shutil
        
        temp_dir = Path(tempfile.mkdtemp())
        
        # Index name input
        index_name = st.text_input(
            "Index name",
            value=st.session_state.active_index or "default",
        )
        
        # Index button
        if st.button("Index Documents", key="index_docs"):
            with st.spinner("Indexing documents..."):
                try:
                    # Save uploaded files to temp directory
                    for file in uploaded_files:
                        file_path = temp_dir / file.name
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                    
                    # Index documents
                    rag_system.index_documents(
                        input_dir=temp_dir,
                        index_name=index_name,
                    )
                    
                    # Update active index
                    st.session_state.active_index = index_name
                    
                    st.success(f"Successfully indexed {len(uploaded_files)} documents into '{index_name}'")
                    
                    # Clear uploaded files
                    st.session_state.uploaded_files = []
                    
                    # Remove temp directory
                    shutil.rmtree(temp_dir)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    # Remove temp directory on error
                    shutil.rmtree(temp_dir)
    else:
        st.info(
            "Upload documents to index them. Supported formats: PDF, TXT, MD, DOCX, CSV, HTML."
        )


def manage_tab() -> None:
    """Render the manage indices tab."""
    st.markdown("<h1 class='main-header'>🛠️ Manage Indices</h1>", unsafe_allow_html=True)
    
    rag_system = get_rag_system()
    indices = rag_system.list_indices()
    
    if not indices:
        st.warning("No indices available. Please upload and index some documents first.")
        if st.button("Go to Upload"):
            st.session_state.tab = "upload"
        return
    
    # List indices
    st.markdown("### Available Indices")
    
    for index_name in indices:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            index_path = rag_system.config.persist_dir / index_name
            size = sum(
                f.stat().st_size for f in index_path.glob("**/*") if f.is_file()
            )
            st.markdown(f"**{index_name}** ({human_readable_size(size)})")
        
        with col2:
            if st.button("Delete", key=f"delete_{index_name}"):
                if st.session_state.active_index == index_name:
                    # Set active index to another one if available
                    other_indices = [i for i in indices if i != index_name]
                    if other_indices:
                        st.session_state.active_index = other_indices[0]
                    else:
                        st.session_state.active_index = "default"
                
                with st.spinner(f"Deleting index '{index_name}'..."):
                    success = rag_system.delete_index(index_name)
                    
                    if success:
                        st.success(f"Index '{index_name}' deleted successfully")
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to delete index '{index_name}'")
    
    # Create new index
    st.markdown("---")
    st.markdown("### Create New Index")
    
    new_index_name = st.text_input("New index name")
    
    if st.button("Create", key="create_index") and new_index_name:
        with st.spinner(f"Creating index '{new_index_name}'..."):
            try:
                rag_system.load_or_create_index(new_index_name)
                st.success(f"Index '{new_index_name}' created successfully")
                st.session_state.active_index = new_index_name
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")


def settings_tab() -> None:
    """Render the settings tab."""
    st.markdown("<h1 class='main-header'>⚙️ Settings</h1>", unsafe_allow_html=True)
    
    # System information
    st.markdown("### System Information")
    
    rag_system = get_rag_system()
    config = rag_system.config
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Storage Directory**")
        st.code(str(config.storage_dir))
        
        st.markdown("**Persist Directory**")
        st.code(str(config.persist_dir))
        
        st.markdown("**Chunking**")
        st.code(f"Chunk Size: {config.chunk_size}\nChunk Overlap: {config.chunk_overlap}")
    
    with col2:
        st.markdown("**Retrieval Settings**")
        st.code(f"Top K: {config.top_k}\nSimilarity Top K: {config.similarity_top_k}")
        
        st.markdown("**Advanced Settings**")
        st.code(f"Hybrid Search: {config.use_hybrid_search}")
        
        # Available API keys
        st.markdown("**API Keys**")
        api_keys = check_api_keys()
        api_keys_str = "\n".join([f"{k.capitalize()}: {'✅' if v else '❌'}" for k, v in api_keys.items()])
        st.code(api_keys_str)
    
    # Clear data
    st.markdown("---")
    st.markdown("### Danger Zone")
    
    if st.button("Clear All Indices", key="clear_indices"):
        confirm = st.checkbox("I understand this will delete all indices and cannot be undone")
        
        if confirm:
            with st.spinner("Deleting all indices..."):
                indices = rag_system.list_indices()
                for index_name in indices:
                    rag_system.delete_index(index_name)
                
                st.session_state.active_index = "default"
                st.success("All indices deleted successfully")
                st.experimental_rerun()


def main():
    """Main function."""
    render_sidebar()
    
    if st.session_state.tab == "query":
        query_tab()
    elif st.session_state.tab == "upload":
        upload_tab()
    elif st.session_state.tab == "manage":
        manage_tab()
    elif st.session_state.tab == "settings":
        settings_tab()


if __name__ == "__main__":
    main() 