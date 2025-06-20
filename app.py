"""
Streamlit Interactive RAG Application
====================================

Interactive web interface for the CPU-based RAG pipeline.
Allows users to upload documents, configure settings, and query the system.

Usage: streamlit run streamlit_app.py
"""

import streamlit as st
import os
import time
import json
import shutil
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

# Import the RAG pipeline (assuming it's in the same directory)
try:
    from rag_pipeline import RAGPipeline, RAGConfig
except ImportError:
    st.error("Please ensure rag_pipeline.py is in the same directory as this Streamlit app.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="CPU-Based RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .source-card {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'indexed_files' not in st.session_state:
    st.session_state.indexed_files = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

def initialize_pipeline():
    """Initialize the RAG pipeline with current configuration."""
    config = RAGConfig(
        embedding_model=st.session_state.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
        llm_model=st.session_state.get('llm_model', 'microsoft/DialoGPT-small'),
        vector_db_type=st.session_state.get('vector_db_type', 'faiss'),
        top_k=st.session_state.get('top_k', 5),
        max_length=st.session_state.get('max_length', 512),
        temperature=st.session_state.get('temperature', 0.7),
        similarity_threshold=st.session_state.get('similarity_threshold', 0.3)
    )
    
    with st.spinner("Initializing RAG Pipeline..."):
        st.session_state.rag_pipeline = RAGPipeline(config)
        
        # Try to load existing index
        if st.session_state.rag_pipeline.load_existing_index():
            st.success("✅ Loaded existing document index!")
            return True
        else:
            st.info("ℹ️ No existing index found. Please upload and index documents first.")
            return False

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🔍 CPU-Based RAG System</div>', unsafe_allow_html=True)
    st.markdown("**Retrieval-Augmented Generation Pipeline optimized for CPU inference**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("Model Settings")
        embedding_model = st.selectbox(
            "Embedding Model",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/paraphrase-MiniLM-L6-v2"
            ],
            key='embedding_model'
        )
        
        llm_model = st.selectbox(
            "Language Model",
            [
                "microsoft/DialoGPT-small",
                "microsoft/DialoGPT-medium",
                "distilgpt2",
                "gpt2"
            ],
            key='llm_model'
        )
        
        # Vector database selection
        vector_db_type = st.selectbox(
            "Vector Database",
            ["faiss", "chroma"],
            key='vector_db_type'
        )
        
        # Retrieval settings
        st.subheader("Retrieval Settings")
        top_k = st.slider("Top K Results", 1, 10, 5, key='top_k')
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.3, step=0.1, key='similarity_threshold')
        
        # Generation settings
        st.subheader("Generation Settings")
        max_length = st.slider("Max Response Length", 128, 1024, 512, step=64, key='max_length')
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, step=0.1, key='temperature')
        
        # Initialize pipeline button
        if st.button("Initialize Pipeline", type="primary"):
            success = initialize_pipeline()
            if success:
                st.rerun()
    
    # Main content area
    if st.session_state.rag_pipeline is None:
        st.warning("Please initialize the pipeline using the sidebar configuration.")
        return
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["Document Management", "Query Interface", "Analytics", "System Info"])
    
    with tab1:
        document_management_tab()
    
    with tab2:
        query_interface_tab()
    
    with tab3:
        analytics_tab()
    
    with tab4:
        system_info_tab()

def document_management_tab():
    """Document upload and indexing interface."""
    st.markdown('<div class="sub-header"> Document Management</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'csv', 'json'],
            help="Supported formats: TXT, PDF, DOCX, CSV, JSON"
        )
        
        chunk_documents = st.checkbox("Chunk documents for better retrieval", value=True)
        
        if uploaded_files:
            # Display uploaded files
            st.write("**Uploaded Files:**")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size:,} bytes)")
            
            if st.button("Index Documents", type="primary"):
                index_uploaded_files(uploaded_files, chunk_documents)
    
    with col2:
        st.subheader("Indexed Documents")
        
        if st.session_state.indexed_files:
            for file_info in st.session_state.indexed_files:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{file_info['name']}</strong><br>
                    <small>Size: {file_info['size']:,} bytes</small><br>
                    <small>Indexed: {file_info['indexed_at']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No documents indexed yet.")
        
        if st.button("Clear Index"):
            clear_document_index()

def index_uploaded_files(uploaded_files, chunk_documents):
    """Index uploaded files into the RAG system."""
    try:
        # Save uploaded files temporarily
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        file_paths = []
        file_info = []
        
        for uploaded_file in uploaded_files:
            # Save file
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            file_paths.append(str(file_path))
            file_info.append({
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'indexed_at': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Index documents
        with st.spinner("Indexing documents... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Processing documents...")
            progress_bar.progress(25)
            
            st.session_state.rag_pipeline.index_documents(file_paths, chunk_documents)
            
            progress_bar.progress(100)
            status_text.text("Indexing complete!")
        
        # Update session state
        st.session_state.indexed_files.extend(file_info)
        
        # Cleanup temp files
        for file_path in file_paths:
            os.remove(file_path)
        temp_dir.rmdir()
        
        st.success(f"Successfully indexed {len(uploaded_files)} documents!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error indexing documents: {str(e)}")

def safe_remove_dir(dir_path: Path):
    """Safely remove a directory and all its contents."""
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)

def clean_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe columns to avoid pyarrow serialization errors."""
    for col in df.columns:
        if df[col].dtype == object:
            # Convert bytes to string if any
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        # Convert all columns to string to avoid mixed types
        df[col] = df[col].astype(str)
    return df

def clear_document_index():
    """Clear the document index."""
    try:
        # Remove index files
        config = st.session_state.rag_pipeline.config
        if os.path.exists(f"{config.vector_index_path}.faiss"):
            os.remove(f"{config.vector_index_path}.faiss")
        if os.path.exists(config.documents_path):
            os.remove(config.documents_path)
        
        st.session_state.indexed_files = []
        st.success("Document index cleared!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error clearing index: {str(e)}")

# Update index_uploaded_files to use safe_remove_dir for temp_dir cleanup
def index_uploaded_files(uploaded_files, chunk_documents):
    """Index uploaded files into the RAG system."""
    try:
        # Save uploaded files temporarily
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        file_paths = []
        file_info = []
        
        for uploaded_file in uploaded_files:
            # Save file
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            file_paths.append(str(file_path))
            file_info.append({
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'indexed_at': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Index documents
        with st.spinner("Indexing documents... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Processing documents...")
            progress_bar.progress(25)
            
            st.session_state.rag_pipeline.index_documents(file_paths, chunk_documents)
            
            progress_bar.progress(100)
            status_text.text("Indexing complete!")
        
        # Update session state
        st.session_state.indexed_files.extend(file_info)
        
        # Cleanup temp files and directory safely
        for file_path in file_paths:
            os.remove(file_path)
        safe_remove_dir(temp_dir)
        
        st.success(f"Successfully indexed {len(uploaded_files)} documents!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error indexing documents: {str(e)}")

# Example of cleaning dataframe before display (modify where dataframes are shown)
# For example, in system_info_tab or analytics_tab, before st.table or st.dataframe calls,
# call clean_dataframe_for_display(df) to avoid ArrowTypeError.

def query_interface_tab():
    """Query interface for interacting with the RAG system."""
    st.markdown('<div class="sub-header">🔍 Query Interface</div>', unsafe_allow_html=True)
    
    if not st.session_state.indexed_files:
        st.warning("Please index some documents first before querying.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Ask a Question")
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="What would you like to know about your documents?"
        )
        
        # Query options
        return_sources = st.checkbox("Show source documents", value=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Submit Query", type="primary", disabled=not query.strip()):
                process_query(query, return_sources)
        
        with col_btn2:
            if st.button("Clear History"):
                st.session_state.query_history = []
                st.rerun()
    
    with col2:
        st.subheader("Query Statistics")
        
        if st.session_state.query_history:
            avg_time = sum(q['processing_time'] for q in st.session_state.query_history) / len(st.session_state.query_history)
            
            st.metric("Total Queries", len(st.session_state.query_history))
            st.metric("Average Response Time", f"{avg_time:.2f}s")
            st.metric("Last Query Time", f"{st.session_state.query_history[-1]['processing_time']:.2f}s")
        else:
            st.info("No queries yet.")
        
        # Show number of indexed documents
        num_indexed = len(st.session_state.indexed_files) if 'indexed_files' in st.session_state else 0
        st.metric("Documents Indexed", num_indexed)
    
    # Display query results
    if st.session_state.query_history:
        st.markdown('<div class="sub-header"> Recent Queries</div>', unsafe_allow_html=True)
        
        # Show last few queries
        for i, query_result in enumerate(reversed(st.session_state.query_history[-3:])):
            with st.expander(f"Query {len(st.session_state.query_history) - i}: {query_result['query'][:50]}..."):
                st.write("**Question:**", query_result['query'])
                st.write("**Answer:**", query_result['response'])
                st.write(f"**Processing Time:** {query_result['processing_time']:.2f} seconds")
                st.write(f"**Sources Found:** {query_result['num_sources']}")
                
                if 'sources' in query_result and query_result['sources']:
                    st.write("**Source Documents:**")
                    for j, source in enumerate(query_result['sources']):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong> {source['filename']}</strong> 
                            <small>(Similarity: {source['similarity_score']:.3f})</small><br>
                            <small>{source['content_preview']}</small>
                        </div>
                        """, unsafe_allow_html=True)

def process_query(query: str, return_sources: bool):
    """Process a user query through the RAG pipeline."""
    try:
        with st.spinner("Processing your query..."):
            result = st.session_state.rag_pipeline.query(query, return_sources=return_sources)
        
        # Add to query history
        st.session_state.query_history.append(result)
        
        # Display results
        st.success("Query processed successfully!")
        
        # Show response
        st.markdown("### Response")
        st.write(result['response'])
        
        # Show metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Processing Time", f"{result['processing_time']:.2f}s")
        with col2:
            st.metric("Sources Found", result['num_sources'])
        with col3:
            st.metric("Query Length", len(query))
        
        # Show sources if requested
        if return_sources and 'sources' in result and result['sources']:
            st.markdown("### Source Documents")
            for i, source in enumerate(result['sources']):
                with st.expander(f"Source {i+1}: {source['filename']} (Score: {source['similarity_score']:.3f})"):
                    st.write(source['content_preview'])
        
        st.rerun()
        
    except Exception as e:
        st.error(f" Error processing query: {str(e)}")

def analytics_tab():
    """Analytics and performance monitoring."""
    st.markdown('<div class="sub-header"> Analytics Dashboard</div>', unsafe_allow_html=True)
    
    if not st.session_state.query_history:
        st.info("No query data available yet. Submit some queries to see analytics.")
        return
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    query_times = [q['processing_time'] for q in st.session_state.query_history]
    source_counts = [q['num_sources'] for q in st.session_state.query_history]
    
    with col1:
        st.metric("Total Queries", len(st.session_state.query_history))
    with col2:
        st.metric("Avg Response Time", f"{sum(query_times)/len(query_times):.2f}s")
    with col3:
        st.metric("Fastest Query", f"{min(query_times):.2f}s")
    with col4:
        st.metric("Avg Sources Found", f"{sum(source_counts)/len(source_counts):.1f}")
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Response Time Trend")
        df_times = pd.DataFrame({
            'Query': range(1, len(query_times) + 1),
            'Response Time (s)': query_times
        })
        df_times = clean_dataframe_for_display(df_times)
        fig_time = px.line(df_times, x='Query', y='Response Time (s)', title="Query Response Times")
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        st.subheader("Sources Retrieved Distribution")
        df_sources = pd.DataFrame({'Sources': source_counts})
        df_sources = clean_dataframe_for_display(df_sources)
        fig_sources = px.histogram(df_sources, x='Sources', title="Distribution of Sources Found")
        st.plotly_chart(fig_sources, use_container_width=True)
    
    # Query length analysis
    query_lengths = [len(q['query']) for q in st.session_state.query_history]
    
    st.subheader("Query Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Query Length Statistics:**")
        st.write(f"- Average: {sum(query_lengths)/len(query_lengths):.1f} characters")
        st.write(f"- Shortest: {min(query_lengths)} characters")
        st.write(f"- Longest: {max(query_lengths)} characters")
    
    with col2:
        df_lengths = pd.DataFrame({'Query Length': query_lengths})
        df_lengths = clean_dataframe_for_display(df_lengths)
        fig_lengths = px.histogram(df_lengths, x='Query Length', title="Query Length Distribution")
        st.plotly_chart(fig_lengths, use_container_width=True)

def system_info_tab():
    """System information and configuration details."""
    st.markdown('<div class="sub-header"> System Information</div>', unsafe_allow_html=True)
    
    if st.session_state.rag_pipeline is None:
        st.warning("Pipeline not initialized.")
        return
    
    # Get system stats
    stats = st.session_state.rag_pipeline.get_stats()
    
    # Configuration section
    st.subheader(" Current Configuration")
    config_df = pd.DataFrame([
        {"Setting": "Embedding Model", "Value": stats['config']['embedding_model']},
        {"Setting": "Language Model", "Value": stats['config']['llm_model']},
        {"Setting": "Vector Database", "Value": stats['config']['vector_db_type']},
        {"Setting": "Top K Results", "Value": stats['config']['top_k']},
    ])
    config_df = clean_dataframe_for_display(config_df)
    st.table(config_df)
    
    # System stats
    st.subheader(" System Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Documents Indexed", stats['vector_db_size'])
    with col2:
        st.metric("Total Queries", len(st.session_state.query_history))
    with col3:
        st.metric("Data Directory", stats['data_directory'])
    
    # Performance info
    st.subheader(" Performance Information")
    st.info("""
    **CPU Optimization Features:**
    - ✅ Sentence Transformers for efficient embeddings
    - ✅ FAISS with CPU backend for fast similarity search
    - ✅ Lightweight language models optimized for CPU
    - ✅ Batch processing for better throughput
    - ✅ Memory-efficient document chunking
    """)
    
    # Export functionality
    st.subheader(" Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(" Export Query History"):
            if st.session_state.query_history:
                df = pd.DataFrame(st.session_state.query_history)
                df = clean_dataframe_for_display(df)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"query_history_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No query history to export.")
    
    with col2:
        if st.button("Export Configuration"):
            config_json = json.dumps(stats['config'], indent=2)
            st.download_button(
                label="Download JSON",
                data=config_json,
                file_name=f"rag_config_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Technical details
    with st.expander("Technical Details"):
        st.code(f"""
RAG Pipeline Architecture:
├── Document Processing
│   ├── Supported formats: TXT, PDF, DOCX, CSV, JSON
│   ├── Document chunking with overlap
│   └── Metadata extraction
├── Embedding Generation
│   ├── Model: {stats['config']['embedding_model']}
│   ├── Device: CPU
│   └── Batch processing enabled
├── Vector Database
│   ├── Type: {stats['config']['vector_db_type'].upper()}
│   ├── Similarity: Cosine similarity
│   └── Index size: {stats['vector_db_size']} documents
└── Response Generation
    ├── Model: {stats['config']['llm_model']}
    ├── Max length: {st.session_state.get('max_length', 512)} tokens
    └── Temperature: {st.session_state.get('temperature', 0.7)}
        """, language="text")

if __name__ == "__main__":
    main()