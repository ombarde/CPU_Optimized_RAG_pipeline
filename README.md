# CPU-Based Retrieval-Augmented Generation (RAG) System

This project implements a CPU-optimized Retrieval-Augmented Generation (RAG) pipeline with a Streamlit web interface. It allows users to upload documents, index them, and query the system to get AI-generated answers based on the indexed content.

## Features

- Document upload and indexing for multiple formats: TXT, PDF, DOCX, CSV, JSON
- Embedding generation using Sentence Transformers
- Vector similarity search using FAISS or ChromaDB
- Lightweight language model inference using Hugging Face transformers
- Configurable retrieval and generation parameters via UI
- Analytics dashboard for query performance monitoring
- Export functionality for query history and configuration
- Modular and extensible pipeline design

## Installation

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed installation and running instructions.

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Use the sidebar to configure models and retrieval settings.

3. Upload documents and index them.

4. Enter queries to get AI-generated answers based on your documents.

5. View analytics and system info in the respective tabs.

## Extending the Pipeline

Refer to [EXTENDING_PIPELINE.md](./EXTENDING_PIPELINE.md) for guidance on modifying or extending the pipeline components.

## Requirements

See [requirements.txt](./requirements.txt) for the list of Python dependencies.

## Deployment

Refer to [DEPLOYMENT.md](./DEPLOYMENT.md) for deployment and benchmark details.

## License

This project is provided as-is for educational and demonstration purposes.
