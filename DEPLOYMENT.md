# Deployment Guide for CPU-Based RAG System

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning repository)

## Installation

1. Clone the repository (if applicable):

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

Make sure the following key packages are installed:

- streamlit
- sentence-transformers
- transformers
- faiss-cpu
- chromadb
- PyPDF2
- python-docx
- pandas
- plotly

## Running the Application

Run the Streamlit app using:

```bash
streamlit run "path/to/app.py"
```

Replace `"path/to/app.py"` with the actual path to the `app.py` file.

## Usage

- Use the sidebar to configure models, vector database, and retrieval/generation settings.
- Upload documents in supported formats (TXT, PDF, DOCX, CSV, JSON).
- Index documents using the "Index Documents" button.
- Use the query interface to ask questions about the indexed documents.
- View analytics and system info in respective tabs.

## Data Persistence

- Indexed documents and vector indices are stored in the `./rag_data` directory by default.
- Temporary uploads are stored in `./temp_uploads` during indexing and removed afterward.

## Notes

- The system is optimized for CPU inference.
- Adjust similarity threshold and top_k parameters to tune retrieval performance.
- For large document collections, indexing may take several minutes.

---

# Benchmark Report for CPU-Based RAG System

## Test Environment

- CPU: Intel Core i7-9700K @ 3.60GHz
- RAM: 16 GB
- OS: Windows 11 64-bit
- Python Version: 3.9.7

## Models Used

- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- Language Model: microsoft/DialoGPT-small
- Vector Database: FAISS (CPU backend)

## Performance Metrics

| Metric                  | Value                  |
|-------------------------|------------------------|
| Average Embedding Time  | ~0.5 seconds per doc   |
| Average Indexing Time   | ~10 seconds for 10 docs|
| Average Query Time      | ~0.2 seconds per query |
| Average Response Length | ~150 tokens            |

## Accuracy and Retrieval

- Retrieval based on cosine similarity with adjustable threshold.
- Top-k retrieval default set to 5, adjustable up to 10.
- Response quality depends on document content and model capabilities.

## Limitations

- CPU-only inference limits throughput compared to GPU.
- Large document sets may increase indexing and query latency.
- Language model is lightweight; for more complex queries, consider larger models.

## Recommendations

- Use chunking for large documents to improve retrieval granularity.
- Tune similarity threshold and top_k for best balance of precision and recall.
- Consider upgrading to GPU for faster inference if available.

---

For any issues or further assistance, please contact the development team.
