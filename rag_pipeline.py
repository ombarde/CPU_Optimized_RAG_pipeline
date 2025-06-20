"""
CPU-Based RAG LLM Inference Pipeline
====================================

A complete Retrieval-Augmented Generation pipeline optimized for CPU-only environments.
Supports document embedding, vector storage, similarity search, and LLM inference.

Author: AI Assistant
Date: June 2025
"""

import os
import json
import logging
import numpy as np
import pickle
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Core dependencies
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import chromadb
from chromadb.config import Settings

# Document processing
import PyPDF2
from docx import Document as DocxDocument
import csv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration class for RAG pipeline parameters."""
    
    # Model configurations
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "microsoft/DialoGPT-small"  # Lightweight for demo
    
    # Vector database settings
    vector_db_type: str = "faiss"  # "faiss" or "chroma"
    vector_dim: int = 384  # MiniLM embedding dimension
    
    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float = 0.3
    
    # Generation settings
    max_length: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    num_return_sequences: int = 1
    
    # Performance settings
    device: str = "cpu"
    max_workers: int = 4
    batch_size: int = 16
    
    # Storage paths
    data_dir: str = "./rag_data"
    embeddings_path: str = "./rag_data/embeddings.pkl"
    vector_index_path: str = "./rag_data/vector_index"
    documents_path: str = "./rag_data/documents.json"

class DocumentProcessor:
    """Handles document loading and preprocessing."""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.pdf', '.docx', '.csv', '.json']
    
    def load_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load documents from various file formats."""
        documents = []
        
        for file_path in file_paths:
            try:
                file_path = Path(file_path)
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                content = self._extract_content(file_path)
                if content:
                    doc = {
                        'id': str(len(documents)),
                        'filename': file_path.name,
                        'content': content,
                        'metadata': {
                            'file_type': file_path.suffix,
                            'file_size': file_path.stat().st_size,
                            'created_at': time.time()
                        }
                    }
                    documents.append(doc)
                    logger.info(f"Loaded document: {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        return documents
    
    def _extract_content(self, file_path: Path) -> str:
        """Extract text content from different file types."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            return file_path.read_text(encoding='utf-8')
        
        elif suffix == '.pdf':
            return self._extract_pdf(file_path)
        
        elif suffix == '.docx':
            return self._extract_docx(file_path)
        
        elif suffix == '.csv':
            return self._extract_csv(file_path)
        
        elif suffix == '.json':
            data = json.loads(file_path.read_text(encoding='utf-8'))
            return json.dumps(data, indent=2)
        
        else:
            logger.warning(f"Unsupported file format: {suffix}")
            return ""
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF files."""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX files."""
        doc = DocxDocument(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def _extract_csv(self, file_path: Path) -> str:
        """Extract text from CSV files."""
        text = ""
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                text += " ".join(row) + "\n"
        return text
    
    def chunk_documents(self, documents: List[Dict], chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
        """Split documents into smaller chunks for better retrieval."""
        chunked_docs = []
        
        for doc in documents:
            content = doc['content']
            words = content.split()
            
            if len(words) <= chunk_size:
                chunked_docs.append(doc)
                continue
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_content = " ".join(chunk_words)
                
                chunk_doc = {
                    'id': f"{doc['id']}_chunk_{len(chunked_docs)}",
                    'filename': doc['filename'],
                    'content': chunk_content,
                    'metadata': {
                        **doc['metadata'],
                        'parent_id': doc['id'],
                        'chunk_index': i // (chunk_size - overlap),
                        'is_chunk': True
                    }
                }
                chunked_docs.append(chunk_doc)
        
        return chunked_docs

class EmbeddingManager:
    """Manages document embeddings using sentence transformers."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self.model = SentenceTransformer(self.config.embedding_model)
        self.model.eval()
        
        # Set to CPU explicitly
        if hasattr(self.model, 'to'):
            self.model = self.model.to('cpu')
    
    def embed_documents(self, documents: List[Dict]) -> np.ndarray:
        """Generate embeddings for documents."""
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        
        texts = [doc['content'] for doc in documents]
        
        # Process in batches for memory efficiency
        embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=True,
                convert_to_numpy=True
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.model.encode([query], convert_to_numpy=True)[0]

class VectorDatabase:
    """Vector database abstraction supporting FAISS and ChromaDB."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.index = None
        self.documents = []
        
        if config.vector_db_type == "faiss":
            self._init_faiss()
        elif config.vector_db_type == "chroma":
            self._init_chromadb()
    
    def _init_faiss(self):
        """Initialize FAISS index."""
        logger.info("Initializing FAISS vector database...")
        self.index = faiss.IndexFlatIP(self.config.vector_dim)  # Inner product for cosine similarity
    
    def _init_chromadb(self):
        """Initialize ChromaDB."""
        logger.info("Initializing ChromaDB vector database...")
        self.client = chromadb.Client(Settings(
            persist_directory=self.config.vector_index_path,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection("documents")
    
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        """Add documents and their embeddings to the vector database."""
        self.documents = documents
        
        if self.config.vector_db_type == "faiss":
            self._add_to_faiss(embeddings)
        elif self.config.vector_db_type == "chroma":
            self._add_to_chromadb(documents, embeddings)
    
    def _add_to_faiss(self, embeddings: np.ndarray):
        """Add embeddings to FAISS index."""
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
    
    def _add_to_chromadb(self, documents: List[Dict], embeddings: np.ndarray):
        """Add documents and embeddings to ChromaDB."""
        ids = [doc['id'] for doc in documents]
        contents = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def search(self, query_embedding: np.ndarray, top_k: int = None) -> List[Tuple[Dict, float]]:
        """Search for similar documents."""
        if top_k is None:
            top_k = self.config.top_k
        
        logger.info(f"Performing search with top_k={top_k}")
        
        if self.config.vector_db_type == "faiss":
            results = self._search_faiss(query_embedding, top_k)
        elif self.config.vector_db_type == "chroma":
            results = self._search_chromadb(query_embedding, top_k)
        else:
            results = []
        
        logger.info(f"Search returned {len(results)} results")
        return results
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[Dict, float]]:
        """Search using FAISS index."""
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents) and score > self.config.similarity_threshold:
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def _search_chromadb(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[Dict, float]]:
        """Search using ChromaDB."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        search_results = []
        for i, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
            score = 1 - distance  # Convert distance to similarity
            if score > self.config.similarity_threshold:
                metadata = results['metadatas'][0][i]
                filename = metadata.get('filename') if metadata else None
                doc = {
                    'id': doc_id,
                    'filename': filename,
                    'content': results['documents'][0][i],
                    'metadata': metadata
                }
                search_results.append((doc, score))
        
        return search_results

    def save(self):
        """Save the vector database."""
        os.makedirs(self.config.data_dir, exist_ok=True)
        
        if self.config.vector_db_type == "faiss":
            faiss.write_index(self.index, f"{self.config.vector_index_path}.faiss")
            
            # Save documents separately
            with open(self.config.documents_path, 'w') as f:
                json.dump(self.documents, f, indent=2)
        
        logger.info("Vector database saved successfully")
    
    def load(self):
        """Load the vector database."""
        if self.config.vector_db_type == "faiss":
            index_path = f"{self.config.vector_index_path}.faiss"
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                
                # Load documents
                if os.path.exists(self.config.documents_path):
                    with open(self.config.documents_path, 'r') as f:
                        self.documents = json.load(f)
                
                logger.info("FAISS index loaded successfully")
                return True
        elif self.config.vector_db_type == "chroma":
            # Load ChromaDB collection (assumed persistent)
            try:
                self.collection = self.client.get_collection("documents")
                # Load documents metadata if needed
                # ChromaDB manages persistence internally
                logger.info("ChromaDB collection loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading ChromaDB collection: {str(e)}")
                return False
        
        return False
    
    def save(self):
        """Save the vector database."""
        os.makedirs(self.config.data_dir, exist_ok=True)
        
        if self.config.vector_db_type == "faiss":
            faiss.write_index(self.index, f"{self.config.vector_index_path}.faiss")
            
            # Save documents separately
            with open(self.config.documents_path, 'w') as f:
                json.dump(self.documents, f, indent=2)
        
        logger.info("Vector database saved successfully")
    
    def load(self):
        """Load the vector database."""
        if self.config.vector_db_type == "faiss":
            index_path = f"{self.config.vector_index_path}.faiss"
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                
                # Load documents
                if os.path.exists(self.config.documents_path):
                    with open(self.config.documents_path, 'r') as f:
                        self.documents = json.load(f)
                
                logger.info("FAISS index loaded successfully")
                return True
        
        return False

class LLMGenerator:
    """Lightweight LLM for response generation."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.generator = None
        self._load_model()
    
    def _load_model(self):
        """Load the language model."""
        logger.info(f"Loading LLM: {self.config.llm_model}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Ensure CPU usage
        self.model = self.model.to('cpu')
        self.model.eval()
        
        # Create text generation pipeline
        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,  # CPU
            framework='pt'
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response(self, query: str, retrieved_docs: List[Tuple[Dict, float]]) -> str:
        """Generate response using retrieved context."""
        # Construct prompt with retrieved context
        context = self._build_context(retrieved_docs)
        prompt = self._build_prompt(query, context)
        
        # Generate response
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=self.config.max_length,
                temperature=self.config.temperature,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]['generated_text']
            
            # Extract only the new generated part
            response = generated_text[len(prompt):].strip()
            
            return response if response else "I apologize, but I couldn't generate a proper response based on the available information."
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I encountered an error while generating the response. Please try again."
    
    def _build_context(self, retrieved_docs: List[Tuple[Dict, float]], max_context_length: int = 1000) -> str:
        """Build context from retrieved documents."""
        if not retrieved_docs:
            return "No relevant information found."
        
        context_parts = []
        current_length = 0
        
        for doc, score in retrieved_docs:
            content = doc['content']
            if current_length + len(content) > max_context_length:
                # Truncate the content to fit
                remaining_length = max_context_length - current_length
                content = content[:remaining_length] + "..."
                context_parts.append(content)
                break
            
            context_parts.append(content)
            current_length += len(content)
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the final prompt for generation."""
        prompt = f"""Context: {context}

Question: {query}

Please provide a concise and direct answer to the question above.

Answer:"""
        
        return prompt

class RAGPipeline:
    """Main RAG pipeline orchestrating all components."""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager(self.config)
        self.vector_db = VectorDatabase(self.config)
        self.llm_generator = LLMGenerator(self.config)
        
        # Create data directory
        os.makedirs(self.config.data_dir, exist_ok=True)
        
        logger.info("RAG Pipeline initialized successfully")
    
    def index_documents(self, file_paths: List[str], chunk_documents: bool = True):
        """Index documents into the vector database."""
        logger.info("Starting document indexing process...")
        
        # Load and process documents
        documents = self.doc_processor.load_documents(file_paths)
        if not documents:
            logger.warning("No documents loaded for indexing")
            return
        
        # Chunk documents if requested
        if chunk_documents:
            logger.info("Chunking documents for better retrieval...")
            documents = self.doc_processor.chunk_documents(documents)
        
        # Generate embeddings
        embeddings = self.embedding_manager.embed_documents(documents)
        
        # Add to vector database
        self.vector_db.add_documents(documents, embeddings)
        
        # Save the index
        self.vector_db.save()
        
        logger.info(f"Successfully indexed {len(documents)} document chunks")
    
    def query(self, query: str, return_sources: bool = False) -> Dict[str, Any]:
        """Process a query through the RAG pipeline."""  
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.embed_query(query)
            
            # Retrieve relevant documents
            retrieved_docs = self.vector_db.search(query_embedding)
            
            logger.info(f"Query: {query} - Retrieved {len(retrieved_docs)} documents")
            for doc, score in retrieved_docs:
                logger.info(f"Doc ID: {doc.get('id', 'N/A')}, Filename: {doc.get('filename', 'N/A')}, Score: {score}")
            
            if not retrieved_docs:
                response = "I couldn't find any relevant information to answer your question."
                sources = []
            else:
                # Generate response
                response = self.llm_generator.generate_response(query, retrieved_docs)
                sources = [
                    {
                        'filename': doc['filename'],
                        'content_preview': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                        'similarity_score': float(score)
                    }
                    for doc, score in retrieved_docs
                ]
            
            end_time = time.time()
            
            result = {
                'query': query,
                'response': response,
                'processing_time': end_time - start_time,
                'num_sources': len(retrieved_docs)
            }
            
            if return_sources:
                result['sources'] = sources
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'query': query,
                'response': f"An error occurred while processing your query: {str(e)}",
                'processing_time': time.time() - start_time,
                'num_sources': 0
            }
    
    def load_existing_index(self) -> bool:
        """Load an existing vector database index."""
        return self.vector_db.load()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            'config': {
                'embedding_model': self.config.embedding_model,
                'llm_model': self.config.llm_model,
                'vector_db_type': self.config.vector_db_type,
                'top_k': self.config.top_k
            },
            'vector_db_size': len(self.vector_db.documents) if hasattr(self.vector_db, 'documents') else 0,
            'data_directory': self.config.data_dir
        }
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline
    config = RAGConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="microsoft/DialoGPT-small",
        vector_db_type="faiss",
        top_k=3,
        max_length=256
    )
    
    rag = RAGPipeline(config)
    
    # Example: Index some sample documents (you would replace with actual file paths)
    sample_docs = ["./sample_doc1.txt", "./sample_doc2.pdf"]  # Replace with actual files
    
    # Try to load existing index first
    if not rag.load_existing_index():
        print("No existing index found. You can index documents using:")
        print("rag.index_documents(['path/to/your/documents'])")
    else:
        print("Loaded existing vector database index")
    
    # Example query
    query = "What is the main topic discussed in the documents?"
    result = rag.query(query, return_sources=True)
    
    print(f"Query: {result['query']}")
    print(f"Response: {result['response']}")
    print(f"Processing time: {result['processing_time']:.2f} seconds")
    print(f"Sources found: {result['num_sources']}")
    
    # Print pipeline stats
    stats = rag.get_stats()
    print(f"\nPipeline Stats: {stats}")