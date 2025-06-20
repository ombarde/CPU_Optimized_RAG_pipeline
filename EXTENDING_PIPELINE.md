# Extending and Modifying the CPU-Based RAG Pipeline

This document provides guidance on how to extend or modify the existing Retrieval-Augmented Generation (RAG) pipeline to suit your specific needs.

## Overview of Pipeline Components

The pipeline consists of the following main components:

1. **DocumentProcessor**  
   Handles loading and preprocessing of documents from various formats (TXT, PDF, DOCX, CSV, JSON).  
   - To add support for new document formats, extend the `_extract_content` method and add a new extraction helper.

2. **EmbeddingManager**  
   Manages embedding generation using Sentence Transformers.  
   - To use a different embedding model, update the `embedding_model` parameter in `RAGConfig` or modify `_load_model` to load a custom model.

3. **VectorDatabase**  
   Abstracts vector storage and similarity search using FAISS or ChromaDB.  
   - To add support for other vector databases, implement new initialization, add, and search methods following the existing pattern.

4. **LLMGenerator**  
   Handles response generation using a lightweight causal language model.  
   - To use a different language model, update the `llm_model` parameter in `RAGConfig` or modify `_load_model` accordingly.  
   - Modify prompt construction in `_build_prompt` to customize input to the model.

5. **RAGPipeline**  
   Orchestrates the above components for indexing and querying.

## How to Extend or Modify

### Adding New Document Formats

- Extend `DocumentProcessor._extract_content` to handle new file extensions.  
- Implement helper methods to extract text from the new formats.

### Changing or Adding Embedding Models

- Update `RAGConfig.embedding_model` with the new model name or path.  
- Ensure the model is compatible with Sentence Transformers or modify `EmbeddingManager` to support other frameworks.

### Using Different Vector Databases

- Add new initialization, add, search, save, and load methods in `VectorDatabase`.  
- Update `RAGConfig.vector_db_type` to include the new option.  
- Modify `RAGPipeline` if needed to support the new database.

### Customizing the Language Model

- Update `RAGConfig.llm_model` with the desired model.  
- Modify `LLMGenerator._load_model` to handle model-specific loading requirements.  
- Customize prompt templates in `LLMGenerator._build_prompt` and context building in `_build_context`.

### Adjusting Generation Parameters

- Modify parameters like `max_length`, `temperature`, `do_sample`, and `num_return_sequences` in `RAGConfig` to control generation behavior.

### Adding New Features

- Add new methods or classes as needed, following the modular design.  
- Ensure to update the Streamlit app (`app.py`) to expose new features in the UI.

## Testing and Validation

- After modifications, thoroughly test indexing and querying workflows.  
- Validate embedding quality and retrieval accuracy.  
- Test response generation for relevance and coherence.

## Best Practices

- Keep components modular and loosely coupled.  
- Use logging extensively for debugging and monitoring.  
- Document changes clearly for maintainability.

## Conclusion

This pipeline is designed to be extensible and adaptable. By following the above guidelines, you can tailor it to your specific use cases and improve its capabilities.

For further assistance or contributions, please contact the development team or refer to the project repository.
