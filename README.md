# LlamaIndex RAG

A powerful Retrieval-Augmented Generation (RAG) system built with LlamaIndex for enhancing large language model responses with external knowledge.

## Features

- **Document Processing**: Ingest and process documents from various sources (PDF, TXT, CSV, HTML, etc.)
- **Flexible Indexing**: Create and maintain vector indices for efficient retrieval
- **Advanced RAG Patterns**: Implement query transformations, multi-step retrieval, and reranking
- **Multi-modal Support**: Process both text and image content (where applicable)
- **Multiple Interfaces**: CLI, API, and Web UI for versatile usage
- **Customizable Components**: Easily swap embedding models, LLMs, and vector stores

## Quick Start

### Installation

```bash
pip install llamaindex-rag
```

Or install directly from the repository:

```bash
pip install git+https://github.com/yourusername/llamaindex-rag.git
```

### Environment Setup

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
# Add other API keys as needed
```

### Basic Usage

```python
from llamaindex_rag import RagSystem
from llamaindex_rag.config import RagConfig

# Initialize the system with default configuration
config = RagConfig()
rag = RagSystem(config)

# Index some documents
rag.index_documents("path/to/documents")

# Ask a question
response = rag.query("What is retrieval-augmented generation?")
print(response)
```

## Command Line Interface

The package provides a command-line interface for common operations:

```bash
# Index documents
llamarag index --input-dir path/to/documents --index-name my_index

# Query the system
llamarag query "What is retrieval-augmented generation?" --index-name my_index

# Start the web UI
llamarag webapp
```

## Components

### Document Processing

- Document loading from various file types
- Text chunking with configurable strategies
- Metadata extraction
- Pre-processing and cleaning

### Indexing

- Vector store integration (ChromaDB by default)
- Multiple index types support
- Hybrid search capabilities
- Index persistence and management

### Retrieval

- Various retrieval strategies (semantic, keyword, hybrid)
- Context window management
- Re-ranking capabilities
- Configurable top-k

### Generation

- Integration with different LLMs
- Prompting strategies
- Response formatting
- Citation tracking

## Advanced Usage

### Custom Embedding Models

```python
from llamaindex_rag import RagSystem
from llamaindex_rag.config import RagConfig
from llamaindex_rag.embeddings import HuggingFaceEmbeddings

# Use a custom embedding model
config = RagConfig(
    embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
)
rag = RagSystem(config)
```

### Custom Retrieval Pipeline

```python
from llamaindex_rag import RagSystem
from llamaindex_rag.config import RagConfig
from llamaindex_rag.retrieval import SentenceSplitter, HybridRetriever

# Configure a custom retrieval pipeline
config = RagConfig(
    chunking_strategy=SentenceSplitter(chunk_size=512, overlap=50),
    retriever=HybridRetriever(
        vector_weight=0.7, 
        keyword_weight=0.3,
        top_k=5
    )
)
rag = RagSystem(config)
```

## Development

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llamaindex-rag.git
   cd llamaindex-rag
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Testing

Run the test suite:

```bash
pytest
```

Or with coverage:

```bash
pytest --cov=llamaindex_rag
```

### Code Quality

Format code with Black and isort:

```bash
black src tests
isort src tests
```

Run static type checking:

```bash
mypy src
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for providing the core RAG framework
- All the open-source libraries this project depends on 