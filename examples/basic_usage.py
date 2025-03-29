#!/usr/bin/env python3
"""
Basic Usage Example for LlamaIndex RAG

This example demonstrates how to:
1. Initialize the RAG system
2. Index documents from a directory
3. Query the system
"""

import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import LlamaIndex RAG
from src.llamaindex_rag import RagSystem, RagConfig
from src.llamaindex_rag.utils import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run the basic usage example."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OPENAI_API_KEY environment variable not set")
        logger.info("Please set your OpenAI API key in the .env file or export it")
        logger.info("Example .env file content: OPENAI_API_KEY=your_key_here")
        return
    
    # Initialize the RAG system with default configuration
    logger.info("Initializing RAG system...")
    config = RagConfig(
        storage_dir=Path("./data"),
        chunk_size=512,
        chunk_overlap=50,
    )
    rag = RagSystem(config)
    
    # Define the documents directory
    docs_dir = project_root / "examples" / "documents"
    
    # Create example directory and sample document if it doesn't exist
    if not docs_dir.exists():
        logger.info(f"Creating example documents directory: {docs_dir}")
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a sample document
        sample_doc = docs_dir / "sample.txt"
        with open(sample_doc, "w") as f:
            f.write("""
# LlamaIndex RAG

LlamaIndex is a data framework for LLM applications to ingest, structure, and access private or domain-specific data.

## Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models by retrieving relevant information from external data sources before generating a response.

### Benefits of RAG:
1. **Reduced Hallucinations**: By grounding responses in factual information
2. **Access to Private Data**: Can leverage internal documents and databases
3. **Up-to-date Information**: Can incorporate recent information not in the model's training data
4. **Transparency**: Can cite sources for claims made in responses

### Key Components of RAG:
1. **Document Loading**: Ingesting documents from various sources
2. **Text Chunking**: Breaking documents into manageable pieces
3. **Embedding Generation**: Converting text chunks into vector representations
4. **Vector Storage**: Storing embeddings for efficient retrieval
5. **Query Processing**: Finding relevant information for user queries
6. **Response Synthesis**: Generating coherent responses using retrieved context
""")
    
    # Index the documents
    logger.info(f"Indexing documents from {docs_dir}...")
    rag.index_documents(
        input_dir=docs_dir,
        index_name="example_index",
    )
    
    # Show available indices
    indices = rag.list_indices()
    logger.info(f"Available indices: {indices}")
    
    # Ask some questions
    queries = [
        "What is LlamaIndex?",
        "What are the key components of RAG?",
        "What are the benefits of using RAG?",
    ]
    
    for query in queries:
        logger.info(f"\nQuery: {query}")
        response = rag.query(
            query_text=query,
            index_name="example_index",
        )
        
        print(f"\nAnswer: {response}")
        
        # Print source documents if available
        if hasattr(response, "source_nodes") and response.source_nodes:
            print("\nSources:")
            for i, node in enumerate(response.source_nodes):
                metadata = node.node.metadata
                source = metadata.get("file_name", "Unknown source")
                print(f"{i+1}. {source}")


if __name__ == "__main__":
    main() 