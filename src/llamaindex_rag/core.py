"""Core implementation of the RAG system."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.query.schema import QueryMode
from llama_index.ingestion import IngestionPipeline
from llama_index.llms.openai import OpenAI
from llama_index.node_parser import SentenceSplitter
from llama_index.readers.file import SimpleDirectoryReader
from llama_index.response_synthesizers import CompactAndRefine
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

from llamaindex_rag.config import RagConfig

logger = logging.getLogger(__name__)


class RagSystem:
    """Retrieval-Augmented Generation system built with LlamaIndex.
    
    This class encapsulates the complete RAG pipeline, from document ingestion
    to query processing and response generation.
    """
    
    def __init__(self, config: Optional[RagConfig] = None):
        """Initialize the RAG system.
        
        Args:
            config: Configuration for the RAG system
        """
        self.config = config or RagConfig()
        self._initialize_components()
        self._index = None
    
    def _initialize_components(self) -> None:
        """Initialize system components based on configuration."""
        # Set up embedding model
        if self.config.embedding_model is None:
            self.config.embedding_model = OpenAIEmbedding()
        
        # Set up LLM
        if self.config.llm is None:
            self.config.llm = OpenAI(temperature=0.1)
        
        # Set up node parser for text chunking
        self.node_parser = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Set up vector store if not provided
        if self.config.vector_store is None:
            chroma_client = chromadb.PersistentClient(
                path=str(self.config.persist_dir / "chroma")
            )
            chroma_collection = chroma_client.get_or_create_collection("documents")
            self.config.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Set up service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.config.llm,
            embed_model=self.config.embedding_model,
            node_parser=self.node_parser,
        )
        
        # Set up response synthesizer
        self.response_synthesizer = CompactAndRefine(
            service_context=self.service_context,
            streaming=True,
        )
    
    def load_or_create_index(self, index_name: str = "default") -> VectorStoreIndex:
        """Load an existing index or create a new one.
        
        Args:
            index_name: Name of the index to load or create
            
        Returns:
            The loaded or created index
        """
        index_path = self.config.persist_dir / index_name
        
        try:
            # Try to load existing index
            if index_path.exists():
                logger.info(f"Loading existing index from {index_path}")
                storage_context = StorageContext.from_defaults(
                    vector_store=self.config.vector_store,
                    persist_dir=str(index_path),
                )
                index = load_index_from_storage(storage_context)
                logger.info("Index loaded successfully")
                self._index = index
                return index
        except Exception as e:
            logger.warning(f"Failed to load index: {e}. Creating a new one.")
        
        # Create a new index
        logger.info("Creating a new index")
        storage_context = StorageContext.from_defaults(
            vector_store=self.config.vector_store
        )
        index = VectorStoreIndex.from_documents(
            [],
            storage_context=storage_context,
            service_context=self.service_context,
        )
        self._index = index
        
        # Persist the index
        index_path.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(index_path))
        logger.info(f"New index created and persisted to {index_path}")
        
        return index
    
    def index_documents(
        self, 
        input_dir: Union[str, Path], 
        index_name: str = "default",
        file_extns: Optional[List[str]] = None,
    ) -> None:
        """Index documents from a directory.
        
        Args:
            input_dir: Directory containing documents to index
            index_name: Name of the index to use or create
            file_extns: List of file extensions to include (default: all)
        """
        input_dir = Path(input_dir) if isinstance(input_dir, str) else input_dir
        if not input_dir.exists():
            raise ValueError(f"Input directory '{input_dir}' does not exist")
        
        # Load documents
        logger.info(f"Loading documents from {input_dir}")
        reader = SimpleDirectoryReader(
            input_dir=str(input_dir),
            file_extns=file_extns,
            recursive=True,
        )
        documents = reader.load_data()
        logger.info(f"Loaded {len(documents)} documents")
        
        # Load or create index
        index = self.load_or_create_index(index_name)
        
        # Index documents
        logger.info("Indexing documents...")
        pipeline = IngestionPipeline(
            transformations=[self.node_parser],
            vector_store=self.config.vector_store,
        )
        nodes = pipeline.run(documents=documents)
        
        # Update index with new nodes
        index.refresh_ref_docs(nodes)
        
        # Persist the updated index
        index_path = self.config.persist_dir / index_name
        index.storage_context.persist(persist_dir=str(index_path))
        logger.info(f"Indexing complete, stored in {index_path}")
    
    def query(
        self, 
        query_text: str,
        index_name: str = "default",
        mode: str = "default",
        similarity_top_k: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Query the RAG system.
        
        Args:
            query_text: The query text
            index_name: Name of the index to query
            mode: Query mode (default, tree, summarize)
            similarity_top_k: Number of similar documents to retrieve
            **kwargs: Additional query parameters
            
        Returns:
            The query response
        """
        # Load index if not already loaded
        if self._index is None:
            self._index = self.load_or_create_index(index_name)
        
        # Set up query mode
        if mode == "tree":
            query_mode = QueryMode.DEFAULT
        elif mode == "summarize":
            query_mode = QueryMode.SUMMARIZE
        else:
            query_mode = QueryMode.DEFAULT
        
        # Set up query parameters
        if similarity_top_k is None:
            similarity_top_k = self.config.similarity_top_k
        
        # Create query engine
        query_engine = self._index.as_query_engine(
            response_synthesizer=self.response_synthesizer,
            similarity_top_k=similarity_top_k,
            node_postprocessors=kwargs.get("node_postprocessors", []),
            **kwargs,
        )
        
        # Execute query
        logger.info(f"Executing query: {query_text}")
        response = query_engine.query(query_text)
        
        return response
    
    def delete_index(self, index_name: str) -> bool:
        """Delete an index.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        index_path = self.config.persist_dir / index_name
        if not index_path.exists():
            logger.warning(f"Index {index_name} does not exist")
            return False
        
        try:
            import shutil
            shutil.rmtree(index_path)
            logger.info(f"Index {index_name} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete index {index_name}: {e}")
            return False
    
    def list_indices(self) -> List[str]:
        """List all available indices.
        
        Returns:
            List of index names
        """
        return [
            path.name for path in self.config.persist_dir.iterdir()
            if path.is_dir() and (path / "docstore.json").exists()
        ] 