"""Configuration module for the LlamaIndex RAG system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from llama_index.embeddings.base import BaseEmbedding
from llama_index.llms.base import LLM
from llama_index.vector_stores.types import VectorStore


@dataclass
class RagConfig:
    """Configuration for the RAG system.
    
    This class holds all configuration parameters for the RagSystem, including
    models, vector stores, chunking strategies, and other settings.
    """
    
    # Storage settings
    storage_dir: Path = field(default_factory=lambda: Path("./data"))
    persist_dir: Path = field(default_factory=lambda: Path("./data/indices"))
    
    # Models and stores
    embedding_model: Optional[BaseEmbedding] = None
    llm: Optional[LLM] = None 
    vector_store: Optional[VectorStore] = None
    
    # Chunking settings
    chunk_size: int = 1024
    chunk_overlap: int = 200
    
    # Retrieval settings
    top_k: int = 4
    similarity_top_k: int = 4
    
    # Advanced settings
    use_hybrid_search: bool = False
    hybrid_search_kwargs: Dict[str, Any] = field(default_factory=dict)
    reranker_model: Optional[str] = None
    
    # System settings
    cache_dir: Optional[Path] = None
    debug: bool = False
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        self.persist_dir.mkdir(exist_ok=True, parents=True)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RagConfig":
        """Create a configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            A RagConfig instance
        """
        # Convert string paths to Path objects
        for key in ["storage_dir", "persist_dir", "cache_dir"]:
            if key in config_dict and isinstance(config_dict[key], str):
                config_dict[key] = Path(config_dict[key])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the config
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, (BaseEmbedding, LLM, VectorStore)):
                # Skip complex objects
                continue
            else:
                config_dict[key] = value
        
        return config_dict 