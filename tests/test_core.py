"""Unit tests for the core RAG system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llamaindex_rag import RagSystem, RagConfig


class TestRagSystem:
    """Tests for the RagSystem class."""
    
    @pytest.fixture
    def test_config(self):
        """Create a test configuration with a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = RagConfig(
                storage_dir=temp_path / "storage",
                persist_dir=temp_path / "indices",
            )
            yield config
    
    @pytest.fixture
    def test_docs_dir(self):
        """Create a temporary directory with test documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.txt"
            with open(test_file, "w") as f:
                f.write("This is a test document for LlamaIndex RAG.")
            yield temp_path
    
    def test_init(self, test_config):
        """Test initialization of the RAG system."""
        system = RagSystem(test_config)
        assert system.config == test_config
        assert system._index is None
        assert system.config.storage_dir.exists()
        assert system.config.persist_dir.exists()
    
    @patch("llamaindex_rag.core.VectorStoreIndex")
    @patch("llamaindex_rag.core.ServiceContext")
    def test_load_or_create_index(self, mock_service_context, mock_vector_store_index, test_config):
        """Test load_or_create_index method."""
        # Set up mocks
        mock_service_context.from_defaults.return_value = MagicMock()
        mock_index = MagicMock()
        mock_vector_store_index.from_documents.return_value = mock_index
        
        # Create system
        system = RagSystem(test_config)
        
        # Test creating a new index
        result = system.load_or_create_index("test_index")
        
        assert result == mock_index
        assert system._index == mock_index
        mock_vector_store_index.from_documents.assert_called_once()
        
        # Test that index directory was created
        index_path = system.config.persist_dir / "test_index"
        assert index_path.exists()
    
    @patch("llamaindex_rag.core.SimpleDirectoryReader")
    @patch("llamaindex_rag.core.IngestionPipeline")
    def test_index_documents(self, mock_pipeline, mock_reader, test_config, test_docs_dir):
        """Test index_documents method."""
        # Set up mocks
        mock_reader.return_value.load_data.return_value = ["doc1", "doc2"]
        mock_nodes = [MagicMock(), MagicMock()]
        mock_pipeline.return_value.run.return_value = mock_nodes
        
        # Create system with mock index
        system = RagSystem(test_config)
        system._index = MagicMock()
        
        # Test indexing documents
        system.index_documents(input_dir=test_docs_dir, index_name="test_index")
        
        # Verify calls
        mock_reader.assert_called_once_with(
            input_dir=str(test_docs_dir),
            file_extns=None,
            recursive=True,
        )
        mock_pipeline.return_value.run.assert_called_once_with(documents=["doc1", "doc2"])
        system._index.refresh_ref_docs.assert_called_once_with(mock_nodes)
        system._index.storage_context.persist.assert_called_once()
    
    @patch("llamaindex_rag.core.ChromaVectorStore")
    def test_query(self, mock_chroma_store, test_config):
        """Test query method."""
        # Set up mocks
        mock_query_engine = MagicMock()
        mock_query_engine.query.return_value = "Test response"
        
        mock_index = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine
        
        # Create system with mock index
        system = RagSystem(test_config)
        system._index = mock_index
        
        # Test querying
        response = system.query("Test query", similarity_top_k=3)
        
        # Verify calls
        mock_index.as_query_engine.assert_called_once()
        mock_query_engine.query.assert_called_once_with("Test query")
        assert response == "Test response"
    
    def test_delete_index(self, test_config):
        """Test delete_index method."""
        # Create system
        system = RagSystem(test_config)
        
        # Create a test index directory
        index_path = system.config.persist_dir / "test_index"
        index_path.mkdir(parents=True, exist_ok=True)
        (index_path / "docstore.json").touch()
        
        # Test deleting index
        result = system.delete_index("test_index")
        
        assert result is True
        assert not index_path.exists()
    
    def test_list_indices(self, test_config):
        """Test list_indices method."""
        # Create system
        system = RagSystem(test_config)
        
        # Create test index directories
        index1_path = system.config.persist_dir / "index1"
        index1_path.mkdir(parents=True, exist_ok=True)
        (index1_path / "docstore.json").touch()
        
        index2_path = system.config.persist_dir / "index2"
        index2_path.mkdir(parents=True, exist_ok=True)
        (index2_path / "docstore.json").touch()
        
        # Create a directory without docstore.json (should be ignored)
        empty_path = system.config.persist_dir / "empty"
        empty_path.mkdir(parents=True, exist_ok=True)
        
        # Test listing indices
        indices = system.list_indices()
        
        assert set(indices) == {"index1", "index2"}
        assert "empty" not in indices 