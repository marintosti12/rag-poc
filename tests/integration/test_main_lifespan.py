import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestLifespan:
    
    @patch('src.api.main.RAGSystem')
    @patch('src.api.main.FAISSVectorStore')
    def test_lifespan_startup_success(self, mock_vs_class, mock_rag_class, tmp_path, monkeypatch):
        index_path = str(tmp_path / "faiss_index")
        monkeypatch.setenv("PERSIST_PATH", index_path)
        monkeypatch.setenv("EMBED_PROVIDER", "huggingface")
        
        mock_vs = Mock()
        mock_vs.load_index = Mock()
        mock_vs_class.return_value = mock_vs
        
        mock_rag = Mock()
        mock_rag_class.return_value = mock_rag
        
        from src.api.main import app
        
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            
            mock_vs_class.assert_called_once_with("huggingface")
            mock_vs.load_index.assert_called_once_with(index_path)
            
            mock_rag_class.assert_called_once_with(mock_vs)
            
            assert app.state.vector_store == mock_vs
            assert app.state.rag == mock_rag
        
        assert app.state.vector_store is None
        assert app.state.rag is None
    
    @patch('src.api.main.RAGSystem')
    @patch('src.api.main.FAISSVectorStore')
    def test_lifespan_default_values(self, mock_vs_class, mock_rag_class, monkeypatch):
        monkeypatch.delenv("PERSIST_PATH", raising=False)
        monkeypatch.delenv("EMBED_PROVIDER", raising=False)
        
        mock_vs = Mock()
        mock_vs.load_index = Mock()
        mock_vs_class.return_value = mock_vs
        
        mock_rag = Mock()
        mock_rag_class.return_value = mock_rag
        
        from src.api.main import app
        
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            
            mock_vs_class.assert_called_once_with("huggingface")
            mock_vs.load_index.assert_called_once_with("data/processed/faiss_index")
    
    @patch('src.api.main.RAGSystem')
    @patch('src.api.main.FAISSVectorStore')
    def test_lifespan_with_mistral(self, mock_vs_class, mock_rag_class, monkeypatch):
        monkeypatch.setenv("PERSIST_PATH", "test/path")
        monkeypatch.setenv("EMBED_PROVIDER", "mistral")
        
        mock_vs = Mock()
        mock_vs.load_index = Mock()
        mock_vs_class.return_value = mock_vs
        
        mock_rag = Mock()
        mock_rag_class.return_value = mock_rag
        
        from src.api.main import app
        
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            
            mock_vs_class.assert_called_once_with("mistral")


@pytest.mark.integration
class TestCORS:    
    @patch('src.api.main.RAGSystem')
    @patch('src.api.main.FAISSVectorStore')
    def test_cors_headers_present(self, mock_vs_class, mock_rag_class):
        mock_vs = Mock()
        mock_vs.load_index = Mock()
        mock_vs_class.return_value = mock_vs
        
        mock_rag = Mock()
        mock_rag_class.return_value = mock_rag
        
        from src.api.main import app
        
        with TestClient(app) as client:
            response = client.options(
                "/health",
                headers={"Origin": "http://localhost:3000"}
            )
            
            assert "access-control-allow-origin" in response.headers