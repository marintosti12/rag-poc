"""
Tests pour le lifespan de l'application
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestLifespan:
    """Tests pour le cycle de vie de l'application"""
    
    @patch('src.api.main.RAGSystem')
    @patch('src.api.main.FAISSVectorStore')
    def test_lifespan_startup_success(self, mock_vs_class, mock_rag_class, tmp_path, monkeypatch):
        """Test du démarrage réussi de l'application"""
        # Configure les variables d'environnement
        index_path = str(tmp_path / "faiss_index")
        monkeypatch.setenv("PERSIST_PATH", index_path)
        monkeypatch.setenv("EMBED_PROVIDER", "huggingface")
        
        # Configure les mocks
        mock_vs = Mock()
        mock_vs.load_index = Mock()
        mock_vs_class.return_value = mock_vs
        
        mock_rag = Mock()
        mock_rag_class.return_value = mock_rag
        
        # Importe l'app APRÈS avoir configuré les patches
        from src.api.main import app
        
        # Utilise TestClient qui déclenche le lifespan
        with TestClient(app) as client:
            # Vérifie que l'app a démarré
            response = client.get("/health")
            assert response.status_code == 200
            
            # Vérifie que le vector store a été créé et chargé
            mock_vs_class.assert_called_once_with("huggingface")
            mock_vs.load_index.assert_called_once_with(index_path)
            
            # Vérifie que le RAG a été créé
            mock_rag_class.assert_called_once_with(mock_vs)
            
            # Vérifie que l'état a été configuré
            assert app.state.vector_store == mock_vs
            assert app.state.rag == mock_rag
        
        # Après le contexte, vérifie le shutdown
        assert app.state.vector_store is None
        assert app.state.rag is None
    
    @patch('src.api.main.RAGSystem')
    @patch('src.api.main.FAISSVectorStore')
    def test_lifespan_default_values(self, mock_vs_class, mock_rag_class, monkeypatch):
        """Test des valeurs par défaut du lifespan"""
        # Supprime les variables d'environnement
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
            
            # Vérifie les valeurs par défaut
            mock_vs_class.assert_called_once_with("huggingface")
            mock_vs.load_index.assert_called_once_with("data/processed/faiss_index")
    
    @patch('src.api.main.RAGSystem')
    @patch('src.api.main.FAISSVectorStore')
    def test_lifespan_with_mistral(self, mock_vs_class, mock_rag_class, monkeypatch):
        """Test du lifespan avec provider Mistral"""
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
            
            # Vérifie que mistral a été utilisé
            mock_vs_class.assert_called_once_with("mistral")


@pytest.mark.integration
class TestCORS:
    """Tests pour la configuration CORS"""
    
    @patch('src.api.main.RAGSystem')
    @patch('src.api.main.FAISSVectorStore')
    def test_cors_headers_present(self, mock_vs_class, mock_rag_class):
        """Test que les headers CORS sont présents"""
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
            
            # Vérifie les headers CORS
            assert "access-control-allow-origin" in response.headers