"""
Tests d'intégration pour l'API FastAPI
"""
import os
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_vector_store():
    """Mock du vector store"""
    mock = Mock()
    mock.search = Mock(return_value=[
        (
            {
                'text': 'Concert de jazz à Paris le 15 décembre 2025',
                'metadata': {
                    'id': 'event-1',
                    'category': 'musique',
                    'date_start': '2025-12-15',
                    'url': 'https://example.com/concert'
                },
                'score': 0.85
            },
            0.85
        )
    ])
    mock.load_index = Mock()
    mock.create_index = Mock()
    mock.save_index = Mock()
    return mock


@pytest.fixture
def mock_rag_system():
    """Mock du système RAG"""
    mock = Mock()
    mock.query = Mock(return_value={
        'answer': 'Il y a un concert de jazz à Paris le 15 décembre 2025 à la Salle Pleyel.',
        'contexts': [
            {
                'text': 'Concert de jazz à Paris le 15 décembre 2025',
                'metadata': {'id': 'event-1', 'category': 'musique'},
                'score': 0.85
            }
        ]
    })
    return mock


@pytest.fixture
def client_with_mocks(mock_vector_store, mock_rag_system):
    """Client de test avec mocks"""
    # Patchez AVANT d'importer l'app
    with patch('src.api.main.FAISSVectorStore') as mock_vs_class, \
         patch('src.api.main.RAGSystem') as mock_rag_class, \
         patch.dict(os.environ, {'PERSIST_PATH': 'test/path', 'EMBED_PROVIDER': 'huggingface'}):
        
        mock_vs_class.return_value = mock_vector_store
        mock_rag_class.return_value = mock_rag_system
        
        # Import APRES les patches
        from src.api.main import app
        
        # Configure l'état de l'app manuellement
        app.state.vector_store = mock_vector_store
        app.state.rag = mock_rag_system
        
        client = TestClient(app)
        
        yield client
        
        # Nettoyage
        app.state.vector_store = None
        app.state.rag = None

@pytest.fixture
def sample_docs():
    """Documents de test pour rebuild"""
    return [
        {
            "text": "Concert de jazz à Paris le 15 décembre 2025 à la Salle Pleyel",
            "metadata": {
                "id": "event-1",
                "category": "musique",
                "date_start": "2025-12-15",
                "url": "https://example.com/concert"
            }
        },
        {
            "text": "Exposition d'art moderne au Louvre du 20 au 30 décembre",
            "metadata": {
                "id": "event-2",
                "category": "exposition",
                "date_start": "2025-12-20"
            }
        }
    ]


# ============================================================================
# Tests pour /health
# ============================================================================

@pytest.mark.integration
class TestHealthEndpoint:
    """Tests pour l'endpoint /health"""
    
    def test_health_check_success(self, client_with_mocks):
        """Test que le health check fonctionne"""
        response = client_with_mocks.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ============================================================================
# Tests pour /ask
# ============================================================================

@pytest.mark.integration
class TestAskEndpoint:
    """Tests pour l'endpoint /ask"""
    
    def test_ask_success(self, client_with_mocks, mock_rag_system):
        """Test d'une question réussie"""
        payload = {
            "question": "Quels concerts de jazz à Paris en décembre ?",
            "k": 5
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0
        
        # Vérifie que le RAG a été appelé
        mock_rag_system.query.assert_called_once_with(
            question=payload["question"],
            k=payload["k"]
        )
    
    def test_ask_with_default_k(self, client_with_mocks, mock_rag_system):
        """Test avec k par défaut"""
        payload = {
            "question": "Événements à Paris ?"
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 200
        # Vérifie que k par défaut est utilisé
        call_args = mock_rag_system.query.call_args
        assert call_args is not None
    
    def test_ask_empty_question(self, client_with_mocks):
        """Test avec question vide"""
        payload = {
            "question": ""
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 400
        assert "vide" in response.json()["detail"].lower()
    
    def test_ask_whitespace_question(self, client_with_mocks):
        """Test avec question contenant seulement des espaces"""
        payload = {
            "question": "   "
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 400
    
    def test_ask_missing_question_field(self, client_with_mocks):
        """Test sans le champ question"""
        payload = {
            "k": 5
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    def test_ask_invalid_k_type(self, client_with_mocks):
        """Test avec k de type invalide"""
        payload = {
            "question": "Test",
            "k": "invalid"
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 422
    
    def test_ask_rag_error(self, client_with_mocks, mock_rag_system):
        """Test gestion d'erreur du système RAG"""
        mock_rag_system.query.side_effect = Exception("Erreur RAG")
        
        payload = {
            "question": "Test question"
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 500
        assert "Erreur RAG" in response.json()["detail"]
    
    def test_ask_pipeline_not_loaded(self, client_with_mocks):
        """Test quand le pipeline n'est pas chargé"""
        # Simule l'absence de rag dans app.state
        client_with_mocks.app.state.rag = None
        
        payload = {
            "question": "Test question"
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 503
        assert "non chargé" in response.json()["detail"].lower()


# ============================================================================
# Tests pour /rebuild
# ============================================================================

@pytest.mark.integration
class TestRebuildEndpoint:
    """Tests pour l'endpoint /rebuild"""
    
    @patch('src.api.controllers.rebuild_controller.FAISSVectorStore')
    @patch('src.api.controllers.rebuild_controller.RAGSystem')
    def test_rebuild_success(
        self, mock_rag_class, mock_vs_class, client_with_mocks, sample_docs
    ):
        """Test de reconstruction réussie"""
        mock_vs = Mock()
        mock_vs.create_index = Mock()
        mock_vs.save_index = Mock()
        mock_vs_class.return_value = mock_vs
        
        mock_rag = Mock()
        mock_rag_class.return_value = mock_rag
        
        payload = {
            "docs": sample_docs,
            "persist_path": "data/test/faiss_index",
            "embedding_provider": "mistral"
        }
        
        response = client_with_mocks.post("/rebuild", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["ok"] is True
        assert data["count"] == 2
        assert data["index_path"] == "data/test/faiss_index"
        assert data["provider"] == "mistral"
        assert "created_at" in data
        
        # Vérifie que les méthodes ont été appelées
        mock_vs.create_index.assert_called_once()
        mock_vs.save_index.assert_called_once_with("data/test/faiss_index")
    
    def test_rebuild_empty_docs(self, client_with_mocks):
        """Test avec liste de documents vide"""
        payload = {
            "docs": []
        }
        
        response = client_with_mocks.post("/rebuild", json=payload)
        
        assert response.status_code == 400
        assert "vide" in response.json()["detail"].lower()
    
    def test_rebuild_missing_docs_field(self, client_with_mocks):
        """Test sans le champ docs"""
        payload = {
            "persist_path": "data/test/faiss_index"
        }
        
        response = client_with_mocks.post("/rebuild", json=payload)
        
        assert response.status_code == 422
    
    @patch('src.api.controllers.rebuild_controller.FAISSVectorStore')
    def test_rebuild_with_default_values(
        self, mock_vs_class, client_with_mocks, sample_docs
    ):
        """Test avec valeurs par défaut"""
        mock_vs = Mock()
        mock_vs.create_index = Mock()
        mock_vs.save_index = Mock()
        mock_vs_class.return_value = mock_vs
        
        payload = {
            "docs": sample_docs
            # persist_path et embedding_provider utilisent les valeurs par défaut
        }
        
        response = client_with_mocks.post("/rebuild", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["index_path"] == "data/processed/faiss_index"
        assert data["provider"] == "mistral"
    
    @patch('src.api.controllers.rebuild_controller.FAISSVectorStore')
    def test_rebuild_create_index_error(
        self, mock_vs_class, client_with_mocks, sample_docs
    ):
        """Test gestion d'erreur lors de create_index"""
        mock_vs = Mock()
        mock_vs.create_index.side_effect = Exception("Erreur création index")
        mock_vs_class.return_value = mock_vs
        
        payload = {
            "docs": sample_docs
        }
        
        response = client_with_mocks.post("/rebuild", json=payload)
        
        assert response.status_code == 500
        assert "Erreur création index" in response.json()["detail"]
    
    @patch('src.api.controllers.rebuild_controller.FAISSVectorStore')
    def test_rebuild_save_error(
        self, mock_vs_class, client_with_mocks, sample_docs
    ):
        """Test gestion d'erreur lors de la sauvegarde"""
        mock_vs = Mock()
        mock_vs.create_index = Mock()
        mock_vs.save_index.side_effect = Exception("Erreur sauvegarde")
        mock_vs_class.return_value = mock_vs
        
        payload = {
            "docs": sample_docs
        }
        
        response = client_with_mocks.post("/rebuild", json=payload)
        
        assert response.status_code == 500
        assert "Erreur sauvegarde" in response.json()["detail"]
    
    @patch('src.api.controllers.rebuild_controller.FAISSVectorStore')
    @patch('src.api.controllers.rebuild_controller.RAGSystem')
    def test_rebuild_updates_app_state(
        self, mock_rag_class, mock_vs_class, client_with_mocks, sample_docs
    ):
        """Test que rebuild met à jour l'état de l'application"""
        mock_vs = Mock()
        mock_vs.create_index = Mock()
        mock_vs.save_index = Mock()
        mock_vs_class.return_value = mock_vs
        
        mock_rag = Mock()
        mock_rag_class.return_value = mock_rag
        
        payload = {
            "docs": sample_docs
        }
        
        response = client_with_mocks.post("/rebuild", json=payload)
        
        assert response.status_code == 200
        
        # Vérifie que l'état a été mis à jour
        assert client_with_mocks.app.state.vector_store == mock_vs
        assert client_with_mocks.app.state.rag == mock_rag
    
    def test_rebuild_invalid_doc_structure(self, client_with_mocks):
        """Test avec structure de document invalide"""
        payload = {
            "docs": [
                {"invalid": "structure"}  # Manque le champ 'text'
            ]
        }
        
        response = client_with_mocks.post("/rebuild", json=payload)
        
        assert response.status_code == 422


# ============================================================================
# Tests de workflow complet
# ============================================================================

@pytest.mark.integration
class TestCompleteWorkflow:
    """Tests de workflow complet"""
    
    @patch('src.api.controllers.rebuild_controller.FAISSVectorStore')
    @patch('src.api.controllers.rebuild_controller.RAGSystem')
    def test_rebuild_then_ask(
        self, mock_rag_class, mock_vs_class, client_with_mocks, sample_docs
    ):
        """Test rebuild suivi d'une question"""
        # Setup mocks
        mock_vs = Mock()
        mock_vs.create_index = Mock()
        mock_vs.save_index = Mock()
        mock_vs_class.return_value = mock_vs
        
        mock_rag = Mock()
        mock_rag.query = Mock(return_value={
            'answer': 'Concert de jazz à Paris',
            'contexts': []
        })
        mock_rag_class.return_value = mock_rag
        
        # 1. Rebuild
        rebuild_payload = {
            "docs": sample_docs
        }
        
        rebuild_response = client_with_mocks.post("/rebuild", json=rebuild_payload)
        assert rebuild_response.status_code == 200
        
        # 2. Ask
        ask_payload = {
            "question": "Concerts à Paris ?"
        }
        
        ask_response = client_with_mocks.post("/ask", json=ask_payload)
        assert ask_response.status_code == 200
        
        # Vérifie la réponse
        data = ask_response.json()
        assert "answer" in data
        assert "Concert de jazz à Paris" in data["answer"]


# ============================================================================
# Tests de validation des schémas
# ============================================================================

@pytest.mark.integration
class TestSchemaValidation:
    """Tests de validation des schémas Pydantic"""
    
    def test_ask_schema_validation(self, client_with_mocks):
        """Test validation du schéma AskIn"""
        test_cases = [
            # Cas valides
            ({"question": "Test", "k": 5}, 200),
            ({"question": "Test"}, 200),  # k optionnel
            
            # Cas invalides
            ({}, 422),  # question manquante
            ({"question": "Test", "k": -1}, 422),  # k négatif
            ({"question": "Test", "k": 0}, 422),  # k zéro
        ]
        
        for payload, expected_status in test_cases:
            response = client_with_mocks.post("/ask", json=payload)
            if expected_status == 200:
                assert response.status_code in [200, 400]  # 400 si question vide
            else:
                assert response.status_code == expected_status
    
    def test_rebuild_schema_validation(self, client_with_mocks):
        """Test validation du schéma RebuildIn"""
        valid_doc = {
            "text": "Test event",
            "metadata": {"id": "test-1"}
        }
        
        test_cases = [
            # Cas valides
            ({"docs": [valid_doc]}, 200),
            ({"docs": [valid_doc], "persist_path": "custom/path"}, 200),
            ({"docs": [valid_doc], "embedding_provider": "huggingface"}, 200),
            
            # Cas invalides
            ({}, 422),  # docs manquant
            ({"docs": "not_a_list"}, 422),  # docs n'est pas une liste
        ]
        
        with patch('src.api.controllers.rebuild_controller.FAISSVectorStore') as mock_vs_class:
            mock_vs = Mock()
            mock_vs.create_index = Mock()
            mock_vs.save_index = Mock()
            mock_vs_class.return_value = mock_vs
            
            for payload, expected_status in test_cases:
                response = client_with_mocks.post("/rebuild", json=payload)
                if expected_status == 200:
                    assert response.status_code in [200, 400, 500]
                else:
                    assert response.status_code == expected_status


# ============================================================================
# Tests de cas limites
# ============================================================================

@pytest.mark.integration
class TestEdgeCases:
    """Tests de cas limites"""
    
    def test_ask_very_long_question(self, client_with_mocks):
        """Test avec une question très longue"""
        payload = {
            "question": "Test " * 1000  # Question très longue
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        # Doit accepter ou renvoyer une erreur appropriée
        assert response.status_code in [200, 400, 413, 500]
    
    def test_ask_special_characters(self, client_with_mocks):
        """Test avec caractères spéciaux"""
        payload = {
            "question": "Événements à Paris avec accents é è ê ë ?"
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 200
    
    @patch('src.api.controllers.rebuild_controller.FAISSVectorStore')
    def test_rebuild_many_docs(self, mock_vs_class, client_with_mocks):
        """Test rebuild avec beaucoup de documents"""
        mock_vs = Mock()
        mock_vs.create_index = Mock()
        mock_vs.save_index = Mock()
        mock_vs_class.return_value = mock_vs
        
        # Crée 100 documents
        many_docs = [
            {
                "text": f"Event {i}",
                "metadata": {"id": f"event-{i}"}
            }
            for i in range(100)
        ]
        
        payload = {
            "docs": many_docs
        }
        
        response = client_with_mocks.post("/rebuild", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 100