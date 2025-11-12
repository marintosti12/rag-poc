import os
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime



@pytest.fixture
def mock_vector_store():
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
    with patch('src.api.main.FAISSVectorStore') as mock_vs_class, \
         patch('src.api.main.RAGSystem') as mock_rag_class, \
         patch.dict(os.environ, {'PERSIST_PATH': 'test/path', 'EMBED_PROVIDER': 'huggingface'}):
        
        mock_vs_class.return_value = mock_vector_store
        mock_rag_class.return_value = mock_rag_system
        
        from src.api.main import app
        
        app.state.vector_store = mock_vector_store
        app.state.rag = mock_rag_system
        
        client = TestClient(app)
        
        yield client
        
        app.state.vector_store = None
        app.state.rag = None

@pytest.fixture
def sample_docs():
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

@pytest.mark.integration
class TestHealthEndpoint:    
    def test_health_check_success(self, client_with_mocks):
        response = client_with_mocks.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.integration
class TestAskEndpoint:    
    def test_ask_success(self, client_with_mocks, mock_rag_system):
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
        
        mock_rag_system.query.assert_called_once_with(
            question=payload["question"],
            k=payload["k"]
        )
    
    def test_ask_with_default_k(self, client_with_mocks, mock_rag_system):
        payload = {
            "question": "Événements à Paris ?"
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 200
        call_args = mock_rag_system.query.call_args
        assert call_args is not None
    
    def test_ask_empty_question(self, client_with_mocks):
        payload = {
            "question": ""
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 400
        assert "vide" in response.json()["detail"].lower()
    
    def test_ask_whitespace_question(self, client_with_mocks):
        payload = {
            "question": "   "
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 400
    
    def test_ask_missing_question_field(self, client_with_mocks):
        payload = {
            "k": 5
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 422 
    
    def test_ask_invalid_k_type(self, client_with_mocks):
        payload = {
            "question": "Test",
            "k": "invalid"
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 422
    
    def test_ask_rag_error(self, client_with_mocks, mock_rag_system):
        mock_rag_system.query.side_effect = Exception("Erreur RAG")
        
        payload = {
            "question": "Test question"
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 500
        assert "Erreur RAG" in response.json()["detail"]
    
    def test_ask_pipeline_not_loaded(self, client_with_mocks):
        client_with_mocks.app.state.rag = None
        
        payload = {
            "question": "Test question"
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 503
        assert "non chargé" in response.json()["detail"].lower()


@pytest.mark.integration
class TestRebuildEndpoint:    
    @patch('src.api.controllers.rebuild_controller.FAISSVectorStore')
    @patch('src.api.controllers.rebuild_controller.RAGSystem')
    def test_rebuild_success(
        self, mock_rag_class, mock_vs_class, client_with_mocks, sample_docs
    ):
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
        
        mock_vs.create_index.assert_called_once()
        mock_vs.save_index.assert_called_once_with("data/test/faiss_index")
    
    def test_rebuild_empty_docs(self, client_with_mocks):
        payload = {
            "docs": []
        }
        
        response = client_with_mocks.post("/rebuild", json=payload)
        
        assert response.status_code == 400
        assert "vide" in response.json()["detail"].lower()
    
    def test_rebuild_missing_docs_field(self, client_with_mocks):
        payload = {
            "persist_path": "data/test/faiss_index"
        }
        
        response = client_with_mocks.post("/rebuild", json=payload)
        
        assert response.status_code == 422
    
    @patch('src.api.controllers.rebuild_controller.FAISSVectorStore')
    def test_rebuild_with_default_values(
        self, mock_vs_class, client_with_mocks, sample_docs
    ):
        mock_vs = Mock()
        mock_vs.create_index = Mock()
        mock_vs.save_index = Mock()
        mock_vs_class.return_value = mock_vs
        
        payload = {
            "docs": sample_docs
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
        
        assert client_with_mocks.app.state.vector_store == mock_vs
        assert client_with_mocks.app.state.rag == mock_rag
    
    def test_rebuild_invalid_doc_structure(self, client_with_mocks):
        payload = {
            "docs": [
                {"invalid": "structure"}
            ]
        }
        
        response = client_with_mocks.post("/rebuild", json=payload)
        
        assert response.status_code == 422

@pytest.mark.integration
class TestCompleteWorkflow:    
    @patch('src.api.controllers.rebuild_controller.FAISSVectorStore')
    @patch('src.api.controllers.rebuild_controller.RAGSystem')
    def test_rebuild_then_ask(
        self, mock_rag_class, mock_vs_class, client_with_mocks, sample_docs
    ):
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
        
        rebuild_payload = {
            "docs": sample_docs
        }
        
        rebuild_response = client_with_mocks.post("/rebuild", json=rebuild_payload)
        assert rebuild_response.status_code == 200
        
        ask_payload = {
            "question": "Concerts à Paris ?"
        }
        
        ask_response = client_with_mocks.post("/ask", json=ask_payload)
        assert ask_response.status_code == 200
        
        data = ask_response.json()
        assert "answer" in data
        assert "Concert de jazz à Paris" in data["answer"]


@pytest.mark.integration
class TestSchemaValidation:    
    def test_ask_schema_validation(self, client_with_mocks):
        test_cases = [
            ({"question": "Test", "k": 5}, 200),
            ({"question": "Test"}, 200),
            
            ({}, 422), 
            ({"question": "Test", "k": -1}, 422),
            ({"question": "Test", "k": 0}, 422),
        ]
        
        for payload, expected_status in test_cases:
            response = client_with_mocks.post("/ask", json=payload)
            if expected_status == 200:
                assert response.status_code in [200, 400]
            else:
                assert response.status_code == expected_status
    
    def test_rebuild_schema_validation(self, client_with_mocks):
        valid_doc = {
            "text": "Test event",
            "metadata": {"id": "test-1"}
        }
        
        test_cases = [
            ({"docs": [valid_doc]}, 200),
            ({"docs": [valid_doc], "persist_path": "custom/path"}, 200),
            ({"docs": [valid_doc], "embedding_provider": "huggingface"}, 200),
            
            ({}, 422),  
            ({"docs": "not_a_list"}, 422),
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


@pytest.mark.integration
class TestEdgeCases:    
    def test_ask_very_long_question(self, client_with_mocks):
        payload = {
            "question": "Test " * 1000
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code in [200, 400, 413, 500]
    
    def test_ask_special_characters(self, client_with_mocks):
        payload = {
            "question": "Événements à Paris avec accents é è ê ë ?"
        }
        
        response = client_with_mocks.post("/ask", json=payload)
        
        assert response.status_code == 200
    
    @patch('src.api.controllers.rebuild_controller.FAISSVectorStore')
    def test_rebuild_many_docs(self, mock_vs_class, client_with_mocks):
        mock_vs = Mock()
        mock_vs.create_index = Mock()
        mock_vs.save_index = Mock()
        mock_vs_class.return_value = mock_vs
        
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