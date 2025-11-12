"""
Tests unitaires pour RAGSystem
"""
import pytest
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.unit
class TestRAGSystemInit:
    """Tests pour l'initialisation de RAGSystem"""
    
    @patch('src.rag.rag_system.ChatMistralAI')
    def test_init_with_mistral(self, mock_mistral, mock_vector_store, monkeypatch):
        """Test initialisation avec Mistral"""
        monkeypatch.setenv("MISTRAL_API_KEY", "test_key")
        
        from src.rag.rag_system import RAGSystem
        
        rag = RAGSystem(mock_vector_store)
        
        assert rag.vector_store == mock_vector_store
        mock_mistral.assert_called_once()
    
    def test_init_no_api_key(self, mock_vector_store, monkeypatch):
        """Test erreur sans clé API"""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        
        from src.rag.rag_system import RAGSystem
        
        # Adapté au message d'erreur réel
        with pytest.raises((ValueError, KeyError)):
            RAGSystem(mock_vector_store)


@pytest.mark.unit
class TestRAGSystemQuery:
    @patch('src.rag.rag_system.ChatMistralAI')
    def test_query_no_results(self, mock_mistral, mock_vector_store, monkeypatch):
        """Test requête sans résultats"""
        monkeypatch.setenv("MISTRAL_API_KEY", "test_key")
        
        mock_vector_store.search = Mock(return_value=[])
        
        mock_llm_instance = Mock()
        mock_llm_instance.invoke = Mock(return_value=Mock(
            content="Aucun événement trouvé."
        ))
        mock_mistral.return_value = mock_llm_instance
        
        from src.rag.rag_system import RAGSystem
        
        rag = RAGSystem(mock_vector_store)
        result = rag.query("Événements inexistants ?", k=5)
        
        assert 'answer' in result
        assert 'context' in result
        # Le contexte devrait être vide ou indiquer aucun résultat
        assert result['context'] == '' or 'Aucun' in result['context']
    
    @patch('src.rag.rag_system.ChatMistralAI')
    def test_query_with_multiple_results(self, mock_mistral, mock_vector_store, monkeypatch):
        """Test requête avec plusieurs résultats"""
        monkeypatch.setenv("MISTRAL_API_KEY", "test_key")
        
        # Configure le mock vector store avec plusieurs résultats
        mock_vector_store.search = Mock(return_value=[
            (
                {
                    'text': 'Concert de jazz à Paris',
                    'metadata': {
                        'id': 'event-1',
                        'title': 'Concert de Jazz',
                        'date_start': '2025-12-15',
                        'location_name': 'Salle Pleyel',
                        'location_city': 'Paris',
                        'category': 'musique'
                    },
                    'score': 0.85
                },
                0.85
            ),
            (
                {
                    'text': 'Exposition d\'art moderne',
                    'metadata': {
                        'id': 'event-2',
                        'title': 'Expo Art',
                        'date_start': '2025-12-20',
                        'location_name': 'Musée',
                        'location_city': 'Lyon',
                        'category': 'exposition'
                    },
                    'score': 0.75
                },
                0.75
            )
        ])
        
        mock_llm_instance = Mock()
        mock_llm_instance.invoke = Mock(return_value=Mock(
            content="Il y a un concert à Paris et une exposition à Lyon."
        ))
        mock_mistral.return_value = mock_llm_instance
        
        from src.rag.rag_system import RAGSystem
        
        rag = RAGSystem(mock_vector_store)
        result = rag.query("Événements culturels ?", k=5)
        
        assert 'answer' in result
        assert 'context' in result
        # Le contexte devrait contenir les deux événements
        assert 'ÉVÉNEMENT 1' in result['context']
        assert 'ÉVÉNEMENT 2' in result['context']
    
    @patch('src.rag.rag_system.ChatMistralAI')
    def test_query_llm_error(self, mock_mistral, mock_vector_store, monkeypatch):
        """Test gestion d'erreur du LLM"""
        monkeypatch.setenv("MISTRAL_API_KEY", "test_key")
        
        mock_vector_store.search = Mock(return_value=[
            (
                {
                    'text': 'Concert de jazz',
                    'metadata': {'id': 'event-1', 'title': 'Concert'},
                    'score': 0.85
                },
                0.85
            )
        ])
        
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.side_effect = Exception("Erreur LLM")
        mock_mistral.return_value = mock_llm_instance
        
        from src.rag.rag_system import RAGSystem
        
        rag = RAGSystem(mock_vector_store)
        
        # Vérifie que l'erreur est gérée
        with pytest.raises(Exception, match="Erreur LLM"):
            rag.query("Test", k=5)
    
    @patch('src.rag.rag_system.ChatMistralAI')
    def test_query_search_error(self, mock_mistral, mock_vector_store, monkeypatch):
        """Test gestion d'erreur de la recherche"""
        monkeypatch.setenv("MISTRAL_API_KEY", "test_key")
        
        mock_vector_store.search.side_effect = Exception("Erreur recherche")
        
        mock_llm_instance = Mock()
        mock_mistral.return_value = mock_llm_instance
        
        from src.rag.rag_system import RAGSystem
        
        rag = RAGSystem(mock_vector_store)
        
        with pytest.raises(Exception, match="Erreur recherche"):
            rag.query("Test", k=5)


@pytest.mark.unit
class TestRAGSystemFormatContext:
    """Tests pour le formatage du contexte"""
    
    @patch('src.rag.rag_system.ChatMistralAI')
    def test_format_context_with_metadata(self, mock_mistral, mock_vector_store, monkeypatch):
        """Test que les métadonnées sont bien formatées"""
        monkeypatch.setenv("MISTRAL_API_KEY", "test_key")
        
        mock_vector_store.search = Mock(return_value=[
            (
                {
                    'text': 'Concert de jazz',
                    'metadata': {
                        'id': 'event-1',
                        'title': 'Super Concert',
                        'date_start': '2025-12-15T20:00:00',
                        'location_name': 'Salle Pleyel',
                        'location_city': 'Paris',
                        'category': 'musique',
                        'url': 'https://example.com/concert'
                    },
                    'score': 0.85
                },
                0.85
            )
        ])
        
        mock_llm_instance = Mock()
        mock_llm_instance.invoke = Mock(return_value=Mock(content="Réponse"))
        mock_mistral.return_value = mock_llm_instance
        
        from src.rag.rag_system import RAGSystem
        
        rag = RAGSystem(mock_vector_store)
        result = rag.query("Test", k=5)
        
        context = result['context']
        
        # Vérifie que les métadonnées importantes sont présentes
        assert 'Super Concert' in context or 'event-1' in context
        assert '2025-12-15' in context
        assert 'Paris' in context or 'Salle Pleyel' in context

