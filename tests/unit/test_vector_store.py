import json
import os
from unittest.mock import Mock, patch
import pytest
from src.vector.langchain_faiss import FAISSVectorStore


@pytest.mark.unit
class TestInit:    
    @patch('src.vector.langchain_faiss.MistralAIEmbeddings')
    def test_init_with_mistral_provider(self, mock_mistral, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test_key_123")
        
        store = FAISSVectorStore(embedding_provider="mistral")
        
        assert store.embedding_provider == "mistral"
        assert store.vector_store is None
        mock_mistral.assert_called_once_with(
            model="mistral-embed",
            api_key="test_key_123"
        )
    
    def test_init_mistral_without_api_key(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="MISTRAL_API_KEY non trouvée"):
            FAISSVectorStore(embedding_provider="mistral")
    
    @patch('src.vector.langchain_faiss.HuggingFaceEmbeddings')
    def test_init_with_huggingface_default(self, mock_hf):
        store = FAISSVectorStore(embedding_provider="huggingface")
        
        assert store.embedding_provider == "huggingface"
        mock_hf.assert_called_once()
        call_kwargs = mock_hf.call_args[1]
        assert call_kwargs['model_name'] == "paraphrase-multilingual-MiniLM-L12-v2"
    
    @patch('src.vector.langchain_faiss.HuggingFaceEmbeddings')
    def test_init_with_custom_model(self, mock_hf):
        """Test initialisation avec modèle personnalisé"""
        custom_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        
        store = FAISSVectorStore(
            embedding_provider="huggingface",
            model_name=custom_model
        )
        
        call_kwargs = mock_hf.call_args[1]
        assert call_kwargs['model_name'] == custom_model
    
    def test_init_with_custom_embeddings(self, mock_embeddings):
        """Test initialisation avec embeddings fournis"""
        store = FAISSVectorStore(embeddings=mock_embeddings)
        
        assert store.embeddings == mock_embeddings
        assert store.vector_store is None


# ============================================================================
# Tests pour create_index
# ============================================================================

@pytest.mark.unit
class TestCreateIndex:
    """Tests pour la création d'index"""
    
    @patch('src.vector.langchain_faiss.FAISS')
    def test_create_index_success(self, mock_faiss, mock_embeddings, sample_chunks):
        """Test création d'index réussie"""
        store = FAISSVectorStore(embeddings=mock_embeddings)
        
        store.create_index(sample_chunks)
        
        mock_faiss.from_texts.assert_called_once()
        call_args = mock_faiss.from_texts.call_args
        
        # Vérifie que les textes sont extraits correctement
        texts = call_args[1]['texts']
        assert len(texts) == 3
        assert texts[0] == 'Concert de jazz à Paris le 15 décembre'
        
        # Vérifie que les métadonnées sont correctes
        metadatas = call_args[1]['metadatas']
        assert len(metadatas) == 3
        assert metadatas[0]['id'] == 'event-1'
        assert 'text' not in metadatas[0]  # 'text' ne doit pas être dans metadata
    
    def test_create_index_empty_list(self, mock_embeddings):
        """Test erreur avec liste vide"""
        store = FAISSVectorStore(embeddings=mock_embeddings)
        
        with pytest.raises(ValueError, match="La liste de chunks est vide"):
            store.create_index([])
    
    def test_create_index_missing_text_key(self, mock_embeddings, chunks_without_text):
        """Test erreur si chunk sans clé 'text'"""
        store = FAISSVectorStore(embeddings=mock_embeddings)
        
        with pytest.raises(ValueError, match="n'a pas de clé 'text'"):
            store.create_index(chunks_without_text)
    
    @patch('src.vector.langchain_faiss.FAISS')
    def test_create_index_filters_empty_chunks(
        self, mock_faiss, mock_embeddings, sample_chunks_with_empty, capsys
    ):
        """Test que les chunks vides sont filtrés"""
        store = FAISSVectorStore(embeddings=mock_embeddings)
        
        store.create_index(sample_chunks_with_empty)
        
        # Vérifie les messages d'avertissement
        captured = capsys.readouterr()
        assert "texte vide, ignoré" in captured.out
        
        # Vérifie que seuls les chunks valides sont utilisés
        call_args = mock_faiss.from_texts.call_args
        texts = call_args[1]['texts']
        assert len(texts) == 2  # Seulement les 2 chunks valides
    
    def test_create_index_all_empty_chunks(self, mock_embeddings):
        """Test erreur si tous les chunks sont vides"""
        chunks = [
            {'text': ''},
            {'text': '   '},
            {'text': None}
        ]
        store = FAISSVectorStore(embeddings=mock_embeddings)
        
        with pytest.raises(ValueError, match="Aucun chunk valide après filtrage"):
            store.create_index(chunks)
    
    @patch('src.vector.langchain_faiss.FAISS')
    def test_create_index_faiss_error(self, mock_faiss, mock_embeddings, sample_chunks):
        """Test gestion d'erreur lors de la création FAISS"""
        mock_faiss.from_texts.side_effect = Exception("Erreur FAISS")
        
        store = FAISSVectorStore(embeddings=mock_embeddings)
        
        with pytest.raises(RuntimeError, match="Erreur lors de la création de l'index"):
            store.create_index(sample_chunks)


# ============================================================================
# Tests pour search
# ============================================================================

@pytest.mark.unit
class TestSearch:
    """Tests pour la recherche"""
    
    def test_search_success(self, mock_embeddings, mock_vector_store):
        """Test recherche réussie"""
        # Mock des résultats de recherche
        mock_doc1 = Mock()
        mock_doc1.page_content = "Concert de jazz"
        mock_doc1.metadata = {'id': 'event-1', 'city': 'Paris'}
        
        mock_doc2 = Mock()
        mock_doc2.page_content = "Exposition d'art"
        mock_doc2.metadata = {'id': 'event-2', 'city': 'Lyon'}
        
        mock_vector_store.similarity_search_with_score.return_value = [
            (mock_doc1, 0.85),
            (mock_doc2, 0.72)
        ]
        
        store = FAISSVectorStore(embeddings=mock_embeddings)
        store.vector_store = mock_vector_store
        
        results = store.search("concert musique", k=2)
        
        assert len(results) == 2
        assert results[0][0]['text'] == "Concert de jazz"
        assert results[0][0]['metadata']['id'] == 'event-1'
        assert results[0][1] == 0.85
    
    def test_search_empty_query(self, mock_embeddings, mock_vector_store):
        """Test erreur avec requête vide"""
        store = FAISSVectorStore(embeddings=mock_embeddings)
        store.vector_store = mock_vector_store
        
        with pytest.raises(ValueError, match="La requête ne peut pas être vide"):
            store.search("")
    
    def test_search_with_score_threshold(self, mock_embeddings, mock_vector_store):
        """Test filtrage par score threshold"""
        mock_doc1 = Mock()
        mock_doc1.page_content = "Concert de jazz"
        mock_doc1.metadata = {'id': 'event-1'}
        
        mock_doc2 = Mock()
        mock_doc2.page_content = "Exposition"
        mock_doc2.metadata = {'id': 'event-2'}
        
        mock_vector_store.similarity_search_with_score.return_value = [
            (mock_doc1, 0.85),
            (mock_doc2, 0.45)  # Score trop bas
        ]
        
        store = FAISSVectorStore(embeddings=mock_embeddings)
        store.vector_store = mock_vector_store
        
        results = store.search("concert", k=2, score_threshold=0.5)
        
        # Seul le résultat avec score <= 0.5 doit être retourné
        assert len(results) == 1
        assert results[0][0]['metadata']['id'] == 'event-2'
    
    def test_search_with_filter(self, mock_embeddings, mock_vector_store):
        """Test recherche avec filtre"""
        store = FAISSVectorStore(embeddings=mock_embeddings)
        store.vector_store = mock_vector_store
        
        filter_dict = {'city': 'Paris'}
        store.search("concert", k=5, filter_dict=filter_dict)
        
        # Vérifie que le filtre est passé à FAISS
        mock_vector_store.similarity_search_with_score.assert_called_once_with(
            query="concert",
            k=5,
            filter=filter_dict
        )
    
    def test_search_faiss_error(self, mock_embeddings, mock_vector_store):
        """Test gestion d'erreur lors de la recherche"""
        mock_vector_store.similarity_search_with_score.side_effect = Exception("Erreur FAISS")
        
        store = FAISSVectorStore(embeddings=mock_embeddings)
        store.vector_store = mock_vector_store
        
        with pytest.raises(RuntimeError, match="Erreur lors de la recherche"):
            store.search("concert")


