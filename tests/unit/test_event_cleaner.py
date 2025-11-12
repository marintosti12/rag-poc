"""
Tests unitaires pour EventsCleaner
"""
import pytest
import pandas as pd
import json
from src.fetching.clean_events import EventsCleaner


# ============================================================================
# Tests pour extract_key_fields
# ============================================================================

@pytest.mark.unit
class TestExtractKeyFields:
    """Tests pour la méthode extract_key_fields"""
    
    def test_extract_single_event_complete(self, sample_event):
        """Test extraction d'un événement complet"""
        result = EventsCleaner.extract_key_fields([sample_event])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['id'] == 'event-123'
        assert result.iloc[0]['title'] == 'Concert de Jazz'
        assert result.iloc[0]['description'] == 'Un superbe concert de jazz au coeur de Paris'
        assert result.iloc[0]['location_name'] == 'Salle Pleyel'
        assert result.iloc[0]['location_city'] == 'Paris'
        assert result.iloc[0]['date_start'] == '2025-12-15T20:00:00'
        assert result.iloc[0]['date_end'] == '2025-12-15T23:00:00'
        assert result.iloc[0]['url'] == 'https://example.com/concert-jazz'
        assert result.iloc[0]['keywords'] == 'jazz, musique, concert'
        assert result.iloc[0]['category'] == 'musique'
    
    def test_extract_minimal_event(self, sample_event_minimal):
        """Test extraction d'un événement avec données minimales"""
        result = EventsCleaner.extract_key_fields([sample_event_minimal])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['id'] == 'event-minimal'
        assert result.iloc[0]['title'] == 'Événement minimal'
        assert result.iloc[0]['description'] == 'Description minimale'
        assert result.iloc[0]['date_start'] == '2025-12-20T10:00:00'
    
    def test_extract_multiple_events(self, sample_events_list):
        """Test extraction de plusieurs événements"""
        result = EventsCleaner.extract_key_fields(sample_events_list)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result['id']) == ['event-123', 'event-minimal', 'event-456']
    
    def test_extract_empty_list(self):
        """Test avec une liste vide"""
        result = EventsCleaner.extract_key_fields([])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_extract_event_without_timings(self):
        """Test événement sans timings"""
        event = {
            'uid': 'event-no-timing',
            'title': {'fr': 'Sans timing'},
            'description': {'fr': 'Description'},
        }
        result = EventsCleaner.extract_key_fields([event])
        
        assert len(result) == 1
        assert pd.isna(result.iloc[0]['date_start'])
    
    def test_extract_event_with_empty_timings(self):
        """Test événement avec timings vide"""
        event = {
            'uid': 'event-empty-timing',
            'title': {'fr': 'Timing vide'},
            'description': {'fr': 'Description'},
            'timings': []
        }
        result = EventsCleaner.extract_key_fields([event])
        
        assert len(result) == 1
    
    def test_extract_event_with_firstTiming_fallback(self):
        """Test utilisation de firstTiming comme fallback"""
        event = {
            'uid': 'event-first-timing',
            'title': {'fr': 'Avec firstTiming'},
            'description': {'fr': 'Description'},
            'firstTiming': {'begin': '2025-12-25T10:00:00'},
            'lastTiming': {'end': '2025-12-25T18:00:00'}
        }
        result = EventsCleaner.extract_key_fields([event])
        
        assert result.iloc[0]['date_start'] == '2025-12-25T10:00:00'
        assert result.iloc[0]['date_end'] == '2025-12-25T18:00:00'
    
    def test_extract_event_with_location_dict(self):
        """Test extraction location depuis dict"""
        event = {
            'uid': 'event-location-dict',
            'title': {'fr': 'Event'},
            'description': {'fr': 'Desc'},
            'location': {
                'name': 'Lieu Test',
                'city': 'Ville Test'
            }
        }
        result = EventsCleaner.extract_key_fields([event])
        
        assert result.iloc[0]['location_name'] == 'Lieu Test'
        assert result.iloc[0]['location_city'] == 'Ville Test'
    
    def test_extract_event_with_links_array(self):
        """Test extraction URL depuis links array"""
        event = {
            'uid': 'event-links',
            'title': {'fr': 'Event'},
            'description': {'fr': 'Desc'},
            'links': [{'link': 'https://test.com'}]
        }
        result = EventsCleaner.extract_key_fields([event])
        
        assert result.iloc[0]['url'] == 'https://test.com'
    
    def test_extract_event_with_empty_keywords(self):
        """Test événement avec keywords vides"""
        event = {
            'uid': 'event-no-keywords',
            'title': {'fr': 'Event'},
            'description': {'fr': 'Desc'},
            'keywords': {'fr': []}
        }
        result = EventsCleaner.extract_key_fields([event])
        
        assert result.iloc[0]['keywords'] == ''
    
    def test_extract_event_malformed_skipped(self, capsys):
        """Test qu'un événement malformé est ignoré avec un message d'erreur"""
        events = [
            {
                'uid': 'good-event',
                'title': {'fr': 'Good'},
                'description': {'fr': 'Desc'}
            },
            None,  # Événement malformé (None)
            "invalid",  # Événement malformé (string)
            {
                'uid': 'another-good',
                'title': {'fr': 'Another'},
                'description': {'fr': 'Desc'}
            }
        ]
        
        result = EventsCleaner.extract_key_fields(events)
        captured = capsys.readouterr()
        
        # Vérifie qu'un message d'erreur a été affiché
        assert "Erreur sur événement" in captured.out
        # Vérifie que les bons événements ont été extraits (ignore les 2 malformés)
        assert len(result) == 2
        assert list(result['id']) == ['good-event', 'another-good']


# ============================================================================
# Tests pour remove_duplicates
# ============================================================================

@pytest.mark.unit
class TestRemoveDuplicates:
    """Tests pour la méthode remove_duplicates"""
    
    def test_remove_duplicates_basic(self, dataframe_with_duplicates):
        """Test suppression de doublons basique"""
        result = EventsCleaner.remove_duplicates(dataframe_with_duplicates)
        
        assert len(result) == 2
        assert list(result['id']) == ['event-1', 'event-2']
    
    def test_remove_duplicates_no_duplicates(self, sample_dataframe):
        """Test avec DataFrame sans doublons"""
        original_len = len(sample_dataframe)
        result = EventsCleaner.remove_duplicates(sample_dataframe)
        
        assert len(result) == original_len
    
    def test_remove_duplicates_empty_dataframe(self):
        """Test avec DataFrame vide"""
        df = pd.DataFrame(columns=['id', 'title', 'description'])
        result = EventsCleaner.remove_duplicates(df)
        
        assert len(result) == 0
    
    def test_remove_duplicates_all_duplicates(self):
        """Test avec tous les événements dupliqués"""
        df = pd.DataFrame([
            {'id': 'event-1', 'title': 'Event'},
            {'id': 'event-1', 'title': 'Event'},
            {'id': 'event-1', 'title': 'Event'},
        ])
        result = EventsCleaner.remove_duplicates(df)
        
        assert len(result) == 1


# ============================================================================
# Tests pour remove_missing_descriptions
# ============================================================================

@pytest.mark.unit
class TestRemoveMissingDescriptions:
    """Tests pour la méthode remove_missing_descriptions"""
    
    def test_remove_missing_descriptions_basic(self, dataframe_with_missing_descriptions):
        """Test suppression des descriptions manquantes"""
        result = EventsCleaner.remove_missing_descriptions(dataframe_with_missing_descriptions)
        
        assert len(result) == 2
        assert list(result['id']) == ['event-1', 'event-4']
    
    def test_remove_missing_descriptions_none_missing(self, sample_dataframe):
        """Test avec aucune description manquante"""
        original_len = len(sample_dataframe)
        result = EventsCleaner.remove_missing_descriptions(sample_dataframe)
        
        assert len(result) == original_len
    
    def test_remove_missing_descriptions_all_missing(self):
        """Test avec toutes les descriptions manquantes"""
        df = pd.DataFrame([
            {'id': 'event-1', 'description': ''},
            {'id': 'event-2', 'description': None},
            {'id': 'event-3', 'description': ''},
        ])
        result = EventsCleaner.remove_missing_descriptions(df)
        
        assert len(result) == 0
    
    def test_remove_missing_descriptions_empty_dataframe(self):
        """Test avec DataFrame vide"""
        df = pd.DataFrame(columns=['id', 'description'])
        result = EventsCleaner.remove_missing_descriptions(df)
        
        assert len(result) == 0


# ============================================================================
# Tests pour clean_pipeline
# ============================================================================

@pytest.mark.unit
class TestCleanPipeline:
    """Tests pour la méthode clean_pipeline (pipeline complet)"""
    
    def test_clean_pipeline_complete(self, sample_events_list, capsys):
        """Test du pipeline complet"""
        result = EventsCleaner.clean_pipeline(sample_events_list)
        captured = capsys.readouterr()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # Vérifie que les messages de log sont présents
        assert "Événements après extraction" in captured.out
        assert "Événements après suppression doublons" in captured.out
        assert "Événements après filtrage descriptions" in captured.out
    
    def test_clean_pipeline_with_duplicates(self, duplicate_events):
        """Test pipeline avec doublons"""
        result = EventsCleaner.clean_pipeline(duplicate_events)
        
        # Doit supprimer les doublons
        assert len(result) == 1
    
    def test_clean_pipeline_with_missing_descriptions(self, sample_event, sample_event_without_description):
        """Test pipeline avec descriptions manquantes"""
        events = [sample_event, sample_event_without_description]
        result = EventsCleaner.clean_pipeline(events)
        
        # Doit filtrer l'événement sans description
        assert len(result) == 1
        assert result.iloc[0]['id'] == 'event-123'
    
    def test_clean_pipeline_empty_list(self):
        """Test pipeline avec liste vide"""
        result = EventsCleaner.clean_pipeline([])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        # Vérifie que le DataFrame vide a toujours les bonnes colonnes
        expected_columns = [
            'id', 'title', 'description', 'location_name', 
            'location_city', 'date_start', 'date_end', 
            'url', 'keywords', 'category'
        ]
        assert list(result.columns) == expected_columns
    
    def test_clean_pipeline_all_filtered_out(self):
        """Test pipeline où tous les événements sont filtrés"""
        events = [
            {
                'uid': 'event-1',
                'title': {'fr': 'Event 1'},
                'description': None
            },
            {
                'uid': 'event-1',  # Doublon
                'title': {'fr': 'Event 1'},
                'description': ''
            }
        ]
        result = EventsCleaner.clean_pipeline(events)
        
        assert len(result) == 0
    
    def test_clean_pipeline_complex_scenario(self):
        """Test avec un scénario complexe"""
        events = [
            # Événement valide
            {
                'uid': 'event-1',
                'title': {'fr': 'Event 1'},
                'description': {'fr': 'Description 1'},
                'timings': [{'begin': '2025-12-15T20:00:00'}]
            },
            # Doublon
            {
                'uid': 'event-1',
                'title': {'fr': 'Event 1'},
                'description': {'fr': 'Description 1'},
                'timings': [{'begin': '2025-12-15T20:00:00'}]
            },
            # Sans description
            {
                'uid': 'event-2',
                'title': {'fr': 'Event 2'},
                'description': None,
                'timings': [{'begin': '2025-12-16T10:00:00'}]
            },
            # Événement valide
            {
                'uid': 'event-3',
                'title': {'fr': 'Event 3'},
                'description': {'fr': 'Description 3'},
                'timings': [{'begin': '2025-12-17T14:00:00'}]
            }
        ]
        
        result = EventsCleaner.clean_pipeline(events)
        
        # Doit garder seulement les 2 événements valides uniques
        assert len(result) == 2
        assert list(result['id']) == ['event-1', 'event-3']


# ============================================================================
# Tests d'intégrité des données
# ============================================================================

@pytest.mark.unit
class TestDataIntegrity:
    """Tests d'intégrité et de cohérence des données"""
    
    def test_column_names_are_correct(self, sample_events_list):
        """Vérifie que les noms de colonnes sont corrects"""
        result = EventsCleaner.extract_key_fields(sample_events_list)
        
        expected_columns = [
            'id', 'title', 'description', 'location_name', 
            'location_city', 'date_start', 'date_end', 
            'url', 'keywords', 'category'
        ]
        assert list(result.columns) == expected_columns
    
    def test_no_data_corruption_through_pipeline(self, sample_event):
        """Vérifie qu'aucune donnée n'est corrompue dans le pipeline"""
        events = [sample_event]
        result = EventsCleaner.clean_pipeline(events)
        
        assert result.iloc[0]['title'] == 'Concert de Jazz'
        assert result.iloc[0]['description'] == 'Un superbe concert de jazz au coeur de Paris'
        assert result.iloc[0]['location_name'] == 'Salle Pleyel'
    
    def test_dataframe_dtypes(self, sample_events_list):
        """Vérifie les types de données du DataFrame"""
        result = EventsCleaner.extract_key_fields(sample_events_list)
        
        # Toutes les colonnes doivent être des objets (strings)
        for col in result.columns:
            assert result[col].dtype == 'object' or pd.api.types.is_string_dtype(result[col])