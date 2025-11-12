import pytest
import pandas as pd
import json
from src.fetching.clean_events import EventsCleaner


@pytest.mark.unit
class TestExtractKeyFields:    
    def test_extract_single_event_complete(self, sample_event):
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
        result = EventsCleaner.extract_key_fields([sample_event_minimal])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['id'] == 'event-minimal'
        assert result.iloc[0]['title'] == 'Événement minimal'
        assert result.iloc[0]['description'] == 'Description minimale'
        assert result.iloc[0]['date_start'] == '2025-12-20T10:00:00'
    
    def test_extract_multiple_events(self, sample_events_list):
        result = EventsCleaner.extract_key_fields(sample_events_list)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result['id']) == ['event-123', 'event-minimal', 'event-456']
    
    def test_extract_empty_list(self):
        result = EventsCleaner.extract_key_fields([])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_extract_event_without_timings(self):
        event = {
            'uid': 'event-no-timing',
            'title': {'fr': 'Sans timing'},
            'description': {'fr': 'Description'},
        }
        result = EventsCleaner.extract_key_fields([event])
        
        assert len(result) == 1
        assert pd.isna(result.iloc[0]['date_start'])
    
    def test_extract_event_with_empty_timings(self):
        event = {
            'uid': 'event-empty-timing',
            'title': {'fr': 'Timing vide'},
            'description': {'fr': 'Description'},
            'timings': []
        }
        result = EventsCleaner.extract_key_fields([event])
        
        assert len(result) == 1
    
    def test_extract_event_with_firstTiming_fallback(self):
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
        event = {
            'uid': 'event-links',
            'title': {'fr': 'Event'},
            'description': {'fr': 'Desc'},
            'links': [{'link': 'https://test.com'}]
        }
        result = EventsCleaner.extract_key_fields([event])
        
        assert result.iloc[0]['url'] == 'https://test.com'
    
    def test_extract_event_with_empty_keywords(self):
        event = {
            'uid': 'event-no-keywords',
            'title': {'fr': 'Event'},
            'description': {'fr': 'Desc'},
            'keywords': {'fr': []}
        }
        result = EventsCleaner.extract_key_fields([event])
        
        assert result.iloc[0]['keywords'] == ''
    
    def test_extract_event_malformed_skipped(self, capsys):
        events = [
            {
                'uid': 'good-event',
                'title': {'fr': 'Good'},
                'description': {'fr': 'Desc'}
            },
            None, 
            "invalid",
            {
                'uid': 'another-good',
                'title': {'fr': 'Another'},
                'description': {'fr': 'Desc'}
            }
        ]
        
        result = EventsCleaner.extract_key_fields(events)
        captured = capsys.readouterr()
        
        assert "Erreur sur événement" in captured.out
        assert len(result) == 2
        assert list(result['id']) == ['good-event', 'another-good']


@pytest.mark.unit
class TestRemoveDuplicates:
    
    def test_remove_duplicates_basic(self, dataframe_with_duplicates):
        result = EventsCleaner.remove_duplicates(dataframe_with_duplicates)
        
        assert len(result) == 2
        assert list(result['id']) == ['event-1', 'event-2']
    
    def test_remove_duplicates_no_duplicates(self, sample_dataframe):
        original_len = len(sample_dataframe)
        result = EventsCleaner.remove_duplicates(sample_dataframe)
        
        assert len(result) == original_len
    
    def test_remove_duplicates_empty_dataframe(self):
        df = pd.DataFrame(columns=['id', 'title', 'description'])
        result = EventsCleaner.remove_duplicates(df)
        
        assert len(result) == 0
    
    def test_remove_duplicates_all_duplicates(self):
        df = pd.DataFrame([
            {'id': 'event-1', 'title': 'Event'},
            {'id': 'event-1', 'title': 'Event'},
            {'id': 'event-1', 'title': 'Event'},
        ])
        result = EventsCleaner.remove_duplicates(df)
        
        assert len(result) == 1



@pytest.mark.unit
class TestRemoveMissingDescriptions:
    
    def test_remove_missing_descriptions_basic(self, dataframe_with_missing_descriptions):
        result = EventsCleaner.remove_missing_descriptions(dataframe_with_missing_descriptions)
        
        assert len(result) == 2
        assert list(result['id']) == ['event-1', 'event-4']
    
    def test_remove_missing_descriptions_none_missing(self, sample_dataframe):
        original_len = len(sample_dataframe)
        result = EventsCleaner.remove_missing_descriptions(sample_dataframe)
        
        assert len(result) == original_len
    
    def test_remove_missing_descriptions_all_missing(self):
        df = pd.DataFrame([
            {'id': 'event-1', 'description': ''},
            {'id': 'event-2', 'description': None},
            {'id': 'event-3', 'description': ''},
        ])
        result = EventsCleaner.remove_missing_descriptions(df)
        
        assert len(result) == 0
    
    def test_remove_missing_descriptions_empty_dataframe(self):
        df = pd.DataFrame(columns=['id', 'description'])
        result = EventsCleaner.remove_missing_descriptions(df)
        
        assert len(result) == 0


@pytest.mark.unit
class TestCleanPipeline:    
    def test_clean_pipeline_with_duplicates(self, duplicate_events):
        result = EventsCleaner.clean_pipeline(duplicate_events)
        
        assert len(result) == 1
    
    def test_clean_pipeline_with_missing_descriptions(self, sample_event, sample_event_without_description):
        events = [sample_event, sample_event_without_description]
        result = EventsCleaner.clean_pipeline(events)
        
        assert len(result) == 1
        assert result.iloc[0]['id'] == 'event-123'
    
    def test_clean_pipeline_empty_list(self):
        result = EventsCleaner.clean_pipeline([])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

        expected_columns = [
            'id', 'title', 'description', 'location_name', 
            'location_city', 'date_start', 'date_end', 
            'url', 'keywords', 'category'
        ]
        assert list(result.columns) == expected_columns
    
    def test_clean_pipeline_all_filtered_out(self):

        events = [
            {
                'uid': 'event-1',
                'title': {'fr': 'Event 1'},
                'description': None
            },
            {
                'uid': 'event-1', 
                'title': {'fr': 'Event 1'},
                'description': ''
            }
        ]
        result = EventsCleaner.clean_pipeline(events)
        
        assert len(result) == 0
    
    def test_clean_pipeline_complex_scenario(self):
        events = [
            {
                'uid': 'event-1',
                'title': {'fr': 'Event 1'},
                'description': {'fr': 'Description 1'},
                'timings': [{'begin': '2025-12-15T20:00:00'}]
            },
            {
                'uid': 'event-1',
                'title': {'fr': 'Event 1'},
                'description': {'fr': 'Description 1'},
                'timings': [{'begin': '2025-12-15T20:00:00'}]
            },
            {
                'uid': 'event-2',
                'title': {'fr': 'Event 2'},
                'description': None,
                'timings': [{'begin': '2025-12-16T10:00:00'}]
            },
            {
                'uid': 'event-3',
                'title': {'fr': 'Event 3'},
                'description': {'fr': 'Description 3'},
                'timings': [{'begin': '2025-12-17T14:00:00'}]
            }
        ]
        
        result = EventsCleaner.clean_pipeline(events)
        
        assert len(result) == 2
        assert list(result['id']) == ['event-1', 'event-3']


@pytest.mark.unit
class TestDataIntegrity:    
    def test_column_names_are_correct(self, sample_events_list):
        result = EventsCleaner.extract_key_fields(sample_events_list)
        
        expected_columns = [
            'id', 'title', 'description', 'location_name', 
            'location_city', 'date_start', 'date_end', 
            'url', 'keywords', 'category'
        ]
        assert list(result.columns) == expected_columns
    
    def test_no_data_corruption_through_pipeline(self, sample_event):
        events = [sample_event]
        result = EventsCleaner.clean_pipeline(events)
        
        assert result.iloc[0]['title'] == 'Concert de Jazz'
        assert result.iloc[0]['description'] == 'Un superbe concert de jazz au coeur de Paris'
        assert result.iloc[0]['location_name'] == 'Salle Pleyel'
    
    def test_dataframe_dtypes(self, sample_events_list):
        result = EventsCleaner.extract_key_fields(sample_events_list)
        
        for col in result.columns:
            assert result[col].dtype == 'object' or pd.api.types.is_string_dtype(result[col])