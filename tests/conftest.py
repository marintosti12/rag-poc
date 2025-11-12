from unittest.mock import Mock
import pytest
import json
import os
from datetime import datetime
from typing import List, Dict
import pandas as pd



@pytest.fixture
def sample_event():
    return {
        'uid': 'event-123',
        'title': {'fr': 'Concert de Jazz'},
        'description': {'fr': 'Un superbe concert de jazz au coeur de Paris'},
        'timings': [
            {
                'begin': '2025-12-15T20:00:00',
                'end': '2025-12-15T23:00:00'
            }
        ],
        'locationName': 'Salle Pleyel',
        'locationCity': 'Paris',
        'canonicalUrl': 'https://example.com/concert-jazz',
        'keywords': {'fr': ['jazz', 'musique', 'concert']},
        'category': 'musique',
        'links': [{'link': 'https://example.com/concert-jazz'}]
    }


@pytest.fixture
def sample_event_minimal():
    return {
        'uid': 'event-minimal',
        'title': {'fr': 'Événement minimal'},
        'description': {'fr': 'Description minimale'},
        'timings': [{'begin': '2025-12-20T10:00:00'}]
    }


@pytest.fixture
def sample_event_without_description():
    return {
        'uid': 'event-no-desc',
        'title': {'fr': 'Événement sans description'},
        'description': None,
        'timings': [{'begin': '2025-12-20T10:00:00'}]
    }


@pytest.fixture
def sample_events_list(sample_event, sample_event_minimal):
    return [
        sample_event,
        sample_event_minimal,
        {
            'uid': 'event-456',
            'title': {'fr': 'Exposition d\'art'},
            'description': {'fr': 'Belle exposition au musée'},
            'timings': [
                {
                    'begin': '2025-12-16T10:00:00',
                    'end': '2025-12-16T18:00:00'
                }
            ],
            'locationName': 'Musée du Louvre',
            'locationCity': 'Paris',
            'canonicalUrl': 'https://example.com/expo-art',
            'keywords': {'fr': ['art', 'exposition', 'musée']},
            'category': 'exposition'
        }
    ]


@pytest.fixture
def duplicate_events(sample_event):
    event_copy = sample_event.copy()
    return [sample_event, event_copy, sample_event]


@pytest.fixture
def sample_agenda():
    return {
        'uid': 'agenda-paris-123',
        'title': {'fr': 'Agenda Culturel Paris'},
        'description': {'fr': 'Tous les événements culturels à Paris'},
        'slug': 'paris-culture'
    }


@pytest.fixture
def sample_agendas_list(sample_agenda):
    return [
        sample_agenda,
        {
            'uid': 'agenda-lyon-456',
            'title': {'fr': 'Agenda Lyon'},
            'description': {'fr': 'Événements à Lyon'},
            'slug': 'lyon-events'
        },
        {
            'uid': 'agenda-marseille-789',
            'title': 'Agenda Marseille', 
            'slug': 'marseille-events'
        }
    ]


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame([
        {
            'id': 'event-123',
            'title': 'Concert de Jazz',
            'description': 'Un superbe concert',
            'location_name': 'Salle Pleyel',
            'location_city': 'Paris',
            'date_start': '2025-12-15T20:00:00',
            'date_end': '2025-12-15T23:00:00',
            'url': 'https://example.com/concert',
            'keywords': 'jazz, musique',
            'category': 'musique'
        },
        {
            'id': 'event-456',
            'title': 'Exposition',
            'description': 'Belle exposition',
            'location_name': 'Musée',
            'location_city': 'Paris',
            'date_start': '2025-12-16T10:00:00',
            'date_end': '2025-12-16T18:00:00',
            'url': 'https://example.com/expo',
            'keywords': 'art, expo',
            'category': 'exposition'
        }
    ])


@pytest.fixture
def dataframe_with_duplicates():
    return pd.DataFrame([
        {'id': 'event-1', 'title': 'Event 1', 'description': 'Desc 1'},
        {'id': 'event-1', 'title': 'Event 1', 'description': 'Desc 1'},
        {'id': 'event-2', 'title': 'Event 2', 'description': 'Desc 2'},
    ])


@pytest.fixture
def dataframe_with_missing_descriptions():
    return pd.DataFrame([
        {'id': 'event-1', 'title': 'Event 1', 'description': 'Desc 1'},
        {'id': 'event-2', 'title': 'Event 2', 'description': ''},
        {'id': 'event-3', 'title': 'Event 3', 'description': None},
        {'id': 'event-4', 'title': 'Event 4', 'description': 'Desc 4'},
    ])


@pytest.fixture
def mock_api_key():
    return "test_api_key_123456789"


@pytest.fixture
def mock_env_vars(monkeypatch, mock_api_key):
    monkeypatch.setenv("OPENAGENDA_API_KEY", mock_api_key)


@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    
    return {
        'base': data_dir,
        'raw': raw_dir,
        'processed': processed_dir
    }


@pytest.fixture
def sample_raw_json_file(temp_data_dir, sample_events_list):
    filepath = temp_data_dir['raw'] / 'events_raw.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sample_events_list, f, ensure_ascii=False, indent=2)
    return filepath


@pytest.fixture
def mock_agendas_response():
    return {
        'agendas': [
            {
                'uid': 'agenda-1',
                'title': {'fr': 'Agenda Paris'},
                'slug': 'paris'
            },
            {
                'uid': 'agenda-2',
                'title': {'fr': 'Agenda Lyon'},
                'slug': 'lyon'
            }
        ],
        'total': 2
    }


@pytest.fixture
def mock_events_response():
    return {
        'events': [
            {
                'uid': 'event-1',
                'title': {'fr': 'Concert'},
                'description': {'fr': 'Super concert'},
                'timings': [{'begin': '2025-12-15T20:00:00'}],
                'locationName': 'Salle',
                'locationCity': 'Paris'
            }
        ],
        'total': 1
    }


@pytest.fixture
def freeze_time():
    from freezegun import freeze_time as _freeze_time
    with _freeze_time("2025-11-12 10:00:00"):
        yield

@pytest.fixture
def sample_chunks():
    return [
        {
            'text': 'Concert de jazz à Paris le 15 décembre',
            'id': 'event-1',
            'category': 'musique',
            'city': 'Paris'
        },
        {
            'text': 'Exposition d\'art moderne au musée',
            'id': 'event-2',
            'category': 'exposition',
            'city': 'Lyon'
        },
        {
            'text': 'Festival de théâtre en plein air',
            'id': 'event-3',
            'category': 'théâtre',
            'city': 'Marseille'
        }
    ]


@pytest.fixture
def sample_chunks_with_empty():
    return [
        {'text': 'Concert de jazz', 'id': 'event-1'},
        {'text': '', 'id': 'event-2'},  
        {'text': '   ', 'id': 'event-3'},  
        {'text': 'Exposition d\'art', 'id': 'event-4'},
    ]


@pytest.fixture
def chunks_without_text():
    return [
        {'id': 'event-1', 'category': 'musique'},
        {'id': 'event-2', 'category': 'exposition'}
    ]


@pytest.fixture
def mock_embeddings():
    mock = Mock()
    mock.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3]] * 3)
    mock.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    return mock


@pytest.fixture
def mock_vector_store():
    mock = Mock()
    mock.docstore = Mock()
    mock.docstore._dict = {'doc1': 'value1', 'doc2': 'value2'}
    mock.similarity_search_with_score = Mock(return_value=[])
    mock.add_texts = Mock()
    mock.save_local = Mock()
    return mock


@pytest.fixture
def temp_index_dir(tmp_path):
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    return str(index_dir)



