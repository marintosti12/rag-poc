"""
Configuration pytest et fixtures réutilisables
"""
import pytest
import json
import os
from datetime import datetime
from typing import List, Dict
import pandas as pd


# ============================================================================
# FIXTURES - Données de test
# ============================================================================

@pytest.fixture
def sample_event():
    """Un événement unique pour les tests"""
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
    """Événement avec données minimales"""
    return {
        'uid': 'event-minimal',
        'title': {'fr': 'Événement minimal'},
        'description': {'fr': 'Description minimale'},
        'timings': [{'begin': '2025-12-20T10:00:00'}]
    }


@pytest.fixture
def sample_event_without_description():
    """Événement sans description"""
    return {
        'uid': 'event-no-desc',
        'title': {'fr': 'Événement sans description'},
        'description': None,
        'timings': [{'begin': '2025-12-20T10:00:00'}]
    }


@pytest.fixture
def sample_events_list(sample_event, sample_event_minimal):
    """Liste d'événements pour les tests"""
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
    """Liste avec des événements dupliqués"""
    event_copy = sample_event.copy()
    return [sample_event, event_copy, sample_event]


@pytest.fixture
def sample_agenda():
    """Un agenda pour les tests"""
    return {
        'uid': 'agenda-paris-123',
        'title': {'fr': 'Agenda Culturel Paris'},
        'description': {'fr': 'Tous les événements culturels à Paris'},
        'slug': 'paris-culture'
    }


@pytest.fixture
def sample_agendas_list(sample_agenda):
    """Liste d'agendas"""
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
            'title': 'Agenda Marseille',  # String direct
            'slug': 'marseille-events'
        }
    ]


# ============================================================================
# FIXTURES - DataFrames
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """DataFrame nettoyé pour les tests"""
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
    """DataFrame avec doublons"""
    return pd.DataFrame([
        {'id': 'event-1', 'title': 'Event 1', 'description': 'Desc 1'},
        {'id': 'event-1', 'title': 'Event 1', 'description': 'Desc 1'},
        {'id': 'event-2', 'title': 'Event 2', 'description': 'Desc 2'},
    ])


@pytest.fixture
def dataframe_with_missing_descriptions():
    """DataFrame avec descriptions manquantes"""
    return pd.DataFrame([
        {'id': 'event-1', 'title': 'Event 1', 'description': 'Desc 1'},
        {'id': 'event-2', 'title': 'Event 2', 'description': ''},
        {'id': 'event-3', 'title': 'Event 3', 'description': None},
        {'id': 'event-4', 'title': 'Event 4', 'description': 'Desc 4'},
    ])


# ============================================================================
# FIXTURES - Configuration et API
# ============================================================================

@pytest.fixture
def mock_api_key():
    """Clé API fictive pour les tests"""
    return "test_api_key_123456789"


@pytest.fixture
def mock_env_vars(monkeypatch, mock_api_key):
    """Configure les variables d'environnement pour les tests"""
    monkeypatch.setenv("OPENAGENDA_API_KEY", mock_api_key)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Crée une structure de dossiers temporaire pour les tests"""
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
    """Crée un fichier JSON de test"""
    filepath = temp_data_dir['raw'] / 'events_raw.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sample_events_list, f, ensure_ascii=False, indent=2)
    return filepath


# ============================================================================
# FIXTURES - Responses mock pour API
# ============================================================================

@pytest.fixture
def mock_agendas_response():
    """Réponse API simulée pour list_agendas"""
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
    """Réponse API simulée pour fetch_events"""
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


# ============================================================================
# FIXTURES - Utilitaires
# ============================================================================

@pytest.fixture
def freeze_time():
    """Fixture pour figer le temps dans les tests"""
    from freezegun import freeze_time as _freeze_time
    with _freeze_time("2025-11-12 10:00:00"):
        yield


# Fixture désactivée car elle cause des problèmes avec matplotlib
# @pytest.fixture(autouse=True)
# def reset_pandas_options():
#     """Réinitialise les options pandas après chaque test"""
#     yield
#     pd.reset_option('all')