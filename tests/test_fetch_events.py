import json
import pytest
from src.openAgenda.fetch_events import OpenAgendaFetcher
from src.openAgenda.clean_events import EventsCleaner

def test_fetcher_initialization():
    """Test l'initialisation du fetcher"""
    fetcher = OpenAgendaFetcher("test_key")
    assert fetcher.api_key == "test_key"
    assert fetcher.base_url is not None

def test_events_structure():
    """Vérifie la structure des événements récupérés"""
    # Charger un échantillon de données
    with open('data/raw/events_raw.json', 'r') as f:
        events = json.load(f)
    
    assert len(events) > 0
    assert 'uid' in events[0]
    assert 'title' in events[0]

def test_cleaning_pipeline():
    """Test le pipeline de nettoyage"""
    with open('data/raw/events_raw.json', 'r') as f:
        events = json.load(f)
    
    df = EventsCleaner.clean_pipeline(events)
    
    # Vérifications
    assert len(df) > 0
    assert 'description' in df.columns
    assert df['description'].isna().sum() == 0  # Pas de NaN
    assert len(df) == len(df.drop_duplicates(subset=['id']))  # Pas de doublons