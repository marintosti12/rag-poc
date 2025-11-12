"""
Tests unitaires pour OpenAgendaFetcher
"""
import pytest
import responses
import json
from src.fetching.fetch_events import OpenAgendaFetcher


# ============================================================================
# Tests d'initialisation
# ============================================================================

@pytest.mark.unit
class TestInitialization:
    """Tests pour l'initialisation de OpenAgendaFetcher"""
    
    def test_init_with_api_key(self, mock_api_key):
        """Test initialisation avec clé API"""
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        
        assert fetcher.api_key == mock_api_key
        assert fetcher.base_url == "https://api.openagenda.com/v2"
        assert fetcher.headers['Authorization'] == f'Bearer {mock_api_key}'
        assert fetcher.headers['Content-Type'] == 'application/json'
    
    def test_init_without_api_key(self):
        """Test initialisation sans clé API"""
        fetcher = OpenAgendaFetcher(api_key="")
        
        assert fetcher.api_key == ""
        assert fetcher.headers['Authorization'] == 'Bearer '


# ============================================================================
# Tests pour test_connection
# ============================================================================

@pytest.mark.unit
class TestConnectionTest:
    """Tests pour la méthode test_connection"""
    
    @responses.activate
    def test_connection_success(self, mock_api_key, capsys):
        """Test connexion réussie"""
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas",
            json={'agendas': []},
            status=200
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.test_connection()
        captured = capsys.readouterr()
        
        assert result is True
        assert "✓ Connexion à l'API réussie" in captured.out
    
    @responses.activate
    def test_connection_authentication_error(self, mock_api_key, capsys):
        """Test erreur d'authentification"""
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas",
            json={'error': 'Unauthorized'},
            status=401
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.test_connection()
        captured = capsys.readouterr()
        
        assert result is False
        assert "✗ Erreur d'authentification (401)" in captured.out
    
    @responses.activate
    def test_connection_network_error(self, mock_api_key, capsys):
        """Test erreur réseau"""
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.test_connection()
        captured = capsys.readouterr()
        
        assert result is False
        assert "✗ Erreur de connexion" in captured.out


# ============================================================================
# Tests pour list_agendas
# ============================================================================

@pytest.mark.unit
class TestListAgendas:
    """Tests pour la méthode list_agendas"""
    
    @responses.activate
    def test_list_agendas_success(self, mock_api_key, mock_agendas_response, capsys):
        """Test liste agendas avec succès"""
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas",
            json=mock_agendas_response,
            status=200
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.list_agendas()
        captured = capsys.readouterr()
        
        assert len(result) == 2
        assert result[0]['uid'] == 'agenda-1'
        assert "✓ 2 agenda(s) trouvé(s)" in captured.out
        assert "Agenda Paris (UID: agenda-1)" in captured.out
    
    @responses.activate
    def test_list_agendas_with_search(self, mock_api_key):
        """Test recherche d'agendas avec terme de recherche"""
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas",
            json={'agendas': [{'uid': 'agenda-paris', 'title': {'fr': 'Paris Events'}}]},
            status=200
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.list_agendas(search="Paris")
        
        assert len(result) == 1
        # Vérifie que le paramètre search a été envoyé
        assert len(responses.calls) == 1
        assert 'search=Paris' in responses.calls[0].request.url
    
    @responses.activate
    def test_list_agendas_with_limit(self, mock_api_key, capsys):
        """Test limite d'affichage des agendas"""
        many_agendas = {
            'agendas': [
                {'uid': f'agenda-{i}', 'title': {'fr': f'Agenda {i}'}} 
                for i in range(20)
            ]
        }
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas",
            json=many_agendas,
            status=200
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.list_agendas(agendaLimit=5)
        captured = capsys.readouterr()
        
        assert len(result) == 20  # Tous les agendas retournés
        # Mais seulement 5 affichés dans les logs
        assert captured.out.count('UID: agenda-') == 5
    
    @responses.activate
    def test_list_agendas_with_string_title(self, mock_api_key, capsys):
        """Test agenda avec titre string au lieu de dict"""
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas",
            json={'agendas': [{'uid': 'agenda-1', 'title': 'Simple Title'}]},
            status=200
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.list_agendas()
        captured = capsys.readouterr()
        
        assert len(result) == 1
        assert "Simple Title (UID: agenda-1)" in captured.out
    
    @responses.activate
    def test_list_agendas_empty_response(self, mock_api_key, capsys):
        """Test réponse vide"""
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas",
            json={'agendas': []},
            status=200
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.list_agendas()
        captured = capsys.readouterr()
        
        assert result == []
        assert "✓ 0 agenda(s) trouvé(s)" in captured.out
    
    @responses.activate
    def test_list_agendas_request_error(self, mock_api_key, capsys):
        """Test erreur de requête"""
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas",
            json={'error': 'Bad request'},
            status=400
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.list_agendas()
        captured = capsys.readouterr()
        
        assert result == []
        assert "Erreur lors de la récupération des agendas" in captured.out


# ============================================================================
# Tests pour fetch_events
# ============================================================================

@pytest.mark.unit
class TestFetchEvents:
    """Tests pour la méthode fetch_events"""
    
    @responses.activate
    def test_fetch_events_success(self, mock_api_key, mock_events_response, capsys):
        """Test récupération d'événements avec succès"""
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas/agenda-1/events",
            json=mock_events_response,
            status=200
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.fetch_events(agenda_uid="agenda-1")
        captured = capsys.readouterr()
        
        assert len(result) == 1
        assert result[0]['uid'] == 'event-1'
        assert "✓ Page 1 : 1 événements récupérés" in captured.out
        assert "✓ Total : 1 événements récupérés" in captured.out
    
    @responses.activate
    def test_fetch_events_with_date_filters(self, mock_api_key, mock_events_response):
        """Test avec filtres de dates"""
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas/agenda-1/events",
            json=mock_events_response,
            status=200
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.fetch_events(
            agenda_uid="agenda-1",
            date_start="2025-12-01",
            date_end="2025-12-31"
        )
        
        assert len(result) == 1
        # Vérifie que les paramètres de date ont été envoyés
        request_url = responses.calls[0].request.url
        assert 'timings%5Bgte%5D=2025-12-01' in request_url
        assert 'timings%5Blte%5D=2025-12-31' in request_url
    
    @responses.activate
    def test_fetch_events_with_limit(self, mock_api_key):
        """Test avec limite d'événements"""
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas/agenda-1/events",
            json={'events': [{'uid': f'event-{i}'} for i in range(50)], 'total': 50},
            status=200
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.fetch_events(agenda_uid="agenda-1", limit=50)
        
        assert len(result) == 50
    
    @responses.activate
    def test_fetch_events_pagination(self, mock_api_key, capsys):
        """Test pagination avec plusieurs pages"""
        # Page 1
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas/agenda-1/events",
            json={
                'events': [{'uid': f'event-{i}'} for i in range(100)],
                'total': 150
            },
            status=200
        )
        # Page 2
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas/agenda-1/events",
            json={
                'events': [{'uid': f'event-{i}'} for i in range(100, 150)],
                'total': 150
            },
            status=200
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.fetch_events(agenda_uid="agenda-1", limit=150)
        captured = capsys.readouterr()
        
        assert len(result) == 150
        assert "✓ Page 1 : 100 événements récupérés" in captured.out
        assert "✓ Page 2 : 50 événements récupérés" in captured.out
    
    @responses.activate
    def test_fetch_events_empty_page(self, mock_api_key, capsys):
        """Test page vide"""
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas/agenda-1/events",
            json={'events': [], 'total': 0},
            status=200
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.fetch_events(agenda_uid="agenda-1")
        captured = capsys.readouterr()
        
        assert result == []
        assert "Aucun événement supplémentaire" in captured.out
    
    @responses.activate
    def test_fetch_events_api_error(self, mock_api_key, capsys):
        """Test erreur API"""
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas/agenda-1/events",
            json={'error': 'Agenda not found'},
            status=200
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.fetch_events(agenda_uid="agenda-1")
        captured = capsys.readouterr()
        
        assert result == []
        assert "Erreur API" in captured.out
    
    @responses.activate
    def test_fetch_events_http_error(self, mock_api_key, capsys):
        """Test erreur HTTP"""
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas/agenda-1/events",
            json={'error': 'Not found'},
            status=404
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.fetch_events(agenda_uid="agenda-1")
        captured = capsys.readouterr()
        
        assert result == []
        assert "Erreur HTTP 404" in captured.out
    
    @responses.activate
    def test_fetch_events_max_pages_limit(self, mock_api_key, capsys):
        """Test limite maximale de pages (10)"""
        # Simule 15 pages disponibles
        for i in range(15):
            responses.add(
                responses.GET,
                "https://api.openagenda.com/v2/agendas/agenda-1/events",
                json={
                    'events': [{'uid': f'event-{i * 100 + j}'} for j in range(100)],
                    'total': 1500
                },
                status=200
            )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.fetch_events(agenda_uid="agenda-1", limit=2000)
        
        # Doit s'arrêter après 11 pages (page 0 à 10)
        assert len(result) <= 1100
        assert len(responses.calls) <= 11


# ============================================================================
# Tests pour fetch_events_from_multiple_agendas
# ============================================================================

@pytest.mark.unit
class TestFetchEventsFromMultipleAgendas:
    """Tests pour la méthode fetch_events_from_multiple_agendas"""
    
    @responses.activate
    def test_fetch_from_multiple_agendas_success(self, mock_api_key, capsys):
        """Test récupération depuis plusieurs agendas"""
        # Mock list_agendas
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas",
            json={
                'agendas': [
                    {'uid': 'agenda-1', 'title': {'fr': 'Agenda 1'}},
                    {'uid': 'agenda-2', 'title': {'fr': 'Agenda 2'}}
                ]
            },
            status=200
        )
        
        # Mock fetch_events pour agenda-1
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas/agenda-1/events",
            json={'events': [{'uid': 'event-1', 'title': {'fr': 'Event 1'}}], 'total': 1},
            status=200
        )
        
        # Mock fetch_events pour agenda-2
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas/agenda-2/events",
            json={'events': [{'uid': 'event-2', 'title': {'fr': 'Event 2'}}], 'total': 1},
            status=200
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.fetch_events_from_multiple_agendas(
            location="Paris",
            date_start="2025-12-01",
            date_end="2025-12-31",
            agendaLimit=2
        )
        captured = capsys.readouterr()
        
        assert len(result) == 2
        assert result[0]['source_agenda'] == 'Agenda 1'
        assert result[1]['source_agenda'] == 'Agenda 2'
        assert "🔍 Recherche d'agendas pour : Paris" in captured.out
    
    @responses.activate
    def test_fetch_from_multiple_agendas_no_agendas(self, mock_api_key, capsys):
        """Test avec aucun agenda trouvé"""
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas",
            json={'agendas': []},
            status=200
        )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.fetch_events_from_multiple_agendas(
            location="Unknown",
            date_start="2025-12-01",
            date_end="2025-12-31"
        )
        captured = capsys.readouterr()
        
        assert result == []
        assert "⚠️ Aucun agenda trouvé pour Unknown" in captured.out
    
    @responses.activate
    def test_fetch_from_multiple_agendas_respects_limit(self, mock_api_key):
        """Test que la limite d'agendas est respectée"""
        responses.add(
            responses.GET,
            "https://api.openagenda.com/v2/agendas",
            json={
                'agendas': [
                    {'uid': f'agenda-{i}', 'title': {'fr': f'Agenda {i}'}} 
                    for i in range(10)
                ]
            },
            status=200
        )
        
        # Mock pour 3 agendas seulement
        for i in range(3):
            responses.add(
                responses.GET,
                f"https://api.openagenda.com/v2/agendas/agenda-{i}/events",
                json={'events': [{'uid': f'event-{i}'}], 'total': 1},
                status=200
            )
        
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        result = fetcher.fetch_events_from_multiple_agendas(
            location="Test",
            date_start="2025-12-01",
            date_end="2025-12-31",
            agendaLimit=3
        )
        
        # Seulement 3 événements (1 par agenda)
        assert len(result) == 3


# ============================================================================
# Tests pour save_raw_data
# ============================================================================

@pytest.mark.unit
class TestSaveRawData:
    """Tests pour la méthode save_raw_data"""
    
    def test_save_raw_data_success(self, mock_api_key, sample_events_list, tmp_path, capsys):
        """Test sauvegarde réussie"""
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        filepath = tmp_path / "data" / "raw" / "test_events.json"
        
        fetcher.save_raw_data(sample_events_list, str(filepath))
        captured = capsys.readouterr()
        
        assert filepath.exists()
        assert f"✓ 3 événements sauvegardés dans {filepath}" in captured.out
        
        # Vérifie le contenu
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert len(data) == 3
        assert data[0]['uid'] == 'event-123'
    
    def test_save_raw_data_creates_directory(self, mock_api_key, sample_events_list, tmp_path):
        """Test que les dossiers sont créés automatiquement"""
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        filepath = tmp_path / "new_dir" / "sub_dir" / "events.json"
        
        fetcher.save_raw_data(sample_events_list, str(filepath))
        
        assert filepath.exists()
        assert filepath.parent.exists()
    
    def test_save_raw_data_empty_list(self, mock_api_key, tmp_path):
        """Test sauvegarde d'une liste vide"""
        fetcher = OpenAgendaFetcher(api_key=mock_api_key)
        filepath = tmp_path / "empty.json"
        
        fetcher.save_raw_data([], str(filepath))
        
        assert filepath.exists()
        with open(filepath, 'r') as f:
            data = json.load(f)
        assert data == []