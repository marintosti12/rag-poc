import requests
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class OpenAgendaFetcher:    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openagenda.com/v2"
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def list_agendas(self, search: Optional[str] = None, agendaLimit: int = 10) -> List[Dict]:
        url = f"{self.base_url}/agendas"
        params = {}
        
        if search:
            params['search'] = search
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            agendas = data.get('agendas', [])
            print(f"✓ {len(agendas)} agenda(s) trouvé(s)")
            
            for agenda in agendas[:agendaLimit]:  
                title = agenda.get('title')
                if isinstance(title, dict):
                    title = title.get('fr') or title.get('en') or 'Sans titre'
                elif not title:
                    title = 'Sans titre'
                
                uid = agenda.get('uid', 'N/A')
                print(f"  - {title} (UID: {uid})")
            
            return agendas
            
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la récupération des agendas : {e}")
            return []
    
    def fetch_events(self, 
                     agenda_uid: str,
                     date_start: Optional[str] = None,
                     date_end: Optional[str] = None,
                     limit: int = 100) -> List[Dict]:
        
        url = f"{self.base_url}/agendas/{agenda_uid}/events"
        
        params = {
            'size': min(limit, 100),
            'detailed': 1,
            'timings[tz]': 'Europe/Paris', 
        }
        
        if date_start:
            params['timings[gte]'] = date_start
        if date_end:
            params['timings[lte]'] = date_end
        
        events = []
        page = 1
        after = None
        max_pages = 11 
        
        while len(events) < limit and page <= max_pages:
            try:
                if after:
                    params['after'] = after
                
                resp = requests.get(url, headers=self.headers, params=params)
                
                if resp.status_code != 200:
                    print(f"Erreur HTTP {resp.status_code} : {resp.text}")
                    break
                
                data = resp.json()
                
                if 'error' in data:
                    print(f"Erreur API : {data.get('error')}")
                    break
                
                page_events = data.get('events', [])
                
                if not page_events:
                    print("Aucun événement supplémentaire trouvé.")
                    break
                
                print(f"✓ Page {page} : {len(page_events)} événements récupérés")
                
                events.extend(page_events)
                
                links = data.get('links', {})
                next_link = links.get('next', {})
                after = next_link.get('after')
                
                if not after:
                    break
                
                page += 1
                    
            except requests.exceptions.HTTPError as e:
                print(f"Erreur HTTP {e.response.status_code} : {e.response.text}")
                break
            except requests.exceptions.RequestException as e:
                print(f"Erreur lors de la récupération : {e}")
                break
        
        final_events = events[:limit]
        print(f"\n✓ Total : {len(final_events)} événements récupérés")
        return final_events
    
    def fetch_events_from_multiple_agendas(self,
                                          location: str,
                                          date_start: str,
                                          date_end: str,
                                          limit: int = 100, 
                                          agendaLimit: int = 5) -> List[Dict]:
        
        print(f"\n🔍 Recherche d'agendas pour : {location}")
        agendas = self.list_agendas(search=location, agendaLimit=20)
        
        if not agendas:
            print(f"⚠️ Aucun agenda trouvé pour {location}")
            return []
        
        all_events = []
        
        for agenda in agendas[:agendaLimit]:
            agenda_uid = agenda.get('uid')
            title = agenda.get('title')
            if isinstance(title, dict):
                agenda_title = title.get('fr') or title.get('en') or 'Sans titre'
            elif isinstance(title, str):
                agenda_title = title
            else:
                agenda_title = 'Sans titre'
            
            print(f"\n📥 Récupération des événements de : {agenda_title}")
            events = self.fetch_events(
                agenda_uid=agenda_uid,
                date_start=date_start,
                date_end=date_end,
                limit=limit
            )
            
            for event in events:
                event['source_agenda'] = agenda_title
                event['source_agenda_uid'] = agenda_uid
            
            all_events.extend(events)
        
        return all_events
    
    def save_raw_data(self, events: List[Dict], filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        print(f"✓ {len(events)} événements sauvegardés dans {filepath}")
    
    def test_connection(self):
        try:
            response = requests.get(
                f"{self.base_url}/agendas",
                headers=self.headers,
                params={'size': 1}
            )
            response.raise_for_status()
            print("✓ Connexion à l'API réussie")
            return True
        except requests.exceptions.HTTPError as e:
            print(f"✗ Erreur d'authentification ({e.response.status_code})")
            print(f"Réponse : {e.response.text}")
            return False
        except Exception as e:
            print(f"✗ Erreur de connexion : {e}")
            return False