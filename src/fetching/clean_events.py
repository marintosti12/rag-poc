import pandas as pd
import json
from typing import List, Dict

class EventsCleaner:
    @staticmethod
    def extract_key_fields(events: List[Dict]) -> pd.DataFrame:
        cleaned = []
        
        for event in events:
            try:
                t0 = (event.get('timings') or [{}])[0]
                date_start = t0.get('begin') or (event.get('firstTiming') or {}).get('begin')
                date_end   = t0.get('end')   or (event.get('lastTiming')  or {}).get('end')

                cleaned_event = {
                    'id': event.get('uid'),
                    'title': (event.get('title') or {}).get('fr', ''),
                    'description': (event.get('description') or {}).get('fr', ''),
                    'location_name': event.get('locationName', '') or (event.get('location') or {}).get('name', ''),
                    'location_city': event.get('locationCity', '') or (event.get('location') or {}).get('city', ''),
                    'date_start': date_start,
                    'date_end': date_end,
                    'url': event.get('canonicalUrl', '') or ((event.get('links') or [{}])[0] or {}).get('link', ''),
                    'keywords': ', '.join((event.get('keywords') or {}).get('fr', [])),
                    'category': event.get('category', ''),
                }

                cleaned.append(cleaned_event)
            except Exception as e:
                print(f"Erreur sur événement {event.get('uid')}: {e}")
                continue
                
        return pd.DataFrame(cleaned)
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les doublons"""
        return df.drop_duplicates(subset=['id'])
    
    @staticmethod
    def remove_missing_descriptions(df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les événements sans description"""
        return df[df['description'].notna() & (df['description'] != '')]
    
    @staticmethod
    def clean_pipeline(events: List[Dict]) -> pd.DataFrame:
        """Pipeline complet de nettoyage"""
        df = EventsCleaner.extract_key_fields(events)
        print(f"Événements après extraction : {len(df)}")
        
        df = EventsCleaner.remove_duplicates(df)
        print(f"Événements après suppression doublons : {len(df)}")
        
        df = EventsCleaner.remove_missing_descriptions(df)
        print(f"Événements après filtrage descriptions : {len(df)}")
        
        return df


if __name__ == "__main__":
    # Test
    with open('data/raw/events_raw.json', 'r') as f:
        events = json.load(f)
    
    cleaner = EventsCleaner()
    df_clean = cleaner.clean_pipeline(events)
    
    # Sauvegarde
    df_clean.to_csv('data/processed/events_clean.csv', index=False)
    df_clean.to_json('data/processed/events_clean.json', 
                     orient='records', force_ascii=False, indent=2)
    
    print(f"\n✓ Données nettoyées sauvegardées")
    print(f"Colonnes : {list(df_clean.columns)}")