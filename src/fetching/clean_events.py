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
                uid = event.get('uid') if event and isinstance(event, dict) else 'inconnu'
                print(f"Erreur sur événement {uid}: {e}")
                continue
        
        columns = [
            'id', 'title', 'description', 'location_name', 
            'location_city', 'date_start', 'date_end', 
            'url', 'keywords', 'category'
        ]
        
        if not cleaned:
            return pd.DataFrame(columns=columns)
                
        return pd.DataFrame(cleaned, columns=columns)
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.drop_duplicates(subset=['id'])
    
    @staticmethod
    def remove_missing_descriptions(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'description' not in df.columns:
            return df
        return df[df['description'].notna() & (df['description'] != '')]
    
    @staticmethod
    def clean_pipeline(events: List[Dict]) -> pd.DataFrame:
        df = EventsCleaner.extract_key_fields(events)
        
        df = EventsCleaner.remove_duplicates(df)
        
        df = EventsCleaner.remove_missing_descriptions(df)
        
        return df
