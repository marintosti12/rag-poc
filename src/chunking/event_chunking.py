from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import pandas as pd

class EventChunking:
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 separators: Optional[List[str]] = None):
     
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            separators = [
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                "; ",
                ", ",
                " ",
                ""
            ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )
    
    @staticmethod
    def format_date(date_str: str) -> str:
        if not date_str:
            return "Date non précisée"
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            months_fr = {
                1: 'janvier', 2: 'février', 3: 'mars', 4: 'avril',
                5: 'mai', 6: 'juin', 7: 'juillet', 8: 'août',
                9: 'septembre', 10: 'octobre', 11: 'novembre', 12: 'décembre'
            }
            return f"{dt.day} {months_fr[dt.month]} {dt.year}"
        except Exception:
            return date_str
    
    def build_event_text(self, event: Dict) -> str:
        parts = []
        
        title = event.get('title', '').strip()
        if title:
            parts.append(f"Événement: {title}")
        
        date_start = event.get('date_start', '')
        if date_start:
            # affichage lisible
            date_readable = self.format_date(date_start)
            parts.append(f"Date: {date_readable}")

            # ==== PATCH MINIMAL: signaux explicites pour les requêtes "année 2024" ====
            try:
                dt = datetime.fromisoformat(date_start.replace('Z', '+00:00'))
                parts.append(f"Année: {dt.year}")        # ex: Année: 2025
                parts.append(f"{dt.year}:")               # ex: 2025: (jeton isolé)
            except Exception:
                # si parsing impossible, on tente quand même d’offrir un signal ISO
                pass

            # toujours utile pour les correspondances exactes
            parts.append(f"DateISO: {date_start}")
            # ==== FIN PATCH ====
        
        location_parts = []
        location_name = event.get('location_name', '').strip()
        location_city = event.get('location_city', '').strip()
        
        if location_city:
            location_parts.append(location_city)
        if location_name:
            location_parts.append(location_name)
        
        if location_parts:
            parts.append(f"Lieu: {' - '.join(location_parts)}")
        
        category = event.get('category', '').strip()
        if category:
            parts.append(f"Catégorie: {category}")
        
        keywords = event.get('keywords', '').strip()
        if keywords:
            parts.append(f"Tags: {keywords}")
        
        description = event.get('description', '').strip()
        if description:
            parts.append(f"\nDescription:\n{description}")
        
        return "\n".join(parts)
    
    def create_chunks(self, event: Dict) -> List[Dict]:
        full_text = self.build_event_text(event)
        text_chunks = self.text_splitter.split_text(full_text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            year = None
            ym = None
            ds = event.get('date_start', '')
            try:
                dt = datetime.fromisoformat(ds.replace('Z', '+00:00'))
                year = dt.year
                ym = f"{dt.year}-{dt.month:02d}"
            except Exception:
                pass

            chunk = {
                'text': chunk_text,
                'chunk_id': i,
                'total_chunks': len(text_chunks),
                'event_id': event.get('id'),
                'title': event.get('title', ''),
                'location_city': event.get('location_city', ''),
                'location_name': event.get('location_name', ''),
                'date_start': event.get('date_start', ''),
                'date_end': event.get('date_end', ''),
                'category': event.get('category', ''),
                'url': event.get('url', ''),
                'keywords': event.get('keywords', ''),
                'source_agenda': event.get('source_agenda', ''),
                'year': year,
                'ym': ym,
            }
            chunks.append(chunk)
        
        return chunks
    
    def process_events(self, events: List[Dict]) -> List[Dict]:
        all_chunks = []
        skipped = 0
        
        for event in events:
            if not event.get('title'):
                skipped += 1
                continue
            try:
                chunks = self.create_chunks(event)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"⚠️  Erreur sur événement {event.get('id', 'unknown')}: {e}")
                skipped += 1
                continue
        
        print(f"✓ {len(all_chunks)} chunks créés depuis {len(events)} événements")
        if skipped > 0:
            print(f"⚠️  {skipped} événements ignorés")
        return all_chunks
