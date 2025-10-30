from typing import List, Dict
import re

class EventTextSplitter:    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                last_break = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end),
                    text.rfind('\n', start, end)
                )
                
                if last_break > start:
                    end = last_break + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap if end < len(text) else end
        
        return chunks
    
    def create_event_chunks(self, event: Dict) -> List[Dict]:
        title = event.get('title', '')
        description = event.get('description', '')
        
        full_text = f"{title}\n\n{description}"
        
        text_chunks = self.split_text(full_text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                'text': chunk_text,
                'chunk_id': i,
                'total_chunks': len(text_chunks),
                'event_id': event.get('id'),
                'event_title': title,
                'location_city': event.get('location_city', ''),
                'location_name': event.get('location_name', ''),
                'date_start': event.get('date_start', ''),
                'date_end': event.get('date_end', ''),
                'url': event.get('url', ''),
                'keywords': event.get('keywords', ''),
                'category': event.get('category', ''),
                'source_agenda': event.get('source_agenda', ''),
            }
            chunks.append(chunk)
        
        return chunks
    
    def process_events(self, events: List[Dict]) -> List[Dict]:
        all_chunks = []
        
        for event in events:
            chunks = self.create_event_chunks(event)
            all_chunks.extend(chunks)
        
        return all_chunks
