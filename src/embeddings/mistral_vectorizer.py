from mistralai import Mistral
import numpy as np
from typing import List, Dict
import json
import os
from dotenv import load_dotenv
import time

load_dotenv()

class MistralVectorizer:
    """Classe pour vectoriser avec l'API Mistral"""
    
    def __init__(self, api_key: str = None):
        """
        Initialise le vectorizer Mistral
        
        Args:
            api_key: Clé API Mistral (ou depuis .env)
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY non trouvée. Ajoutez-la dans .env")
        
        self.client = Mistral(api_key=self.api_key)
        self.model_name = "mistral-embed"
        self.embedding_dimension = 1024  # Dimension de mistral-embed
        
        print(f"✓ Client Mistral initialisé")
        print(f"  Modèle : {self.model_name}")
        print(f"  Dimension : {self.embedding_dimension}")
    
    def create_text_for_embedding(self, event: Dict) -> str:
        """
        Crée un texte optimisé pour l'embedding
        
        Args:
            event: Dictionnaire d'événement
            
        Returns:
            Texte combiné
        """
        parts = []
        
        if event.get('title'):
            parts.append(f"Titre: {event['title']}")
        
        if event.get('description'):
            # Limiter la description à 500 caractères pour l'API
            desc = event['description'][:500]
            parts.append(f"Description: {desc}")
        
        if event.get('location_city'):
            parts.append(f"Lieu: {event['location_city']}")
        
        if event.get('location_name'):
            parts.append(f"à {event['location_name']}")
        
        if event.get('keywords'):
            parts.append(f"Mots-clés: {event['keywords']}")
        
        if event.get('category'):
            parts.append(f"Catégorie: {event['category']}")
        
        return " | ".join(parts)
    
    def vectorize_events(self, 
                        events: List[Dict], 
                        batch_size: int = 10) -> np.ndarray:
        """
        Vectorise une liste d'événements avec l'API Mistral
        
        Args:
            events: Liste d'événements
            batch_size: Nombre d'événements par batch (max 100 selon API)
            
        Returns:
            Array numpy de vecteurs
        """
        print(f"\n🔢 Vectorisation de {len(events)} événements avec Mistral...")
        
        # Créer les textes
        texts = [self.create_text_for_embedding(event) for event in events]
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"  Batch {batch_num}/{total_batches} : {len(batch)} textes...")
            
            try:
                # Appel API Mistral
                response = self.client.embeddings.create(
                    model=self.model_name,
                    inputs=batch
                )
                
                # Extraire les embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                print(f"  ✓ Batch {batch_num} terminé")
                
                # Pause pour respecter les limites de taux
                if i + batch_size < len(texts):
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"  ❌ Erreur sur batch {batch_num}: {e}")
                # En cas d'erreur, ajouter des vecteurs nuls
                for _ in range(len(batch)):
                    all_embeddings.append([0.0] * self.embedding_dimension)
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        print(f"✓ Vectorisation terminée. Shape: {embeddings.shape}")
        
        return embeddings
    
    def save_embeddings(self, 
                       embeddings: np.ndarray, 
                       events: List[Dict],
                       output_dir: str = "data/processed"):
        """
        Sauvegarde les embeddings et métadonnées
        
        Args:
            embeddings: Array numpy des vecteurs
            events: Liste des événements
            output_dir: Dossier de sortie
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarder les embeddings
        embeddings_path = os.path.join(output_dir, "mistral_embeddings.npy")
        np.save(embeddings_path, embeddings)
        print(f"✓ Embeddings sauvegardés : {embeddings_path}")
        
        # Sauvegarder les métadonnées
        metadata_path = os.path.join(output_dir, "events_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        print(f"✓ Métadonnées sauvegardées : {metadata_path}")
        
        # Infos du modèle
        model_info = {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'num_events': len(events),
            'shape': list(embeddings.shape)
        }
        
        model_info_path = os.path.join(output_dir, "model_info.json")
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)
        print(f"✓ Infos du modèle sauvegardées : {model_info_path}")
    
    def test_connection(self):
        """Teste la connexion à l'API Mistral"""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                inputs=["test"]
            )
            print("✓ Connexion à l'API Mistral réussie")
            return True
        except Exception as e:
            print(f"✗ Erreur de connexion : {e}")
            return False


if __name__ == "__main__":
    import pandas as pd
    
    # Test de connexion
    print("🔌 Test de connexion à l'API Mistral...")
    vectorizer = MistralVectorizer()
    
    if not vectorizer.test_connection():
        print("❌ Impossible de se connecter. Vérifiez votre clé API.")
        exit(1)
    
    # Charger les événements
    print("\n📂 Chargement des événements...")
    events_file = 'data/processed/events_clean.json'
    
    if not os.path.exists(events_file):
        print(f"❌ Fichier non trouvé : {events_file}")
        exit(1)
    
    with open(events_file, 'r') as f:
        events = json.load(f)
    
    print(f"✓ {len(events)} événements chargés")
    
    # Vectoriser
    embeddings = vectorizer.vectorize_events(events, batch_size=10)
    
    # Sauvegarder
    vectorizer.save_embeddings(embeddings, events)
    
    print("\n✅ Vectorisation avec Mistral terminée !")
    print(f"📊 Statistiques :")
    print(f"  - Événements : {len(events)}")
    print(f"  - Dimension : {embeddings.shape[1]}")
    print(f"  - Taille : {embeddings.nbytes / 1024 / 1024:.2f} MB")