from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from typing import Any, List, Dict, Tuple, Optional
import json
import os
from dotenv import load_dotenv

load_dotenv()

class FAISSVectorStore:
    """
    Gestionnaire de base vectorielle FAISS pour la recherche sémantique d'événements.
    
    Supporte deux types d'embeddings :
    - Mistral AI (API, meilleure qualité pour le français)
    - HuggingFace (local, gratuit)
    """
    
    def __init__(self, 
                 embedding_provider: str = "mistral",
                 model_name: Optional[str] = None,
                 embeddings: Optional[Any] = None):
        """
        Initialise le vector store.
        
        Args:
            embedding_provider: "mistral" ou "huggingface"
            model_name: Nom du modèle HuggingFace (si provider="huggingface")
            embeddings: Embeddings pré-configurés (pour tests/injection)
        """
        
        self.embedding_provider = embedding_provider
        self.vector_store = None
        
        # Injection d'embeddings (pour tests)
        if embeddings is not None:
            self.embeddings = embeddings
            print("🧪 Embeddings injectés (tests/offline).")
            return
        
        # Configuration Mistral AI
        if embedding_provider == "mistral":
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError(
                    "MISTRAL_API_KEY non trouvée dans .env\n"
                    "Ajoutez: MISTRAL_API_KEY=votre_clé"
                )
            
            print("🔑 Utilisation de Mistral AI Embeddings (mistral-embed)")
            self.embeddings = MistralAIEmbeddings(
                model="mistral-embed",
                api_key=api_key
            )
        
        # Configuration HuggingFace (local)
        else:
            model = model_name or "paraphrase-multilingual-MiniLM-L12-v2"
            print(f"🤗 Utilisation de HuggingFace : {model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def create_index(self, chunks: List[Dict]):
        """
        Crée l'index FAISS à partir des chunks d'événements.
        
        Args:
            chunks: Liste de dictionnaires avec 'text' et métadonnées
                    Exemple: [
                        {
                            'text': 'Concert de jazz à Paris...',
                            'event_id': 'event-123',
                            'title': 'Concert de Jazz',
                            'date': '2025-12-15'
                        }
                    ]
        """
        if not chunks:
            raise ValueError("La liste de chunks est vide")
        
        print(f"\n🔨 Création de l'index FAISS...")
        print(f"  Nombre de chunks : {len(chunks)}")
        
        # Validation : tous les chunks doivent avoir 'text'
        for i, chunk in enumerate(chunks):
            if 'text' not in chunk:
                raise ValueError(f"Chunk {i} n'a pas de clé 'text'")
            if not chunk['text'] or not chunk['text'].strip():
                print(f"⚠️  Chunk {i} a un texte vide, ignoré")
        
        # Filtrer les chunks vides
        valid_chunks = [c for c in chunks if c.get('text', '').strip()]
        
        if not valid_chunks:
            raise ValueError("Aucun chunk valide après filtrage")
        
        # Extraire textes et métadonnées
        texts = [chunk['text'] for chunk in valid_chunks]
        metadatas = [
            {k: v for k, v in chunk.items() if k != 'text'}
            for chunk in valid_chunks
        ]
        
        # Créer le vector store
        print(f"  Vectorisation en cours...")
        try:
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            print(f"✓ Index créé avec {len(texts)} vecteurs")
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la création de l'index: {e}")
    
    def search(self, 
               query: str, 
               k: int = 5,
               filter_dict: Optional[Dict] = None,
               score_threshold: Optional[float] = None) -> List[Tuple[Dict, float]]:
        """
        Recherche sémantique dans l'index.
        
        Args:
            query: Question ou requête en langage naturel
            k: Nombre de résultats à retourner
            filter_dict: Filtres sur les métadonnées (ex: {'location_city': 'Paris'})
            score_threshold: Seuil de score minimum (distance L2, plus bas = meilleur)
        
        Returns:
            Liste de tuples (résultat, score)
        """
        if not self.vector_store:
            raise ValueError("Index non créé. Appelez create_index() d'abord.")
        
        if not query or not query.strip():
            raise ValueError("La requête ne peut pas être vide")
        
        # Recherche avec score
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la recherche: {e}")
        
        # Formater les résultats
        formatted_results = []
        for doc, score in results:
            # Appliquer le seuil de score si spécifié
            if score_threshold is not None and score > score_threshold:
                continue
            
            result = {
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            }
            formatted_results.append((result, score))
        
        return formatted_results
    
    def add_events(self, chunks: List[Dict]):
        """
        Ajoute de nouveaux événements à un index existant.
        
        Args:
            chunks: Liste de nouveaux chunks à ajouter
        """
        if not self.vector_store:
            raise ValueError("Index non créé. Appelez create_index() d'abord.")
        
        if not chunks:
            return
        
        valid_chunks = [c for c in chunks if c.get('text', '').strip()]
        if not valid_chunks:
            return
        
        texts = [chunk['text'] for chunk in valid_chunks]
        metadatas = [
            {k: v for k, v in chunk.items() if k != 'text'}
            for chunk in valid_chunks
        ]
        
        print(f"➕ Ajout de {len(texts)} nouveaux vecteurs...")
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        print(f"✓ {len(texts)} vecteurs ajoutés")
    
    def save_index(self, path: str = "data/processed/faiss_index"):
        """
        Sauvegarde l'index FAISS sur le disque.
        
        Args:
            path: Chemin du dossier de sauvegarde
        """
        if not self.vector_store:
            raise ValueError("Aucun index à sauvegarder")
        
        # Créer le dossier si nécessaire
        os.makedirs(path, exist_ok=True)
        
        # Sauvegarder l'index
        self.vector_store.save_local(path)
        
        # Sauvegarder la config
        config = {
            'embedding_provider': self.embedding_provider,
            'num_vectors': len(self.vector_store.docstore._dict)
        }
        
        config_path = f"{path}/config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Index sauvegardé : {path}")
        print(f"  - {config['num_vectors']} vecteurs")
        print(f"  - Provider: {self.embedding_provider}")
    
    def load_index(self, path: str = "data/processed/faiss_index"):
        """
        Charge un index FAISS depuis le disque.
        
        Args:
            path: Chemin du dossier contenant l'index
        """
        # Vérifier que le dossier existe
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le dossier {path} n'existe pas")
        
        # Charger la config
        config_path = f"{path}/config.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Fichier de config introuvable: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Charger l'index
        try:
            self.vector_store = FAISS.load_local(
                path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            print(f"✓ Index chargé : {config['num_vectors']} vecteurs")
            print(f"  Provider: {config['embedding_provider']}")
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement de l'index: {e}")
    
    def get_stats(self) -> Dict:
        """
        Retourne des statistiques sur l'index.
        
        Returns:
            Dictionnaire avec les stats
        """
        if not self.vector_store:
            return {
                'status': 'empty',
                'num_vectors': 0
            }
        
        return {
            'status': 'loaded',
            'num_vectors': len(self.vector_store.docstore._dict),
            'embedding_provider': self.embedding_provider
        }


if __name__ == "__main__":
    # Exemple d'utilisation
    import pandas as pd
    
    # 1. Charger les événements nettoyés
    df = pd.read_json('data/processed/events_clean.json')
    
    # 2. Créer des chunks pour FAISS
    chunks = []
    for _, event in df.iterrows():
        chunk = {
            'text': f"{event['title']}. {event['description']}",
            'event_id': event['id'],
            'title': event['title'],
            'location_city': event['location_city'],
            'date_start': event['date_start'],
            'category': event['category']
        }
        chunks.append(chunk)
    
    # 3. Créer l'index FAISS
    vector_store = FAISSVectorStore(embedding_provider="mistral")
    vector_store.create_index(chunks)
    
    # 4. Recherche
    results = vector_store.search("concert de jazz à Paris", k=5)
    
    for result, score in results:
        print(f"\n📍 Score: {score:.4f}")
        print(f"Titre: {result['metadata']['title']}")
        print(f"Ville: {result['metadata']['location_city']}")
    
    # 5. Sauvegarder
    vector_store.save_index()