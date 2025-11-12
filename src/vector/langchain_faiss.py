from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from typing import Any, List, Dict, Tuple, Optional
import json
import os
from dotenv import load_dotenv

load_dotenv()

class FAISSVectorStore:
    def __init__(self, 
                 embedding_provider: str = "mistral",
                 model_name: Optional[str] = None,
                 embeddings: Optional[Any] = None):
        self.embedding_provider = embedding_provider
        self.vector_store = None
        
        if embeddings is not None:
            self.embeddings = embeddings
            return
        
        if embedding_provider == "mistral":
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError(
                    "MISTRAL_API_KEY non trouvée dans .env\n"
                )
            
            print("🔑 Utilisation de Mistral AI Embeddings (mistral-embed)")
            self.embeddings = MistralAIEmbeddings(
                model="mistral-embed",
                api_key=api_key
            )
        
        else:
            model = model_name or "paraphrase-multilingual-MiniLM-L12-v2"
            print(f"🤗 Utilisation de HuggingFace : {model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def create_index(self, chunks: List[Dict]):
        if not chunks:
            raise ValueError("La liste de chunks est vide")
        
        print(f"\n🔨 Création de l'index FAISS...")
        print(f"  Nombre de chunks : {len(chunks)}")
        
        for i, chunk in enumerate(chunks):
            if 'text' not in chunk:
                raise ValueError(f"Chunk {i} n'a pas de clé 'text'")
            if not chunk['text'] or not chunk['text'].strip():
                print(f"Chunk {i} a un texte vide, ignoré")
        
        valid_chunks = [c for c in chunks if c.get('text') and str(c.get('text', '')).strip()]
        
        if not valid_chunks:
            raise ValueError("Aucun chunk valide après filtrage")
        
        texts = [chunk['text'] for chunk in valid_chunks]
        metadatas = [
            {k: v for k, v in chunk.items() if k != 'text'}
            for chunk in valid_chunks
        ]
        
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
    
        
        if not query or not query.strip():
            raise ValueError("La requête ne peut pas être vide")
        
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la recherche: {e}")
        
        formatted_results = []
        for doc, score in results:
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
        if not self.vector_store:
            raise ValueError("Aucun index à sauvegarder")
        
        os.makedirs(path, exist_ok=True)
        
        self.vector_store.save_local(path)
        
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
    
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le dossier {path} n'existe pas")
        
        config_path = f"{path}/config.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Fichier de config introuvable: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
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
    