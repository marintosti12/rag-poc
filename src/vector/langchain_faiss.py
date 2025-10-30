from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from typing import List, Dict, Tuple, Optional
import json
import os
from dotenv import load_dotenv

load_dotenv()

class FAISSVectorStore:    
    def __init__(self, 
                 embedding_provider: str = "mistral",
                 model_name: Optional[str] = None):
        
        self.embedding_provider = embedding_provider
        
        if embedding_provider == "mistral":
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY non trouvée dans .env")
            
            print("🔑 Utilisation de Mistral AI Embeddings")
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
        
        self.vector_store = None
    
    def create_index(self, chunks: List[Dict]):
        print(f"\n🔨 Création de l'index FAISS...")
        print(f"  Nombre de chunks : {len(chunks)}")
        
        # Extraire textes et métadonnées
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [
            {k: v for k, v in chunk.items() if k != 'text'}
            for chunk in chunks
        ]
        
        # Créer le vector store
        print(f"  Vectorisation en cours...")
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        print(f"✓ Index créé avec {len(texts)} vecteurs")
    
    def search(self, 
               query: str, 
               k: int = 5,
               filter_dict: Optional[Dict] = None) -> List[Tuple[Dict, float]]:
        
        if not self.vector_store:
            raise ValueError("Index non créé. Appelez create_index() d'abord.")
        
        # Recherche avec score
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        # Formater les résultats
        formatted_results = []
        for doc, score in results:
            result = {
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            }
            formatted_results.append((result, score))
        
        return formatted_results
    
    def save_index(self, path: str = "data/processed/faiss_index"):
        if not self.vector_store:
            raise ValueError("Aucun index à sauvegarder")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.vector_store.save_local(path)
        
        config = {
            'embedding_provider': self.embedding_provider,
            'num_vectors': len(self.vector_store.docstore._dict)
        }
        
        config_path = f"{path}/config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Index sauvegardé : {path}")
    
    def load_index(self, path: str = "data/processed/faiss_index"):
        # Charger la config
        config_path = f"{path}/config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Charger l'index
        self.vector_store = FAISS.load_local(
            path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        print(f"✓ Index chargé : {config['num_vectors']} vecteurs")
    