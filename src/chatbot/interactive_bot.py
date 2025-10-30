from src.rag.rag_system import RAGSystem
from src.vector.langchain_faiss import FAISSVectorStore
import json
from datetime import datetime

class EventChatbot:
    """Chatbot interactif pour recommandations d'événements"""
    
    def __init__(self, vector_store):
        """Initialise le chatbot"""
        self.rag = RAGSystem(vector_store)
        self.history = []
    
    def chat(self, user_input: str, k: int = 5) -> Dict:
        """
        Traite une entrée utilisateur
        
        Args:
            user_input: Message de l'utilisateur
            k: Nombre de résultats
            
        Returns:
            Résultat de la requête
        """
        result = self.rag.query(user_input, k=k)
        
        # Sauvegarder dans l'historique
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'response': result['answer'],
            'num_sources': result['num_sources']
        })
        
        return result
    
    def display_result(self, result: Dict):
        """Affiche le résultat de manière formatée"""
        print("\n" + "="*70)
        print("🤖 ASSISTANT")
        print("="*70)
        print(f"\n{result['answer']}")
        
        if result['sources']:
            print(f"\n📚 Sources ({len(result['sources'])}) :")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n{i}. {source['title']}")
                print(f"   📍 {source['city']}")
                print(f"   📅 {source['date']}")
                if source['url']:
                    print(f"   🔗 {source['url']}")
    
    def run(self):
        """Lance le chatbot en mode interactif"""
        print("="*70)
        print("🎭 CHATBOT D'ÉVÉNEMENTS CULTURELS")
        print("="*70)
        print("\nPosez vos questions sur les événements culturels.")
        print("Tapez 'quit' ou 'exit' pour quitter.\n")
        
        while True:
            try:
                user_input = input("👤 Vous : ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Au revoir !")
                    break
                
                result = self.chat(user_input)
                self.display_result(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir !")
                break
            except Exception as e:
                print(f"\n❌ Erreur : {e}")
    
    def save_history(self, filepath: str = "data/chat_history.json"):
        """Sauvegarde l'historique de conversation"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        print(f"✓ Historique sauvegardé : {filepath}")


if __name__ == "__main__":
    # Charger le vector store
    print("📂 Chargement de la base de données...")
    vector_store = FAISSVectorStore(embedding_provider="huggingface")
    vector_store.load_index()
    
    # Créer et lancer le chatbot
    chatbot = EventChatbot(vector_store)
    chatbot.run()
    
    # Sauvegarder l'historique à la fin
    if chatbot.history:
        chatbot.save_history()