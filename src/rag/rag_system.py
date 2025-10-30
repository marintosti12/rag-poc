from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    """Système RAG combinant FAISS et Mistral"""
    
    def __init__(self, 
                 vector_store,
                 model_name: str = "mistral-large-latest",
                 temperature: float = 0.7):
        """
        Initialise le système RAG
        
        Args:
            vector_store: Instance de FAISSVectorStore
            model_name: Modèle Mistral à utiliser
            temperature: Température pour la génération
        """
        self.vector_store = vector_store
        
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY non trouvée dans .env")
        
        # Initialiser le modèle Mistral
        self.llm = ChatMistralAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
        
        # Template de prompt
        self.prompt_template = self._create_prompt_template()
        
        print(f"✓ Système RAG initialisé")
        print(f"  Modèle : {model_name}")
        print(f"  Température : {temperature}")
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Crée le template de prompt pour le RAG"""
        
        template = """Tu es un assistant intelligent spécialisé dans les recommandations d'événements culturels.

Contexte : Voici les événements pertinents trouvés dans la base de données :

{context}

Question de l'utilisateur : {question}

Instructions :
- Base ta réponse UNIQUEMENT sur les événements fournis dans le contexte
- Recommande les événements les plus pertinents
- Inclus les informations pratiques : titre, lieu, date, description
- Si aucun événement ne correspond, dis-le poliment
- Sois précis et utile
- Formate ta réponse de manière claire et structurée

Réponse :"""

        return ChatPromptTemplate.from_template(template)
    
    def _format_documents(self, results: List[tuple]) -> str:
        """
        Formate les documents récupérés pour le contexte
        
        Args:
            results: Liste de tuples (document, score)
            
        Returns:
            Contexte formaté
        """
        context_parts = []
        
        for i, (doc, score) in enumerate(results, 1):
            metadata = doc['metadata']
            
            event_info = f"""
Événement {i} (pertinence: {score:.2f}):
- Titre: {metadata.get('event_title', 'N/A')}
- Lieu: {metadata.get('location_city', 'N/A')} - {metadata.get('location_name', 'N/A')}
- Date: {metadata.get('date_start', 'N/A')}
- Catégorie: {metadata.get('category', 'N/A')}
- Description: {doc['text']}
- URL: {metadata.get('url', 'N/A')}
"""
            context_parts.append(event_info.strip())
        
        return "\n\n---\n".join(context_parts)
    
    def query(self, 
              question: str, 
              k: int = 5,
              min_score: float = 0.0) -> Dict:
        """
        Effectue une requête RAG complète
        
        Args:
            question: Question de l'utilisateur
            k: Nombre de documents à récupérer
            min_score: Score minimum de pertinence
            
        Returns:
            Dictionnaire contenant la réponse et les métadonnées
        """
        # 1. Récupération (Retrieval)
        print(f"\n🔍 Recherche pour : '{question}'")
        results = self.vector_store.search(question, k=k)
        
        # Filtrer par score
        results = [(doc, score) for doc, score in results if score >= min_score]
        
        if not results:
            return {
                'question': question,
                'answer': "Je n'ai trouvé aucun événement correspondant à votre recherche.",
                'sources': [],
                'context': ""
            }
        
        print(f"✓ {len(results)} événements pertinents trouvés")
        
        # 2. Formatage du contexte
        context = self._format_documents(results)
        
        # 3. Génération (Augmented Generation)
        print(f"🤖 Génération de la réponse...")
        
        # Construire le prompt
        messages = self.prompt_template.format_messages(
            context=context,
            question=question
        )
        
        # Générer la réponse
        response = self.llm.invoke(messages)
        answer = response.content
        
        print(f"✓ Réponse générée ({len(answer)} caractères)")
        
        # 4. Préparer le résultat
        sources = [
            {
                'title': doc['metadata'].get('event_title', 'N/A'),
                'city': doc['metadata'].get('location_city', 'N/A'),
                'date': doc['metadata'].get('date_start', 'N/A'),
                'url': doc['metadata'].get('url', 'N/A'),
                'score': float(score)
            }
            for doc, score in results
        ]
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'context': context,
            'num_sources': len(results)
        }
    
    def batch_query(self, questions: List[str], k: int = 5) -> List[Dict]:
        """
        Traite plusieurs questions
        
        Args:
            questions: Liste de questions
            k: Nombre de documents par requête
            
        Returns:
            Liste de résultats
        """
        results = []
        for question in questions:
            result = self.query(question, k=k)
            results.append(result)
        
        return results


if __name__ == "__main__":
    from src.vector.langchain_faiss import FAISSVectorStore
    
    print("="*70)
    print("TEST DU SYSTÈME RAG")
    print("="*70)
    
    # Charger le vector store
    print("\n📂 Chargement de l'index FAISS...")
    vector_store = FAISSVectorStore(embedding_provider="huggingface")
    vector_store.load_index()
    
    # Initialiser le système RAG
    print("\n🤖 Initialisation du système RAG...")
    rag = RAGSystem(vector_store)
    
    # Tests
    test_questions = [
        "Je cherche un concert de jazz à Paris",
        "Quels sont les événements gratuits pour enfants ?",
        "Y a-t-il des expositions d'art contemporain ?",
    ]
    
    for question in test_questions:
        print("\n" + "="*70)
        result = rag.query(question, k=3)
        
        print(f"\n❓ Question : {result['question']}")
        print(f"\n💬 Réponse :")
        print(result['answer'])
        
        print(f"\n📚 Sources ({result['num_sources']}) :")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['title'][:60]} (score: {source['score']:.3f})")
    
    print("\n✅ Tests terminés !")