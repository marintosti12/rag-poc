#!/usr/bin/env python3

import sys
import os

from src.rag.rag_system import RAGSystem
from src.vector.langchain_faiss import FAISSVectorStore


def main():
    print("="*70)
    print("TEST DU SYSTÈME RAG")
    print("="*70)
    
    INDEX_PATH = "data/processed/faiss_index"
    
    # Vérifications
    if not os.path.exists(INDEX_PATH):
        print(f"\n❌ Index FAISS non trouvé : {INDEX_PATH}")
        print("\n💡 Exécutez d'abord :")
        print("   poetry run python scripts/step3_build_vector_database.py")
        return 1
    
    # Chargement
    print("📂 CHARGEMENT DE L'INDEX FAISS")
    
    print("\n📥 Chargement en cours...")
    vector_store = FAISSVectorStore(embedding_provider="huggingface")
    vector_store.load_index(INDEX_PATH)
    
    print("\n🔧 Initialisation de Mistral...")
    rag = RAGSystem(vector_store)
    
    # Tests
    print("🧪 TESTS DU SYSTÈME")
    
    test_questions = [
        "Je cherche un concert de jazz à Paris",
        "Quels sont les événements gratuits pour enfants ?",
        "Y a-t-il des expositions d'art contemporain ?",
        "Spectacle de danse ce week-end",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_questions)}")
        print(f"{'='*70}")
        
        result = rag.query(question, k=3)
        
        print(f"\n❓ Question : {result['question']}")
        print(f"\n💬 Réponse :")
        print(f"{result['answer']}")
        
        print(f"\n📚 Sources ({result['num_sources']}) :")
        for j, source in enumerate(result['sources'], 1):
            print(f"  {j}. {source['title'][:55]}")
            print(f"     📍 {source['city']} | 📅 {source['date']}")
            print(f"     📊 Score : {source['score']:.3f}")
    

    print(f"\n✅ Le système RAG fonctionne correctement !")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)