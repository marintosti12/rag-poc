#!/usr/bin/env python3
import sys
import os

from src.chunking.text_splitter import EventTextSplitter
from src.vector.langchain_faiss import FAISSVectorStore
import json


def main():        
    print("="*70)
    print("VECTORISATION ET INDEXATION FAISS")
    print("="*70)

    # Configuration
    EVENTS_FILE = 'data/processed/events_clean.json'
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    EMBEDDING_PROVIDER = "huggingface"
    INDEX_PATH = "data/processed/faiss_index"
    
    # Information Configuration
    print(f"\n💻 Source : {EVENTS_FILE}")
    print(f"🔢 Taille des chunks : {CHUNK_SIZE} caractères")
    print(f"📩 Provider d'embeddings : {EMBEDDING_PROVIDER}")
    
    # Vérification du fichier d'entrée
    if not os.path.exists(EVENTS_FILE):
        print(f"\n❌ Fichier non trouvé : {EVENTS_FILE}")
        print("\n💡 Exécutez d'abord :")
        print("   poetry run python scripts/step2_collect_and_clean.py")
        return 1
    
    print("📂 CHARGEMENT DES ÉVÉNEMENTS")
    
    with open(EVENTS_FILE, 'r', encoding='utf-8') as f:
        events = json.load(f)
    
    print(f"✓ {len(events)} événements chargés")
    
    if len(events) == 0:
        print("\n❌ Aucun événement à vectoriser")
        return 1
    
    print("DÉCOUPAGE EN CHUNKS")
    splitter = EventTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    print(f"\n🔪 Découpage en cours...")
    chunks = splitter.process_events(events)
    
    print(f"✓ {len(chunks)} chunks créés")
    
    # Sauvegarder un échantillon de chunks
    chunks_sample_file = 'data/processed/chunks_sample.json'
    with open(chunks_sample_file, 'w', encoding='utf-8') as f:
        json.dump(chunks[:10], f, ensure_ascii=False, indent=2)
    print(f"  • Échantillon sauvegardé : {chunks_sample_file}")
    
    # Exemple de chunk
    print(f"\n📝 Exemple de chunk :")
    example_chunk = chunks[0]
    print(f"  Titre événement : {example_chunk['event_title']}")
    print(f"  Texte : {example_chunk['text'][:150]}...")
    
    print("🔢 VECTORISATION ET INDEXATION FAISS")
    
    print(f"\n🤖 Initialisation du modèle d'embeddings ({EMBEDDING_PROVIDER})...")
    vector_store = FAISSVectorStore(
        embedding_provider=EMBEDDING_PROVIDER
    )
    
    print(f"\n📊 Création de l'index FAISS...")
    vector_store.create_index(chunks)
    
    print("💾 SAUVEGARDE DE L'INDEX")
    
    vector_store.save_index(INDEX_PATH)
    print(f"✓ Index sauvegardé dans : {INDEX_PATH}")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)