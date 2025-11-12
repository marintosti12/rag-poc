#!/usr/bin/env python3
import json
import sys
import os

from src.fetching.fetch_events import OpenAgendaFetcher
from src.fetching.clean_events import EventsCleaner
from dotenv import load_dotenv

load_dotenv()


def clean_events(input_file: str, output_json: str, output_csv: str) -> int:
    if not os.path.exists(input_file):
        print(f"\n Fichier non trouvé : {input_file}")
        return 0
    
    # Chargement
    print(f"\n📂 Chargement de {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        events = json.load(f)
    
    print(f"✓ {len(events)} événements bruts chargés")
    
    if len(events) == 0:
        print("\n Aucun événement à nettoyer")
        return 0
    
    print("\n🧹 Nettoyage en cours...")
    cleaner = EventsCleaner()
    df_clean = cleaner.clean_pipeline(events)
    
    if len(df_clean) == 0:
        print("\n Aucun événement valide après nettoyage")
        return 0
    
    # Sauvegarde
    print(f"\n💾 Sauvegarde des données nettoyées...")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    df_clean.to_json(output_json, orient='records', force_ascii=False, indent=2)
    print(f"✓ JSON : {output_json}")
    
    df_clean.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✓ CSV : {output_csv}")
    

    return len(df_clean)

def main():    
    print("="*70)
    print("COLLECTE DES ÉVÉNEMENTS - OPENAGENDA")
    print("="*70)
    
    # Configuration
    API_KEY = os.getenv("OPEN_AGENDA_API_KEY")
    LOCATION = os.getenv("TARGET_CITY")
    DATE_START = "2024-10-30T00:00:00.000Z"
    DATE_END   = "2026-10-30T23:59:59.999Z"

    LIMIT_PER_AGENDA = 60
    LIMIT_AGENDAS = 7
    OUTPUT_FILE = "data/raw/events_raw.json"
    CLEAN_JSON = "data/processed/events_clean.json"
    CLEAN_CSV = "data/processed/events_clean.csv"

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Information Configuration
    print(f"\n🔑 Clé API : {API_KEY[:10]}***")
    print(f"📍 Localisation : {LOCATION}")
    print(f"📅 Période : {DATE_START} → {DATE_END}")
    print(f"🔢 Limite par agenda : {LIMIT_PER_AGENDA}")
    
    # Initialisation du fetcher
    print("\n" + "-"*70)
    print("[1/3] 🔌 Connexion à l'API OpenAgenda...")
    print("-"*70)
    
    fetcher = OpenAgendaFetcher(API_KEY)
    
    # Test de connexion
    if not fetcher.test_connection():
        print("\n Impossible de se connecter à l'API")
        return 1
    
    # Collecte des événements
    print("\n" + "-"*70)
    print("[2/3] 📥 Collecte des événements...")
    print("-"*70)
    
    events = fetcher.fetch_events_from_multiple_agendas(
        location=LOCATION,
        date_start=DATE_START,
        date_end=DATE_END,
        limit=LIMIT_PER_AGENDA,
        agendaLimit=LIMIT_AGENDAS
    )
    

    if not events:
        print("\n ⚠️ Aucun événement récupéré")
        return 1
    
    # Sauvegarde
    print("\n" + "-"*70)
    print("[3/3] 💾 Sauvegarde des données...")
    print("-"*70)
    
    fetcher.save_raw_data(events, OUTPUT_FILE)
    
    num_cleaned = clean_events(
        input_file=OUTPUT_FILE,
        output_json=CLEAN_JSON,
        output_csv=CLEAN_CSV
    )

    # Statistiques
    print("\n" + "="*70)
    print("✅ COLLECTE TERMINÉE AVEC SUCCÈS")
    print("="*70)
    
    agendas_count = len(set(e.get('source_agenda_uid') for e in events))
    
    print(f"\n📊 Statistiques :")
    print(f"  • Événements collectés : {len(events)}")
    print(f"  • Agendas sources : {agendas_count}")
    print(f"  • Fichier créé : {OUTPUT_FILE}")
    
    # Aperçu des premiers événements
    print(f"\n📋 Aperçu des événements :")
    for i, event in enumerate(events[:3], 1):
        title = event.get('title', {})
        if isinstance(title, dict):
            title = title.get('fr', 'Sans titre')
        print(f"\n  {i}. {title}")
        print(f"     Source : {event.get('source_agenda', 'N/A')}")
    
    if len(events) > 3:
        print(f"\n  ... et {len(events) - 3} autres événements")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)