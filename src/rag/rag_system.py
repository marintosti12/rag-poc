import re
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:    
    def __init__(self, 
                 vector_store,
                 model_name: str = "mistral-medium-2508",
                 temperature: float = 0.0):
        
        self.vector_store = vector_store
        
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY non trouvée dans .env")
        
        self.llm = ChatMistralAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
        
        self.prompt_template = self._create_prompt_template()
        
        print(f"✓ Système RAG initialisé")
        print(f"  Modèle : {model_name}")
        print(f"  Température : {temperature}")
    

    def _create_prompt_template(self) -> ChatPromptTemplate:
        from datetime import datetime
        today_iso = datetime.now().astimezone().isoformat(timespec='seconds')

        template = f"""Tu es un assistant d'événements. 
    DATE ACTUELLE (ISO, Europe/Paris): {today_iso}

    Tu reçois :
    - CONTEXTE = liste d’événements candidats (texte, avec lignes "Titre:", "Date:", "Ville:", "Lieu:", "URL:", "Description:")
    - QUESTION = la demande utilisateur

    OBJECTIF
    - Ne réponds qu’avec des événements du CONTEXTE.
    - Si la QUESTION contient des indices de futur (« bientôt », « à venir », « futurs », « prochain(s) », « upcoming », « next », « this week/end »), n’affiche aucun événement strictement antérieur à la DATE ACTUELLE.
    - Si une date/période explicite est dans la QUESTION, respecte-la. Sinon, privilégie les événements à venir.
    - Respecte un éventuel SUJET/GENRE (jazz, théâtre, expo, etc.).

    DÉTECTION D’ANNÉE (si la QUESTION contient une année AAAA)
    1) Sélectionne en priorité les événements dont la métadonnée `year` == AAAA.
    2) Sinon, utilise l’année de `date_start`.
    3) Sinon, si l’année AAAA apparaît dans le texte du CONTEXTE (ex. "DateISO:" ou "Date:"), considère ces événements comme correspondants.
    4) S’il y a plusieurs correspondances, garde les plus pertinentes (et les plus proches en date).

    ⚠️ IMPORTANT
    - N’écris **jamais** la phrase d’échec si le CONTEXTE contient au moins 1 événement : dans ce cas, rends **toujours** un tableau Markdown avec les meilleurs candidats (même si la correspondance n’est pas parfaite).
    - N’écris la phrase **exacte** "Je n'ai trouvé aucun événement correspondant à votre recherche." **que si** après application des règles ci-dessus, tu n’as **strictement aucun** événement à afficher.

    RÈGLES DE TRI
    - Tri par date croissante (la plus proche d’abord).
    - À date égale, privilégie le meilleur match sémantique au sujet demandé.

    RÈGLES DE SORTIE (TOUJOURS)
    - Sortie **UNIQUEMENT en Markdown** (pas d’intro, pas de conclusion).
    - **Toujours** afficher un **tableau Markdown** si tu as ≥ 1 événement :
    | Date | Titre | Ville | Lieu | Lien |
    - Remplissage :
    - **Date** : convertir ISO → **JJ/MM/AAAA HH:MM** (24h). Si l’heure manque, mettre **JJ/MM/AAAA**.
    - **Titre** : titre exact.
    - **Ville** et **Lieu** : tels que dans le CONTEXTE (laisser vide si manquants).
    - **Lien** : **[Lien](URL)** ; vide si URL manquante.
    - Interdits : “---”, blocs de code, texte hors format, colonnes supplémentaires.
    - N’invente rien.

    CONVERSION DE LA DATE
    - Le CONTEXTE peut contenir des dates ISO (ex: 2026-03-29T10:00:00+02:00). Convertis-les en JJ/MM/AAAA HH:MM (24h).
    - Si l’heure manque, affiche seulement JJ/MM/AAAA.

    === CONTEXTE (événements candidats) ===
    {{context}}

    === QUESTION UTILISATEUR ===
    {{question}}
    """
        return ChatPromptTemplate.from_template(template)



    
    def _format_documents(self, results: List[tuple]) -> str:        
        context_parts = []
        
        months_fr = {
            1: 'janvier', 2: 'février', 3: 'mars', 4: 'avril',
            5: 'mai', 6: 'juin', 7: 'juillet', 8: 'août',
            9: 'septembre', 10: 'octobre', 11: 'novembre', 12: 'décembre'
        }
        
        for i, (doc, score) in enumerate(results, 1):
            metadata = doc['metadata']
            
            date_str = metadata.get('date_start', 'N/A')
            date_readable = date_str
            
            if date_str != 'N/A':
                try:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    date_readable = f"{dt.day} {months_fr[dt.month]} {dt.year} ({date_str})"
                except:
                    pass
            
            event_info = f"""=== ÉVÉNEMENT {i} ===
Titre: {metadata.get('title', 'N/A')}
📅 DATE EXACTE: {date_readable}
Lieu: {metadata.get('location_city', 'N/A')} - {metadata.get('location_name', 'N/A')}
Catégorie: {metadata.get('category', 'N/A')}
Description: {doc['text'][:400]}...
URL: {metadata.get('url', 'N/A')}
Score: {score:.3f}
"""
            context_parts.append(event_info.strip())
        
        return "\n\n".join(context_parts)
    
    def query(self, question: str, k: int = 10, min_score: float = 0.0) -> Dict:
        print(f"\n🔍 Recherche pour : '{question}'")

        # 1) détecter une année explicite
        year_filter = None
        m = re.search(r'\b(20\d{2})\b', question)
        if m:
            y = int(m.group(1))
            if 2000 <= y <= 2099:
                year_filter = y

        # 2) sur-échantillonner si on filtre par année (évite le 0 résultat)
        k_raw = max(k, 10)
        if year_filter is not None:
            k_raw = max(k * 10, 200)  # <- clé : on prend large pour que le post-filtre trouve des 2024

        # IMPORTANT : on n'envoie PAS le filter FAISS ici, on filtre nous-mêmes après
        results = self.vector_store.search(question, k=k_raw, filter_dict=None)

        # 3) filtrage côté Python
        if year_filter is not None:
            results = [(doc, score) for (doc, score) in results
                    if doc["metadata"].get("year") == year_filter]

            # 4) fallback neutre si toujours vide : on relance une requête simple
            if not results:
                neutral_q = f"événement {year_filter}"
                print(f"ℹ️ Fallback neutre: '{neutral_q}'")
                results = self.vector_store.search(neutral_q, k=k_raw, filter_dict=None)
                results = [(doc, score) for (doc, score) in results
                        if doc["metadata"].get("year") == year_filter]

        # 5) seuil de score éventuel
        results = [(doc, score) for (doc, score) in results if score >= min_score]

        if not results:
            return {
                "question": question,
                "answer": "Je n'ai trouvé aucun événement correspondant à votre recherche.",
                "sources": [],
                "context": "",
            }
            
        print(results)

        print(f"✓ {len(results)} événements pertinents trouvés")

        context = self._format_documents(results)
        print("🤖 Génération de la réponse...")
        messages = self.prompt_template.format_messages(context=context, question=question)
        response = self.llm.invoke(messages)
        answer = response.content
        print(f"✓ Réponse générée ({len(answer)} caractères)")

        sources = [
            {
                "title": doc["metadata"].get("title", "N/A"),
                "city": doc["metadata"].get("location_city", "N/A"),
                "date": doc["metadata"].get("date_start", "N/A"),
                "url": doc["metadata"].get("url", "N/A"),
                "score": float(score),
            }
            for doc, score in results
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "context": context,
            "num_sources": len(results),
        }
