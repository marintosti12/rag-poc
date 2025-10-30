#!/usr/bin/env python3

import sys
import os

from src.rag.rag_system import RAGSystem
from src.vector.langchain_faiss import FAISSVectorStore

# Rich pour l'interface
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich import box
from rich.text import Text

# Prompt Toolkit pour l'input avancé
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter

from datetime import datetime
import json


class RichChatbot:
    """Chatbot avec interface Rich terminal"""
    
    def __init__(self, vector_store):
        """Initialise le chatbot"""
        self.rag = RAGSystem(vector_store)
        self.console = Console()
        self.history = []
        
        # Historique des commandes
        self.session = PromptSession(
            history=FileHistory('.chat_history'),
            auto_suggest=AutoSuggestFromHistory(),
        )
        
        # Exemples de questions pour l'autocomplétion
        self.question_completer = WordCompleter([
            'concert de jazz',
            'exposition d\'art',
            'événement gratuit',
            'spectacle enfants',
            'festival',
            'quit',
            'help',
            'history',
            'clear'
        ], ignore_case=True)
    
    def show_welcome(self):
        """Affiche l'écran d'accueil"""
        self.console.clear()
        
        welcome_text = """
# 🎭 Chatbot Puls-Events

Bienvenue dans votre assistant intelligent pour les événements culturels !

## 💡 Exemples de questions
- "Je cherche un concert de jazz à Paris"
- "Quels événements gratuits pour enfants ?"
- "Exposition d'art contemporain ce week-end"

## 🎯 Commandes disponibles
- **quit** ou **exit** : Quitter
- **help** : Afficher l'aide
- **history** : Voir l'historique
- **clear** : Effacer l'écran
- **stats** : Statistiques de la session

Tapez votre question pour commencer !
"""
        
        self.console.print(Panel(
            Markdown(welcome_text),
            title="[bold cyan]Assistant Culturel[/bold cyan]",
            border_style="cyan",
            box=box.DOUBLE
        ))
    
    def show_help(self):
        """Affiche l'aide détaillée"""
        help_text = """
## 📖 Guide d'utilisation

### Questions naturelles
Posez vos questions en langage naturel :
- "Concert de jazz ce soir"
- "Événements gratuits à Paris"
- "Spectacle de danse moderne"

### Commandes
- `quit`, `exit`, `q` : Quitter le chatbot
- `help`, `?` : Afficher cette aide
- `history` : Voir vos questions précédentes
- `clear`, `cls` : Effacer l'écran
- `stats` : Statistiques de la session

### Navigation
- ↑ ↓ : Naviguer dans l'historique des questions
- Tab : Autocomplétion
"""
        
        self.console.print(Panel(
            Markdown(help_text),
            title="[bold yellow]Aide[/bold yellow]",
            border_style="yellow"
        ))
    
    def show_history(self):
        """Affiche l'historique avec un tableau"""
        if not self.history:
            self.console.print("[yellow]📝 Aucun historique pour le moment.[/yellow]")
            return
        
        table = Table(
            title="📜 Historique de conversation",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("#", style="cyan", width=4)
        table.add_column("Heure", style="green", width=10)
        table.add_column("Question", style="white", width=40)
        table.add_column("Sources", style="yellow", width=10)
        
        for i, entry in enumerate(self.history[-10:], 1):  # 10 dernières
            time = entry['timestamp'].split('T')[1].split('.')[0]
            question = entry['user_input'][:40]
            sources = str(entry['num_sources'])
            
            table.add_row(str(i), time, question, sources)
        
        self.console.print(table)
    
    def show_stats(self):
        """Affiche les statistiques de la session"""
        if not self.history:
            self.console.print("[yellow]📊 Aucune statistique disponible.[/yellow]")
            return
        
        total_questions = len(self.history)
        total_sources = sum(h['num_sources'] for h in self.history)
        avg_sources = total_sources / total_questions if total_questions > 0 else 0
        
        stats_text = f"""
## 📊 Statistiques de la session

- **Questions posées** : {total_questions}
- **Sources consultées** : {total_sources}
- **Moyenne sources/question** : {avg_sources:.1f}
- **Durée de la session** : Depuis {self.history[0]['timestamp'].split('T')[1][:8]}
"""
        
        self.console.print(Panel(
            Markdown(stats_text),
            title="[bold green]Statistiques[/bold green]",
            border_style="green"
        ))
    
    def display_thinking(self):
        """Affiche une animation de réflexion"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]🤔 Réflexion en cours...", total=None)
            return progress, task
    
    def display_result(self, result):
        """Affiche la réponse de manière élégante"""
        # Réponse principale
        self.console.print("\n")
        self.console.print(Panel(
            Markdown(result['answer']),
            title="[bold green]🤖 Assistant[/bold green]",
            border_style="green",
            box=box.ROUNDED
        ))
        
        # Sources
        if result['sources']:
            self.console.print("\n")
            
            sources_table = Table(
                title=f"📚 Sources ({len(result['sources'])})",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold cyan"
            )
            
            sources_table.add_column("#", style="cyan", width=3)
            sources_table.add_column("Titre", style="white", width=40)
            sources_table.add_column("Lieu", style="yellow", width=15)
            sources_table.add_column("Date", style="green", width=12)
            sources_table.add_column("Score", style="magenta", width=8)
            
            for i, source in enumerate(result['sources'], 1):
                self.console.print(source)
                sources_table.add_row(
                    str(i),
                    source['title'][:40] if source['title'] else "Sans titre",
                    source['city'] if source['city'] else "N/A",
                    source['date'][:10] if source['date'] else "N/A",
                    f"{source['score']:.3f}"
                )
                            
            self.console.print(sources_table)
    
    def handle_command(self, user_input: str) -> bool:
        """
        Gère les commandes spéciales
        Returns: True si c'était une commande, False sinon
        """
        cmd = user_input.lower().strip()
        
        if cmd in ['quit', 'exit', 'q']:
            return 'quit'
        
        if cmd in ['help', '?']:
            self.show_help()
            return True
        
        if cmd == 'history':
            self.show_history()
            return True
        
        if cmd in ['clear', 'cls']:
            self.console.clear()
            self.show_welcome()
            return True
        
        if cmd == 'stats':
            self.show_stats()
            return True
        
        return False
    
    def chat(self, user_input: str):
        """Traite une question utilisateur"""
        # Animation de réflexion
        with self.console.status("[cyan]🔍 Recherche des événements...", spinner="dots"):
            result = self.rag.query(user_input, k=5)
        
        # Sauvegarder dans l'historique
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'response': result['answer'],
            'num_sources': result['num_sources']
        })
        
        return result
    
    def run(self):
        """Lance le chatbot"""
        self.show_welcome()
        
        try:
            while True:
                try:
                    # Input avec prompt toolkit
                    self.console.print()
                    user_input = self.session.prompt(
                        "👤 Vous > ",
                        completer=self.question_completer
                    ).strip()
                    
                    if not user_input:
                        continue
                    
                    # Commandes spéciales
                    cmd_result = self.handle_command(user_input)
                    if cmd_result == 'quit':
                        break
                    elif cmd_result:
                        continue
                    
                    # Traiter la question
                    result = self.chat(user_input)
                    self.display_result(result)
                    
                except KeyboardInterrupt:
                    continue
                
        except (KeyboardInterrupt, EOFError):
            pass
        
        # Message de sortie
        self.console.print("\n")
        self.console.print(Panel(
            "[bold cyan]👋 Merci d'avoir utilisé Puls-Events ![/bold cyan]\n"
            "À bientôt pour de nouvelles découvertes culturelles ! ✨",
            border_style="cyan"
        ))
        
        # Sauvegarder l'historique
        if self.history:
            save = Prompt.ask(
                "\n💾 Sauvegarder l'historique ?",
                choices=["o", "n"],
                default="n"
            )
            
            if save == "o":
                self.save_history()
    
    def save_history(self, filepath: str = "data/chat_history.json"):
        """Sauvegarde l'historique"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        
        self.console.print(f"[green]✓ Historique sauvegardé : {filepath}[/green]")


def main():
    """Fonction principale"""
    console = Console()
    
    # Vérifications
    INDEX_PATH = "data/processed/faiss_index"
    
    if not os.path.exists(INDEX_PATH):
        console.print(Panel(
            "[bold red]❌ Index FAISS non trouvé ![/bold red]\n\n"
            f"Chemin attendu : {INDEX_PATH}\n\n"
            "💡 Exécutez d'abord :\n"
            "   [cyan]poetry run python scripts/step3_build_vector_database.py[/cyan]",
            border_style="red"
        ))
        return 1
    
    # Chargement avec animation
    with console.status("[cyan]🔄 Chargement du système...", spinner="dots"):
        try:
            vector_store = FAISSVectorStore(embedding_provider="huggingface")
            vector_store.load_index(INDEX_PATH)
        except Exception as e:
            console.print(f"[bold red]❌ Erreur de chargement : {e}[/bold red]")
            return 1
    
    console.print("[green]✓ Système chargé avec succès ![/green]")
    
    # Lancer le chatbot
    chatbot = RichChatbot(vector_store)
    chatbot.run()
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        console = Console()
        console.print(f"\n[bold red]❌ Erreur fatale : {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)