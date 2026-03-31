"""
Package principal de l'application Smart City Building Analyzer

Ce package contient l'application FastAPI complète pour l'analyse
de bâtiments par IA dans le cadre des villes intelligentes durables.

Modules:
    - config: Configuration de l'application
    - models: Modèles d'IA (local et externes)
    - routes: Routes de l'API
    - schemas: Schémas Pydantic pour validation
    - utils: Fonctions utilitaires
"""

__version__ = "1.0.0"
__author__ = "Projet PFA 2025-2026"
__description__ = "API d'analyse de bâtiments par IA pour les villes intelligentes"
__email__ = "hadil.kharroubi@enis.tn"

from .config import settings

# Expose settings pour import facile
__all__ = ["settings"]

# Métadonnées pour l'API
API_METADATA = {
    "title": "Smart City Building Analyzer API",
    "description": """
    ## API d'analyse de bâtiments par IA
    
    Cette API permet d'analyser l'état de détérioration des bâtiments
    en utilisant différents modèles d'intelligence artificielle.
    
    ### Fonctionnalités principales :
    - Détection automatique de la détérioration des bâtiments
    - Support de plusieurs modèles d'IA (local et cloud)
    - Analyse d'images en temps réel
    - Recommandations de maintenance
    
    ### Modèles disponibles :
    - **Local** : MobileNetV2 entraîné localement
    - **Gemini** : Google Gemini 2.0 Flash
    - **Groq** : Llama 3.2 Vision
    - **OpenRouter** : Accès à plusieurs modèles
    
    ### Documentation :
    - [GitHub](https://github.com/votre-username/smart-city-ai-project)
    - [Guide des API Keys](docs/API_KEYS_GUIDE.md)
    """,
    "version": __version__,
    "contact": {
        "name": "Équipe ADEVA",
        "email": __email__,
    },
    "license_info": {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
}