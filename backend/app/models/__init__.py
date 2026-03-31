"""
Module des modèles d'IA

Ce module contient tous les modèles d'intelligence artificielle
utilisés pour l'analyse des bâtiments.

Classes:
    - LocalBuildingDetector: Modèle MobileNetV2 local
    - AIAnalyzer: Analyseurs externes (Gemini, Groq, OpenRouter)

Instances globales:
    - local_detector: Instance du détecteur local
    - ai_analyzer: Instance de l'analyseur IA
"""

from .local_model import LocalBuildingDetector, local_detector
from .ai_analyzer import AIAnalyzer, ai_analyzer

__all__ = [
    "LocalBuildingDetector",
    "local_detector",
    "AIAnalyzer",
    "ai_analyzer",
]

# Informations sur les modèles disponibles
AVAILABLE_MODELS = {
    "local": {
        "name": "MobileNetV2 Local",
        "type": "local",
        "requires_api_key": False,
        "speed": "very_fast",
        "cost": "free",
        "description": "Modèle léger entraîné localement pour une détection rapide",
    },
    "gemini": {
        "name": "Google Gemini",
        "type": "cloud",
        "requires_api_key": True,
        "speed": "fast",
        "cost": "free_tier",
        "description": "Modèle multimodal de Google pour analyse détaillée",
    },
    "groq": {
        "name": "Groq Llama Vision",
        "type": "cloud",
        "requires_api_key": True,
        "speed": "very_fast",
        "cost": "free_tier",
        "description": "Llama 3.2 Vision sur infrastructure Groq ultra-rapide",
    },
    "openrouter": {
        "name": "OpenRouter",
        "type": "cloud",
        "requires_api_key": True,
        "speed": "variable",
        "cost": "free_and_paid",
        "description": "Accès à plusieurs modèles via OpenRouter",
    },
}

def get_model_info(model_name: str) -> dict:
    return AVAILABLE_MODELS.get(model_name, {})

def list_available_models() -> list:
    return list(AVAILABLE_MODELS.keys()) 
