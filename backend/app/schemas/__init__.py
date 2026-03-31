"""
Module des schémas Pydantic

Ce module contient tous les schémas de validation des données
utilisés par l'API.

Classes:
    - ModelType: Énumération des types de modèles
    - AnalysisResponse: Schéma de réponse d'analyse
    - LocalPredictionResponse: Schéma de prédiction locale
    - ErrorResponse: Schéma de réponse d'erreur
    
Usage:
    from app.schemas import AnalysisResponse, ModelType
"""

from .analysis import (
    AnalysisResponse,
    ModelType,
    LocalPredictionResponse,
    ErrorResponse,
)

__all__ = [
    "AnalysisResponse",
    "ModelType",
    "LocalPredictionResponse",
    "ErrorResponse",
]

# Catégories de détection
DETECTION_CATEGORIES = [
    "good_condition",
    "minor_damage",
    "moderate_damage",
    "severe_damage",
]

# Mapping des catégories vers descriptions lisibles
CATEGORY_LABELS = {
    "good_condition": "Bon État",
    "minor_damage": "Dégâts Mineurs",
    "moderate_damage": "Dégradation Modérée",
    "severe_damage": "Dégradation Sévère",
}

# Mapping des niveaux de sévérité
SEVERITY_LEVELS = {
    0: "Aucun",
    1: "Faible",
    2: "Modéré",
    3: "Élevé",
}

def get_category_label(category: str) -> str:
    """
    Convertit une catégorie en label lisible
    
    Args:
        category: Nom de la catégorie
        
    Returns:
        str: Label lisible
    """
    return CATEGORY_LABELS.get(category, category.replace("_", " ").title())

def get_severity_label(level: int) -> str:
    """
    Convertit un niveau de sévérité en label
    
    Args:
        level: Niveau de sévérité (0-3)
        
    Returns:
        str: Label de sévérité
    """
    return SEVERITY_LEVELS.get(level, "Inconnu")
