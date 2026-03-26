"""
Schémas Pydantic pour la validation
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from enum import Enum

class ModelType(str, Enum):
    """Types de modèles disponibles"""
    LOCAL = "local"
    GEMINI = "gemini"
    GROQ = "groq"
    OPENROUTER = "openrouter"

class AnalysisResponse(BaseModel):
    """Réponse d'analyse"""
    model: str = Field(..., description="Modèle utilisé pour l'analyse")
    filename: str = Field(..., description="Nom du fichier analysé")
    report: str = Field(..., description="Rapport d'analyse généré")
    local_prediction: Optional[Dict] = Field(None, description="Prédiction du modèle local")
    
class LocalPredictionResponse(BaseModel):
    """Réponse de prédiction du modèle local"""
    predicted_class: str
    class_index: int
    confidence: float
    confidence_percentage: str
    all_probabilities: Dict[str, float]
    severity_level: str
    recommendation: str

class ErrorResponse(BaseModel):
    """Réponse d'erreur"""
    detail: str
    error_code: Optional[str] = None