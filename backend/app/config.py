"""
Configuration de l'application
"""
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """Paramètres de configuration de l'application"""
    
    # Application
    APP_NAME: str = "Smart City Building Analyzer"
    DEBUG: bool = True
    API_VERSION: str = "1.0.0"
    
    # API Keys
    GEMINI_API_KEY: str
    GROQ_API_KEY: str
    OPENROUTER_API_KEY: str
    
    # Blockchain
    WEB3_PROVIDER: str = "http://127.0.0.1:8545"
    CONTRACT_ADDRESS: str = ""
    BLOCKCHAIN_PRIVATE_KEY: str = ""
    
    # Modèles locaux
    LOCAL_MODEL_PATH: str = "../ml-models/saved_models/mobilenet_building_detector.h5"
    TFLITE_MODEL_PATH: str = "../ml-models/saved_models/mobilenet_building_detector.tflite"
    
    # Images
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp", "gif"]
    IMG_SIZE: int = 224
    
    # Catégories de détection
    CATEGORIES: List[str] = [
        "good_condition",
        "minor_damage", 
        "moderate_damage",
        "severe_damage"
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Instance globale
settings = Settings()