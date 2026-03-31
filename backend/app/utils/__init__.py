"""
Module des utilitaires

Ce module contient toutes les fonctions utilitaires pour le
traitement d'images et autres opérations communes.

Fonctions:
    - get_image_format: Détecte le format d'une image
    - get_mime_type: Convertit le format en MIME type
    - resize_image: Redimensionne une image
    - preprocess_for_model: Prétraite pour le modèle ML
    - encode_image_to_base64: Encode en base64
    
Usage:
    from app.utils import resize_image, preprocess_for_model
"""

from .image_processing import (
    get_image_format,
    get_mime_type,
    resize_image,
    preprocess_for_model,
    encode_image_to_base64,
)

__all__ = [
    "get_image_format",
    "get_mime_type",
    "resize_image",
    "preprocess_for_model",
    "encode_image_to_base64",
]

# Constantes pour le traitement d'images
IMAGE_CONSTANTS = {
    "default_size": 224,  # Taille pour MobileNetV2
    "max_size_mb": 10,    # Taille maximale en MB
    "supported_formats": ["jpg", "jpeg", "png", "gif", "webp"],
    "mime_types": {
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    },
}

def get_supported_formats() -> list:
    return IMAGE_CONSTANTS["supported_formats"]

def is_format_supported(format: str) -> bool:
    return format.lower() in IMAGE_CONSTANTS["supported_formats"]

def get_image_constants() -> dict:
    return IMAGE_CONSTANTS.copy() 
