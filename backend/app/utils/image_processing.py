"""
Utilitaires pour le traitement d'images
"""
import io
import base64
from PIL import Image
from typing import Tuple
import numpy as np

def get_image_format(image_bytes: bytes) -> str:
    """Détecte le format de l'image"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.format.lower() if image.format else "jpeg"
    except:
        return "jpeg"

def get_mime_type(image_format: str) -> str:
    """Convertit le format d'image en mime type"""
    mime_type_map = {
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp"
    }
    return mime_type_map.get(image_format, "image/jpeg")

def resize_image(image_bytes: bytes, max_size_mb: float = 5.0) -> bytes:
    """
    Redimensionne l'image si elle dépasse la taille maximale
    
    Args:
        image_bytes: Bytes de l'image
        max_size_mb: Taille maximale en MB
        
    Returns:
        Bytes de l'image redimensionnée
    """
    image_size_mb = len(image_bytes) / (1024 * 1024)
    
    if image_size_mb <= max_size_mb:
        return image_bytes
    
    print(f"Image trop grande ({image_size_mb:.2f} MB), redimensionnement...")
    
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    img_format = img.format if img.format else "JPEG"
    img.save(buffer, format=img_format, quality=85, optimize=True)
    
    new_bytes = buffer.getvalue()
    new_size_mb = len(new_bytes) / (1024 * 1024)
    print(f"Nouvelle taille : {new_size_mb:.2f} MB")
    
    return new_bytes

def preprocess_for_model(image_bytes: bytes, target_size: int = 224) -> np.ndarray:
    """
    Prétraite l'image pour le modèle MobileNetV2
    
    Args:
        image_bytes: Bytes de l'image
        target_size: Taille cible (224 pour MobileNetV2)
        
    Returns:
        Array numpy normalisé
    """
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convertir en RGB si nécessaire
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Redimensionner
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convertir en array numpy
    img_array = np.array(img)
    
    # Normaliser (0-1)
    img_array = img_array / 255.0
    
    # Ajouter dimension batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode l'image en base64"""
    return base64.b64encode(image_bytes).decode("utf-8")