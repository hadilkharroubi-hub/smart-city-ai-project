"""
Analyseurs IA externes (Gemini, Groq, OpenRouter)
"""
import requests
import base64
from openai import OpenAI
from typing import Dict
from ..config import settings
from ..utils.image_processing import (
    get_image_format, 
    get_mime_type, 
    resize_image,
    encode_image_to_base64
)

class AIAnalyzer:
    """Classe pour analyser avec les modèles externes"""
    
    ANALYSIS_PROMPT = """Analyse cette image de bâtiment. Réponds en français, de façon structurée :

1. **État général** : (bon / moyen / dégradé) + description en 1-2 phrases
2. **Zones à risque** : liste les problèmes visibles (fissures, humidité, décollement, corrosion, etc.)
3. **Recommandations** : actions prioritaires de maintenance ou inspection

Sois précis, concis et professionnel. Maximum 15 lignes."""

    @staticmethod
    def analyze_with_gemini(image_bytes: bytes) -> str:
        """Analyse avec Gemini"""
        try:
            # Redimensionner si nécessaire
            image_bytes = resize_image(image_bytes, max_size_mb=4.0)
            
            # Préparer les données
            image_b64 = encode_image_to_base64(image_bytes)
            image_format = get_image_format(image_bytes)
            mime_type = get_mime_type(image_format)
            
            # Appel API Gemini
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
            
            headers = {
                "Content-Type": "application/json",
            }
            
            data = {
                "contents": [{
                    "parts": [
                        {"text": AIAnalyzer.ANALYSIS_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_b64
                            }
                        }
                    ]
                }]
            }
            
            response = requests.post(
                f"{url}?key={settings.GEMINI_API_KEY}",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"Gemini n'a pas pu analyser l'image. Réponse: {result}"
        
        except requests.exceptions.HTTPError as e:
            error_detail = e.response.json() if e.response.text else str(e)
            return f"Erreur HTTP Gemini : {error_detail}"
        except Exception as e:
            return f"Erreur Gemini : {str(e)}"
    
    @staticmethod
    def analyze_with_groq(image_bytes: bytes) -> str:
        """Analyse avec Groq"""
        try:
            image_b64 = encode_image_to_base64(image_bytes)
            image_format = get_image_format(image_bytes)
            
            client = OpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1"
            )
            
            response = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": AIAnalyzer.ANALYSIS_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{image_b64}"
                            }
                        }
                    ]
                }],
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Erreur Groq : {str(e)}"
    
    @staticmethod
    def analyze_with_openrouter(image_bytes: bytes) -> str:
        """Analyse avec OpenRouter"""
        try:
            image_b64 = encode_image_to_base64(image_bytes)
            image_format = get_image_format(image_bytes)
            
            client = OpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )
            
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-exp:free",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": AIAnalyzer.ANALYSIS_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{image_b64}"
                            }
                        }
                    ]
                }],
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Erreur OpenRouter : {str(e)}"

# Instance globale
ai_analyzer = AIAnalyzer()