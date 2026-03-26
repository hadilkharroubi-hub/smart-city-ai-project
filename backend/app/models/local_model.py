"""
Modèle MobileNetV2 local pour la détection
"""
import tensorflow as tf
import numpy as np
from typing import Dict
from ..config import settings
from ..utils.image_processing import preprocess_for_model

class LocalBuildingDetector:
    """Détecteur de détérioration de bâtiments basé sur MobileNetV2"""
    
    def __init__(self):
        self.model = None
        self.categories = settings.CATEGORIES
        self.img_size = settings.IMG_SIZE
        self.load_model()
    
    def load_model(self):
        """Charge le modèle entraîné"""
        try:
            model_path = settings.LOCAL_MODEL_PATH
            print(f"Chargement du modèle depuis : {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Modèle chargé avec succès")
        except Exception as e:
            print(f"⚠️  Erreur lors du chargement du modèle : {e}")
            print("Le modèle local n'est pas disponible. Entraînez-le d'abord.")
            self.model = None
    
    def predict(self, image_bytes: bytes) -> Dict:
        """
        Prédit l'état du bâtiment
        
        Args:
            image_bytes: Bytes de l'image
            
        Returns:
            Dictionnaire avec les prédictions
        """
        if self.model is None:
            return {
                "error": "Modèle local non disponible",
                "message": "Veuillez entraîner le modèle d'abord"
            }
        
        try:
            # Prétraiter l'image
            img_array = preprocess_for_model(image_bytes, self.img_size)
            
            # Prédiction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Construire le résultat
            result = {
                "predicted_class": self.categories[predicted_class_idx],
                "class_index": int(predicted_class_idx),
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.2f}%",
                "all_probabilities": {
                    category: float(prob) 
                    for category, prob in zip(self.categories, predictions[0])
                },
                "severity_level": self._get_severity_level(predicted_class_idx),
                "recommendation": self._get_recommendation(predicted_class_idx, confidence)
            }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Erreur lors de la prédiction"
            }
    
    def _get_severity_level(self, class_idx: int) -> str:
        """Retourne le niveau de sévérité"""
        levels = {
            0: "Aucun",
            1: "Faible",
            2: "Modéré",
            3: "Élevé"
        }
        return levels.get(class_idx, "Inconnu")
    
    def _get_recommendation(self, class_idx: int, confidence: float) -> str:
        """Génère une recommandation"""
        recommendations = {
            0: "Bâtiment en bon état. Maintenance préventive régulière recommandée.",
            1: "Dégâts mineurs détectés. Inspection visuelle conseillée dans les 3 mois.",
            2: "Dégradation modérée. Inspection professionnelle requise sous 1 mois.",
            3: "Dégradation sévère. Intervention urgente nécessaire. Risque de sécurité."
        }
        
        base_recommendation = recommendations.get(class_idx, "Inspection recommandée")
        
        if confidence < 0.7:
            base_recommendation += " (Confiance faible - vérification humaine recommandée)"
        
        return base_recommendation

# Instance globale
local_detector = LocalBuildingDetector()