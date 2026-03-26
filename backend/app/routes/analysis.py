"""
Routes pour l'analyse d'images
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from ..schemas.analysis import AnalysisResponse, ModelType, ErrorResponse
from ..models.local_model import local_detector
from ..models.ai_analyzer import ai_analyzer
from typing import Optional

router = APIRouter(prefix="/api/v1", tags=["Analysis"])

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_building(
    file: UploadFile = File(..., description="Image du bâtiment à analyser"),
    model: ModelType = Query(ModelType.LOCAL, description="Modèle à utiliser"),
    use_local: bool = Query(True, description="Inclure la prédiction du modèle local")
):
    """
    Analyse une image de bâtiment pour détecter les dégradations
    
    **Modèles disponibles** :
    - `local` : MobileNetV2 entraîné localement (rapide, offline)
    - `gemini` : Google Gemini (analyse détaillée)
    - `groq` : Groq Llama Vision (rapide)
    - `openrouter` : OpenRouter Gemini (alternative)
    
    **Retour** :
    - Rapport d'analyse textuel
    - Prédiction du modèle local (si activé)
    """
    
    # Vérifier le type de fichier
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Le fichier doit être une image (JPEG, PNG, WEBP, GIF)"
        )
    
    # Lire l'image
    image_bytes = await file.read()
    
    # Prédiction du modèle local (si demandé)
    local_prediction = None
    if use_local:
        local_prediction = local_detector.predict(image_bytes)
        if "error" in local_prediction:
            print(f"⚠️  Modèle local non disponible : {local_prediction['error']}")
    
    # Analyse avec le modèle externe
    if model == ModelType.LOCAL:
        if local_prediction and "error" not in local_prediction:
            report = f"""**ANALYSE AUTOMATIQUE - MODÈLE LOCAL**

**État détecté** : {local_prediction['predicted_class'].replace('_', ' ').title()}
**Niveau de sévérité** : {local_prediction['severity_level']}
**Confiance** : {local_prediction['confidence_percentage']}

**Probabilités par catégorie** :
{chr(10).join(f"- {cat.replace('_', ' ').title()} : {prob*100:.1f}%" for cat, prob in local_prediction['all_probabilities'].items())}

**Recommandation** :
{local_prediction['recommendation']}
"""
        else:
            raise HTTPException(
                status_code=500,
                detail="Le modèle local n'est pas disponible. Entraînez-le d'abord ou utilisez un autre modèle."
            )
    
    elif model == ModelType.GEMINI:
        report = ai_analyzer.analyze_with_gemini(image_bytes)
    
    elif model == ModelType.GROQ:
        report = ai_analyzer.analyze_with_groq(image_bytes)
    
    elif model == ModelType.OPENROUTER:
        report = ai_analyzer.analyze_with_openrouter(image_bytes)
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle '{model}' non supporté"
        )
    
    return AnalysisResponse(
        model=model.value,
        filename=file.filename,
        report=report,
        local_prediction=local_prediction if use_local else None
    )

@router.get("/models")
async def list_available_models():
    """Liste les modèles disponibles"""
    return {
        "models": [
            {
                "id": "local",
                "name": "MobileNetV2 Local",
                "description": "Modèle léger entraîné localement",
                "speed": "Très rapide",
                "cost": "Gratuit",
                "requires_api_key": False
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "description": "Modèle multimodal de Google",
                "speed": "Rapide",
                "cost": "Gratuit (quota limité)",
                "requires_api_key": True
            },
            {
                "id": "groq",
                "name": "Groq Llama Vision",
                "description": "Llama 3.2 Vision sur infrastructure Groq",
                "speed": "Très rapide",
                "cost": "Gratuit (quota limité)",
                "requires_api_key": True
            },
            {
                "id": "openrouter",
                "name": "OpenRouter",
                "description": "Accès à plusieurs modèles via OpenRouter",
                "speed": "Variable",
                "cost": "Certains modèles gratuits",
                "requires_api_key": True
            }
        ]
    }

@router.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "local_model_loaded": local_detector.model is not None,
        "api_version": "1.0.0"
    }