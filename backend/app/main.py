"""
Application FastAPI principale
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .routes import analysis

# Créer l'application
app = FastAPI(
    title=settings.APP_NAME,
    description="API d'analyse de bâtiments par IA pour les villes intelligentes",
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure les routes
app.include_router(analysis.router)

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": f"Bienvenue sur {settings.APP_NAME}",
        "version": settings.API_VERSION,
        "documentation": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "analyze": "POST /api/v1/analyze",
            "models": "GET /api/v1/models"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Actions au démarrage"""
    print("=" * 60)
    print(f"🚀 Démarrage de {settings.APP_NAME}")
    print("=" * 60)
    print(f"📊 Version: {settings.API_VERSION}")
    print(f"🔧 Mode Debug: {settings.DEBUG}")
    print(f"🤖 Modèle local: {settings.LOCAL_MODEL_PATH}")
    print("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Actions à l'arrêt"""
    print("👋 Arrêt de l'application")