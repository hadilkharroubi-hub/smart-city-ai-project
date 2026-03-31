"""
Module des routes de l'API

Ce module contient toutes les routes HTTP de l'API FastAPI.

Routers:
    - analysis.router: Routes d'analyse d'images
    
Usage:
    from app.routes import router
    app.include_router(router)
"""

from .analysis import router

__all__ = ["router"]

# Métadonnées des routes
ROUTES_METADATA = {
    "analysis": {
        "prefix": "/api/v1",
        "tags": ["Analysis"],
        "description": "Routes pour l'analyse d'images de bâtiments",
    },
}

def get_route_info(route_name: str) -> dict:
    return ROUTES_METADATA.get(route_name, {})
