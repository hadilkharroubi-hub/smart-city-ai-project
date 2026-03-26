# 🏙️ Smart City Building Analyzer

Système d'analyse intelligente de bâtiments pour les villes durables, combinant IA embarquée et blockchain.

## 🚀 Démarrage rapide

### 1. Obtenir les API keys

Suivez le guide `docs/API_KEYS_GUIDE.md` pour obtenir :
- ✅ Gemini API Key (recommandé pour commencer)
- ✅ Groq API Key (optionnel)
- ✅ OpenRouter API Key (optionnel)

### 2. Configuration
```bash
# Cloner le projet
git clone <votre-repo>
cd smart-city-ai-project

# Backend
cd backend
cp .env.example .env
# Éditer .env et ajouter vos API keys

# Installer les dépendances
pip install -r requirements.txt

# ML Models
cd ../ml-models
pip install -r requirements.txt
```

### 3. Préparer les données (IMPORTANT)
```bash
# Créer la structure de dossiers
mkdir -p ml-models/data/building_images/train/{good_condition,minor_damage,moderate_damage,severe_damage}

# Placez au moins 50 images par catégorie
# Vous pouvez utiliser :
# - Vos propres photos
# - Dataset CODEBRIM : https://zenodo.org/record/2620293
# - Google Images (attention aux droits)
```

### 4. Entraîner le modèle local
```bash
cd ml-models/scripts
python train_model.py

# Ensuite, optimiser pour mobile
python optimize_model.py
```

### 5. Lancer l'API
```bash
cd backend
python run.py

# L'API sera accessible sur : http://localhost:8000
# Documentation : http://localhost:8000/docs
```

### 6. Tester l'API
```bash
# Avec curl
curl -X POST "http://localhost:8000/api/v1/analyze?model=gemini" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/building_image.jpg"

# Ou via l'interface Swagger : http://localhost:8000/docs
```

## 📁 Structure du projet
```
smart-city-ai-project/
├── backend/          # API FastAPI
├── ml-models/        # Entraînement des modèles
├── blockchain/       # Smart contracts
├── integration/      # Pont IA-Blockchain
└── docs/            # Documentation
```

## 🧪 Modèles disponibles

| Modèle | Type | Vitesse | Coût | API Key |
|--------|------|---------|------|---------|
| Local (MobileNetV2) | Embarqué | ⚡⚡⚡ | Gratuit | ❌ Non |
| Gemini | Cloud | ⚡⚡ | Gratuit* | ✅ Oui |
| Groq | Cloud | ⚡⚡⚡ | Gratuit* | ✅ Oui |
| OpenRouter | Cloud | ⚡ | Gratuit* | ✅ Oui |

*Quota limité

## 📚 Documentation

- [Guide des API Keys](docs/API_KEYS_GUIDE.md)
- [Guide de setup](docs/SETUP_GUIDE.md)
- [Documentation API](http://localhost:8000/docs)

## 🤝 Contribution

Ce projet est développé dans le cadre du projet ADEVA 2025-2026.