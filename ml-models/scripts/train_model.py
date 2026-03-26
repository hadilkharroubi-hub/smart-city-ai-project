"""
Script d'entraînement du modèle MobileNetV2
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DATA_DIR = '../data/building_images/train'
SAVE_DIR = '../saved_models'

# Catégories
CATEGORIES = ['good_condition', 'minor_damage', 'moderate_damage', 'severe_damage']

print("=" * 70)
print("ENTRAÎNEMENT DU MODÈLE MOBILENETV2 - DÉTECTION DE DÉTÉRIORATION")
print("=" * 70)

# Créer le dossier de sauvegarde
os.makedirs(SAVE_DIR, exist_ok=True)

def load_dataset(data_dir):
    """Charge le dataset depuis les dossiers"""
    print("\n📂 Chargement du dataset...")
    
    images = []
    labels = []
    
    for idx, category in enumerate(CATEGORIES):
        folder_path = os.path.join(data_dir, category)
        
        if not os.path.exists(folder_path):
            print(f"⚠️  Dossier non trouvé : {folder_path}")
            continue
        
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"   {category}: {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            
            try:
                # Charger et prétraiter
                img = tf.keras.preprocessing.image.load_img(
                    img_path, 
                    target_size=(IMG_SIZE, IMG_SIZE)
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0  # Normalisation
                
                images.append(img_array)
                labels.append(idx)
            
            except Exception as e:
                print(f"   ⚠️  Erreur lors du chargement de {img_file}: {e}")
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"\n✅ Dataset chargé : {len(X)} images")
    print(f"   Shape : {X.shape}")
    
    return X, y

def create_model(num_classes=4):
    """Crée le modèle MobileNetV2"""
    print("\n🔨 Construction du modèle...")
    
    # Base MobileNetV2
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Geler les couches de base
    base_model.trainable = False
    
    # Modèle complet
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    print("✅ Modèle construit")
    
    return model

def plot_training_history(history, save_path):
    """Visualise les courbes d'entraînement"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Accuracy pendant l\'entraînement', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Loss pendant l\'entraînement', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Graphiques sauvegardés : {save_path}")

def main():
    """Fonction principale d'entraînement"""
    
    # 1. Charger les données
    X, y = load_dataset(DATA_DIR)
    
    if len(X) == 0:
        print("\n❌ ERREUR : Aucune image trouvée!")
        print("\n📝 INSTRUCTIONS :")
        print(f"   1. Placez vos images dans : {DATA_DIR}")
        print("   2. Structure requise :")
        print("      data/building_images/train/")
        print("         ├── good_condition/")
        print("         ├── minor_damage/")
        print("         ├── moderate_damage/")
        print("         └── severe_damage/")
        print("\n   Minimum recommandé : 50 images par catégorie")
        sys.exit(1)
    
    # 2. Split train/validation
    print("\n✂️  Split des données...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"   Training set: {len(X_train)} images")
    print(f"   Validation set: {len(X_val)} images")
    
    # 3. Créer le modèle
    model = create_model(num_classes=len(CATEGORIES))
    
    # 4. Compiler
    print("\n⚙️  Compilation du modèle...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 5. Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(SAVE_DIR, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # 6. Entraînement
    print("\n🚀 Démarrage de l'entraînement...")
    print("=" * 70)
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Sauvegarder le modèle final
    final_model_path = os.path.join(SAVE_DIR, 'mobilenet_building_detector.h5')
    model.save(final_model_path)
    print(f"\n💾 Modèle final sauvegardé : {final_model_path}")
    
    # 8. Visualisation
    plot_path = os.path.join(SAVE_DIR, f'training_history_{timestamp}.png')
    plot_training_history(history, plot_path)
    
    # 9. Résumé final
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 70)
    print(f"📁 Modèle sauvegardé : {final_model_path}")
    print(f"📊 Graphiques : {plot_path}")
    print(f"🎯 Meilleure val_accuracy : {max(history.history['val_accuracy']):.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()