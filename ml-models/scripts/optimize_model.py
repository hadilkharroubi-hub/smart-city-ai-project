"""
Script d'optimisation du modèle pour mobile (TensorFlow Lite)
"""
import os
import numpy as np
import tensorflow as tf

# Configuration
MODEL_PATH = '../saved_models/mobilenet_building_detector.h5'
TFLITE_OUTPUT_PATH = '../saved_models/mobilenet_building_detector.tflite'
IMG_SIZE = 224

print("=" * 70)
print("OPTIMISATION DU MODÈLE POUR MOBILE (TensorFlow Lite)")
print("=" * 70)

# 1. Charger le modèle
print("\n📂 Chargement du modèle...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modèle chargé")

# Taille du modèle original
original_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"📏 Taille originale : {original_size:.2f} MB")

# 2. Fonction de représentation du dataset
def representative_dataset():
    """Générateur de données pour la quantification"""
    for _ in range(100):
        # Générer des données aléatoires pour la calibration
        data = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        yield [data]

# 3. Conversion en TFLite avec quantification
print("\n🔧 Conversion en TensorFlow Lite avec quantification INT8...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimisations
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Quantification INT8 complète
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convertir
tflite_model = converter.convert()

# 4. Sauvegarder
with open(TFLITE_OUTPUT_PATH, 'wb') as f:
    f.write(tflite_model)

# Taille du modèle TFLite
tflite_size = os.path.getsize(TFLITE_OUTPUT_PATH) / (1024 * 1024)

print("✅ Conversion terminée")
print(f"💾 Modèle TFLite sauvegardé : {TFLITE_OUTPUT_PATH}")
print(f"📏 Taille TFLite : {tflite_size:.2f} MB")
print(f"📉 Réduction : {((original_size - tflite_size) / original_size * 100):.1f}%")

# 5. Test du modèle TFLite
print("\n🧪 Test du modèle TFLite...")

interpreter = tf.lite.Interpreter(model_path=TFLITE_OUTPUT_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"   Input shape: {input_details[0]['shape']}")
print(f"   Input type: {input_details[0]['dtype']}")
print(f"   Output shape: {output_details[0]['shape']}")
print(f"   Output type: {output_details[0]['dtype']}")

# Test avec une image aléatoire
test_image = np.random.randint(0, 256, (1, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

interpreter.set_tensor(input_details[0]['index'], test_image)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f"   Test output shape: {output.shape}")
print("✅ Test réussi")

print("\n" + "=" * 70)
print("✅ OPTIMISATION TERMINÉE")
print("=" * 70)
print("Le modèle TFLite est prêt pour le déploiement mobile!")
print("=" * 70)