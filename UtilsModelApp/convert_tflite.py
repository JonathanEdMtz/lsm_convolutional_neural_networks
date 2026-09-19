import os
import tensorflow as tf
from pathlib import Path

# Ruta del archivo .h5 (modelo entrenado) y destino .tflite
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_H5 = BASE_DIR / "build" / "train_cnn_v1.h5"
OUTPUT_TFLITE = BASE_DIR / "build" / "train_cnn_v1.tflite"

print("=" * 60)
print(f"CONVIRTIENDO MODELO H5 A TFLITE")
print(f"  - Entrada: {MODEL_H5}")
print(f"  - Salida:  {OUTPUT_TFLITE}")
print("=" * 60)

if not MODEL_H5.exists():
    print(f"[!] Error: No se encontró el modelo entrenado en {MODEL_H5}")
    exit(1)

# Cargar el modelo Keras
modelo_keras = tf.keras.models.load_model(MODEL_H5)

# Crear el convertidor
converter = tf.lite.TFLiteConverter.from_keras_model(modelo_keras)

# Activar optimización predeterminada (tamaño y velocidad)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convertir el modelo
tflite_model = converter.convert()

# Guardar archivo .tflite
os.makedirs(os.path.dirname(OUTPUT_TFLITE), exist_ok=True)
with open(OUTPUT_TFLITE, "wb") as f:
    f.write(tflite_model)

print(f"\n[✓] Modelo convertido exitosamente a: {OUTPUT_TFLITE}")
