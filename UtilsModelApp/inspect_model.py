import os
import tensorflow as tf
from pathlib import Path

# Ruta del modelo a inspeccionar
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "build" / "train_cnn_v1.h5"

print("=" * 60)
print(f"INSPECCIONANDO MODELO: {MODEL_PATH}")
print("=" * 60)

if not MODEL_PATH.exists():
    print(f"[!] Error: No se encontró el modelo en {MODEL_PATH}")
    exit(1)

# Cargar el modelo
model = tf.keras.models.load_model(MODEL_PATH)

# Mostrar resumen
model.summary()

# Imprimir detalles capa por capa
print("\n" + "=" * 60)
print("DETALLE DE CAPAS")
print("=" * 60)

for layer in model.layers:
    print("Nombre:", layer.name)
    print("Tipo:", layer.__class__.__name__)
    try:
        print("Forma de salida:", layer.output_shape)
    except:
        pass
    if hasattr(layer, 'activation'):
        print("Activación:", layer.activation.__name__)
    if 'Dropout' in layer.__class__.__name__:
        print("Dropout rate:", layer.rate)
    print("Parámetros entrenables:", layer.count_params())
    print("-" * 30)
