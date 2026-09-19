import os
import json
import time
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PATH_DATA = Path(__file__).resolve().parent
BASE_DIR = PATH_DATA.parent

# Configuraciones para Dataset Masivo
TAMANO_IMG = 200
RUTA_DATASET = BASE_DIR / "data" / "data_training" / "lsm_aumentado_50der_50izq"
NUM_CLASES = 21
BATCH_SIZE = 128   # Optimizado para procesar grandes conjuntos de datos de forma fluida
EPOCHS = 40

SAVE_DIR = BASE_DIR / "build"
name = SAVE_DIR / "DeepCNNv1"

if not RUTA_DATASET.exists():
    print(f"[!] Error: La ruta del dataset {RUTA_DATASET} no existe.")
    exit(1)

# DETECCIÓN DE HARDWARE (GPU / CPU)
gpus = tf.config.list_physical_devices('GPU')
dispositivo_usado = f"GPU: {tf.config.experimental.get_device_details(gpus[0])['device_name']}" if gpus else "CPU (Sin GPU aceleradora)"

print("=" * 70)
print("INICIANDO ENTRENAMIENTO DE RED NEURONAL CONVOLUCIONAL (CNN AVANZADA)")
print(f"  - Fecha/Hora:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  - Hardware Usado:  {dispositivo_usado}")
print(f"  - Ruta Dataset:    {RUTA_DATASET}")
print(f"  - Tamaño Imagen:   {TAMANO_IMG}x{TAMANO_IMG} (1 canal - Escala de Grises)")
print(f"  - Batch Size:      {BATCH_SIZE}")
print(f"  - Épocas Máximas:  {EPOCHS}")
print("=" * 70)

# Cargar dataset mediante generador con normalización 1/255 y split de validación 20%
datagen = ImageDataGenerator(
    rescale=1./255,              
    validation_split=0.20       
)

train_generator = datagen.flow_from_directory(
    RUTA_DATASET,
    target_size=(TAMANO_IMG, TAMANO_IMG),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
)

val_generator = datagen.flow_from_directory(
    RUTA_DATASET,
    target_size=(TAMANO_IMG, TAMANO_IMG),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
)

# ARQUITECTURA CNN AVANZADA CON BATCH NORMALIZATION Y GLOBAL AVERAGE POOLING
modeloCNN = tf.keras.models.Sequential([
    # Bloque Convolucional 1
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(TAMANO_IMG, TAMANO_IMG, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Bloque Convolucional 2
    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Bloque Convolucional 3
    tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Bloque Convolucional 4 (Extracción profunda de características)
    tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Reducción espacial eficiente (Global Average Pooling previene overfitting)
    tf.keras.layers.GlobalAveragePooling2D(),

    # Capas Clasificadoras Densas
    tf.keras.layers.Dense(512),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(256),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(NUM_CLASES, activation='softmax')
])

modeloCNN.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

modeloCNN.summary()

# CALLBACKS INTELIGENTES PARA OPTIMIZAR ENTRENAMIENTO
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
]

steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
validation_steps = max(1, val_generator.samples // BATCH_SIZE)

inicio = time.time()

history = modeloCNN.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks
)

fin = time.time()
duracion_segundos = fin - inicio
duracion_minutos = duracion_segundos / 60.0

# 1. Guardar el modelo entrenado
os.makedirs(name.parent, exist_ok=True)
modeloCNN.save(str(name) + ".h5")
print(f"\n[✓] Modelo H5 guardado exitosamente en: '{name}.h5'")

# 2. Guardar el historial pickle
with open(str(name) + ".pkl", "wb") as f:
    pickle.dump(history.history, f)

# 3. Extraer métricas clave
best_epoch = int(np.argmax(history.history['val_accuracy'])) + 1
best_val_acc = float(np.max(history.history['val_accuracy'])) * 100
best_train_acc = float(history.history['accuracy'][best_epoch - 1]) * 100
min_val_loss = float(np.min(history.history['val_loss']))
min_train_loss = float(history.history['loss'][best_epoch - 1])

# 4. Guardar Reporte en JSON (Bitácora de experimentos)
reporte_datos = {
    "modelo_nombre": name.name,
    "fecha_entrenamiento": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "dispositivo": dispositivo_usado,
    "duracion_segundos": round(duracion_segundos, 2),
    "duracion_minutos": round(duracion_minutos, 2),
    "total_epocas_ejecutadas": len(history.history['accuracy']),
    "mejor_epoca": best_epoch,
    "precision_entrenamiento_max": round(best_train_acc, 2),
    "precision_validacion_max": round(best_val_acc, 2),
    "perdida_entrenamiento_min": round(min_train_loss, 4),
    "perdida_validacion_min": round(min_val_loss, 4),
    "total_parametros": int(modeloCNN.count_params()),
    "batch_size": BATCH_SIZE,
    "tamano_imagen": f"{TAMANO_IMG}x{TAMANO_IMG}"
}

ruta_reporte_json = str(name) + "_report.json"
with open(ruta_reporte_json, "w", encoding="utf-8") as f:
    json.dump(reporte_datos, f, indent=4, ensure_ascii=False)
print(f"[✓] Reporte JSON de seguimiento guardado en: '{ruta_reporte_json}'")

# 5. Generar Gráfica Profesional de 2 Paneles (Precisión y Pérdida)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Precisión (Accuracy)
ax1.plot(history.history['accuracy'], label='Entrenamiento', color='#1f77b4', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validación', color='#ff7f0e', linewidth=2)
ax1.axvline(best_epoch - 1, color='red', linestyle='--', label=f'Mejor Época ({best_epoch})')
ax1.set_title('Precisión del Modelo (Accuracy)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Precisión')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

# Subplot 2: Pérdida (Loss)
ax2.plot(history.history['loss'], label='Entrenamiento', color='#1f77b4', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validación', color='#ff7f0e', linewidth=2)
ax2.axvline(best_epoch - 1, color='red', linestyle='--', label=f'Mejor Época ({best_epoch})')
ax2.set_title('Pérdida del Modelo (Loss)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Épocas')
ax2.set_ylabel('Pérdida (Loss)')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)

# Pie de gráfica con métricas e información técnica
pie_texto = (f"Duración: {duracion_minutos:.2f} min ({dispositivo_usado}) | "
             f"Max Val Accuracy: {best_val_acc:.2f}% en Época {best_epoch}")
fig.suptitle(f"Seguimiento de Entrenamiento: {name.name}", fontsize=14, fontweight='bold')
fig.text(0.5, 0.01, pie_texto, ha='center', fontsize=10, color='#555555')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(str(name) + ".png", dpi=300)
print(f"[✓] Gráfica de rendimiento guardada en: '{name}.png'")
plt.show()