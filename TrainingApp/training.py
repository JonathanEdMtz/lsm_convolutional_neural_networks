import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import pickle
from pathlib import Path

PATH_DATA = Path(__file__).resolve().parent
BASE_DIR = PATH_DATA.parent

# Configuraciones para Dataset Masivo
TAMANO_IMG = 200
RUTA_DATASET = BASE_DIR / "data" / "data_training" / "lsm_aument_50der_50izq"
NUM_CLASES = 21
BATCH_SIZE = 128   # Optimizado para procesar grandes conjuntos de datos de forma fluida
EPOCHS = 40

SAVE_DIR = BASE_DIR / "build"
name = SAVE_DIR / "train_cnn_v2"

if not RUTA_DATASET.exists():
    print(f"[!] Error: La ruta del dataset {RUTA_DATASET} no existe.")
    exit(1)

print("=" * 70)
print("INICIANDO ENTRENAMIENTO DE RED NEURONAL CONVOLUCIONAL (CNN AVANZADA)")
print(f"  - Ruta del Dataset: {RUTA_DATASET}")
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
    # Reduce la tasa de aprendizaje si la precisión de validación se estanca
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    # Detiene el entrenamiento si ya no mejora para evitar sobreajuste
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
duracion = fin - inicio

os.makedirs(name.parent, exist_ok=True)
modeloCNN.save(str(name) + ".h5")
print(f"\n[✓] Modelo guardado exitosamente en: '{name}.h5'")

# Guardar el historial de entrenamiento
with open(str(name) + ".pkl", "wb") as f:
    pickle.dump(history.history, f)

# Graficar rendimiento
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.title('Rendimiento del Modelo CNN Avanzado (LSM)')
plt.legend()
plt.grid(True)
texto_tiempo = f"Duración total: {duracion:.2f} segundos ({duracion/60:.2f} min)"
plt.text(0.5, 0.05, texto_tiempo, fontsize=10, color='gray', transform=plt.gca().transAxes)
plt.tight_layout()
plt.savefig(str(name) + ".png")
plt.show()