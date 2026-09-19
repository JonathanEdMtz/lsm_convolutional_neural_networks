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

# Configuraciones
TAMANO_IMG = 200
RUTA_DATASET = BASE_DIR / "data" / "data_training" / "lsm_aument_50der_50izq"
NUM_CLASES = 21
BATCH_SIZE = 64
EPOCHS = 50

SAVE_DIR = BASE_DIR / "build"
name = SAVE_DIR / "train_cnn_v1"


if not RUTA_DATASET.exists():
    print(f"[!] Error: La ruta del dataset {RUTA_DATASET} no existe.")
    exit(1)

datagen = ImageDataGenerator(
    rescale=1./255,              
    validation_split=0.20       
)

train_generator = datagen.flow_from_directory(
    RUTA_DATASET,
    target_size=(TAMANO_IMG, TAMANO_IMG),  # Redimensiona cada imagen
    color_mode='grayscale',                # Convierte a escala de grises (1 canal)
    batch_size=BATCH_SIZE,
    class_mode='categorical',              # Etiquetas codificadas como one-hot
    subset='training',                     # Usa el subconjunto de entrenamiento
    shuffle=True,                           # Mezcla aleatoriamente las imágenes
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

modeloCNN2_AD = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(TAMANO_IMG, TAMANO_IMG, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),                                                                                 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASES, activation='softmax')
])

modeloCNN2_AD.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
validation_steps = max(1, val_generator.samples // BATCH_SIZE)

inicio = time.time()
history = modeloCNN2_AD.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

fin = time.time()
duracion = fin - inicio

os.makedirs(name.parent, exist_ok=True)
modeloCNN2_AD.save(str(name) + ".h5")

# Guardar el historial de entrenamiento
with open(str(name) + ".pkl", "wb") as f:
    pickle.dump(history.history, f)

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.title('Precisión del modelo por época')
plt.legend()
plt.grid(True)
# Añadir duración
texto_tiempo = f"Duración: {duracion:.2f} segundos"
plt.text(0.5, 0.05, texto_tiempo, fontsize=10, color='gray', transform=plt.gca().transAxes)
plt.tight_layout()
plt.savefig(str(name) + ".png")
plt.show()