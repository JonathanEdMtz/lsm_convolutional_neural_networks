import os
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

PATH_DATA = Path(__file__).resolve().parent
BASE_DIR = PATH_DATA.parent

RUTA_ORIGEN = BASE_DIR / "data" / "data_training" / "lsm"         # Dataset original
RUTA_DESTINO = BASE_DIR / "data" / "data_training" / "lsm_aument2"  # Dataset aumentado

AUMENTOS_POR_IMAGEN = 3  # Generará 3 aumentos originales + 3 aumentos espejados (+ 1 original + 1 espejo = 8 imágenes en total, 50% derecha / 50% izquierda)

if not RUTA_ORIGEN.exists():
    print(f"[!] Error: La ruta de origen {RUTA_ORIGEN} no existe.")
    exit(1)

# Recorremos todas las clases
for clase in os.listdir(RUTA_ORIGEN):
    ruta_clase_origen = os.path.join(RUTA_ORIGEN, clase)
    if not os.path.isdir(ruta_clase_origen):
        continue
    ruta_clase_destino = os.path.join(RUTA_DESTINO, clase)
    os.makedirs(ruta_clase_destino, exist_ok=True)

    for img_name in os.listdir(ruta_clase_origen):
        img_path = os.path.join(ruta_clase_origen, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        base_name = os.path.splitext(img_name)[0]

        # 1. Guardar imagen original (Mano Derecha / Base)
        cv2.imwrite(os.path.join(ruta_clase_destino, f"{base_name}_original.jpg"), img)

        # 2. Guardar imagen espejo limpia (Mano Izquierda / Base)
        espejo = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(ruta_clase_destino, f"{base_name}_mirror.jpg"), espejo)

        # Generador de aumentos aleatorios
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=15,
            zoom_range=[0.7, 1.4],
            fill_mode='nearest'
        )

        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, 0)
        gen = datagen.flow(img_array, batch_size=1)

        # 3. Generar aumentos balanceados 50/50
        for i in range(AUMENTOS_POR_IMAGEN):
            # Aumento para Mano Derecha
            batch_orig = next(gen)[0] * 255  # Desnormaliza
            batch_orig = np.clip(batch_orig, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(ruta_clase_destino, f"{base_name}_orig_aug{i}.jpg"), batch_orig)

            # Espejo idéntico del aumento para Mano Izquierda
            batch_mirr = cv2.flip(batch_orig, 1)
            cv2.imwrite(os.path.join(ruta_clase_destino, f"{base_name}_mirr_aug{i}.jpg"), batch_mirr)

