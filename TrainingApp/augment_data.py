import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

RUTA_ORIGEN = 'lsm'         # Dataset original
RUTA_DESTINO = 'lsm_aument'  # Dataset aumentado

AUMENTOS_POR_IMAGEN = 6  # Aumentos aleatorios 

# Recorremos todas las clases
for clase in os.listdir(RUTA_ORIGEN):
    ruta_clase_origen = os.path.join(RUTA_ORIGEN, clase)
    ruta_clase_destino = os.path.join(RUTA_DESTINO, clase)
    os.makedirs(ruta_clase_destino, exist_ok=True)

    for img_name in os.listdir(ruta_clase_origen):
        img_path = os.path.join(ruta_clase_origen, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        base_name = os.path.splitext(img_name)[0]

        # Guarda imagen original intacta
        cv2.imwrite(os.path.join(ruta_clase_destino, f"{base_name}_original.jpg"), img)

        # Crea y guarda imagen espejada manualmente
        espejo = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(ruta_clase_destino, f"{base_name}_mirror.jpg"), espejo)

        # Generador
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

        # aumentos aleatorios
        for i in range(AUMENTOS_POR_IMAGEN):
            batch = next(gen)[0] * 255  # Desnormaliza
            batch = np.clip(batch, 0, 255).astype(np.uint8)
            nombre_aug = f"{base_name}_aug{i}.jpg"
            cv2.imwrite(os.path.join(ruta_clase_destino, nombre_aug), batch)
