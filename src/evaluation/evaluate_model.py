import os
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import mediapipe as mp

# Cargar modelo entrenado
modelo = tf.keras.models.load_model('train_cnn_v1.h5')

# Mapeo de índices a letras (sin j, k, ñ, q, x, z)
CLASES_LSM = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

# Ruta base del conjunto de validación (con imágenes en carpetas 00-20)
BASE_PATH = Path('./test/eval_v1')

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Función para segmentar mano desde imagen
def segmentar_mano(imagen_bgr):
    h, w, _ = imagen_bgr.shape
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    resultado = hands.process(imagen_rgb)

    if not resultado.multi_hand_landmarks:
        return None  # No se detectó mano

    landmarks = resultado.multi_hand_landmarks[0].landmark
    x_coords = [int(lm.x * w) for lm in landmarks]
    y_coords = [int(lm.y * h) for lm in landmarks]
    x_min, x_max = max(min(x_coords) - 40, 0), min(max(x_coords) + 40, w)
    y_min, y_max = max(min(y_coords) - 40, 0), min(max(y_coords) + 60, h)

    return imagen_bgr[y_min:y_max, x_min:x_max]

# Preprocesar imagen segmentada
def preparar_imagen_segmentada(img_mano_bgr):
    img_gray = cv2.cvtColor(img_mano_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (200, 200))
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (200, 200, 1)
    img_array = np.expand_dims(img_array, axis=0)   # (1, 200, 200, 1)
    return img_array

# Variables de evaluación
total_correctas = 0
total_imagenes = 0
porcentajes = []

print("\n--- RESULTADOS POR CLASE (con segmentación de mano) ---\n")

for i, letra in enumerate(CLASES_LSM):
    carpeta = BASE_PATH / f"{i:02}"
    if not carpeta.exists():
        print(f"[!] Carpeta {carpeta} no encontrada, se omite.")
        continue

    archivos = [f for f in os.listdir(carpeta) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    correctas = 0
    total = 0

    for archivo in archivos:
        ruta_img = carpeta / archivo
        imagen = cv2.imread(str(ruta_img))

        if imagen is None:
            print(f"[!] Imagen no leída: {archivo}")
            continue

        mano_segmentada = segmentar_mano(imagen)

        if mano_segmentada is None:
            print(f"[!] Mano no detectada: {archivo}")
            continue

        entrada = preparar_imagen_segmentada(mano_segmentada)
        prediccion = modelo.predict(entrada, verbose=0)
        clase_predicha = np.argmax(prediccion)

        if clase_predicha == i:
            correctas += 1
        total += 1

    incorrectas = total - correctas
    porcentaje = (correctas / total) * 100 if total > 0 else 0
    porcentajes.append(porcentaje)
    total_correctas += correctas
    total_imagenes += total

    print(f"Letra {letra} ({i:02}): Correctas: {correctas} | Incorrectas: {incorrectas} | Precisión: {porcentaje:.2f}%")

# Promedio global
promedio_total = sum(porcentajes) / len(porcentajes) if porcentajes else 0

print("\n--- RESUMEN GLOBAL ---")
print(f"Total imágenes evaluadas: {total_imagenes}")
print(f"Total correctamente clasificadas: {total_correctas}")
print(f"Precisión promedio del sistema: {promedio_total:.2f}%")
