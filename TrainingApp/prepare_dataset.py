"""
Módulo de Preprocesamiento y Aumento de Datos Unificado para LSM.

Este script ejecuta el pipeline completo:
1. Segmentación de Mano (ROI) con MediaPipe (compatible con Python 3.8 a 3.13+).
2. Conversión a escala de grises y redimensión a 200x200.
3. Aumento de datos geométrico y espejado simétrico 50/50 (Derecha/Izquierda).

Uso:
  - Para entrenamiento: aplicar_aumento = True
  - Para validación:    aplicar_aumento = False
"""

import os
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

# === RUTAS Y CONFIGURACIÓN ===
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

# Detector de manos (MediaPipe Dual: Legacy / Tasks API)
USE_TASKS_API = False

if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
    mp_hands = mp.solutions.hands
    detector_manos = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
else:
    USE_TASKS_API = True
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    TASK_PATH = str(BASE_DIR / 'build' / 'hand_landmarker.task')
    if not os.path.exists(TASK_PATH):
        print("[*] Descargando modelo hand_landmarker.task para MediaPipe Tasks...")
        os.makedirs(os.path.dirname(TASK_PATH), exist_ok=True)
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, TASK_PATH)
        print("[✓] Modelo descargado exitosamente.")

    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector_manos = vision.HandLandmarker.create_from_options(options)


def segmentar_mano(imagen_bgr):
    """
    Detecta la mano en la imagen y devuelve el recorte (ROI) con un margen de seguridad.
    Retorna None si no se detecta ninguna mano.
    """
    alto, ancho = imagen_bgr.shape[:2]
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB) if len(imagen_bgr.shape) == 3 else imagen_bgr

    landmarks_list = []
    if USE_TASKS_API:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_rgb)
        detection_result = detector_manos.detect(mp_image)
        if detection_result.hand_landmarks:
            landmarks_list = detection_result.hand_landmarks[0]
    else:
        resultado = detector_manos.process(imagen_rgb)
        if resultado.multi_hand_landmarks:
            landmarks_list = resultado.multi_hand_landmarks[0].landmark

    if not landmarks_list:
        return None  # No se detectó mano en la imagen

    x_coords = [int(lm.x * ancho) for lm in landmarks_list]
    y_coords = [int(lm.y * alto) for lm in landmarks_list]

    x_min = max(min(x_coords) - 40, 0)
    x_max = min(max(x_coords) + 40, ancho)
    y_min = max(min(y_coords) - 40, 0)
    y_max = min(max(y_coords) + 60, alto)

    return imagen_bgr[y_min:y_max, x_min:x_max]


def procesar_y_redimensionar(img_bgr, target_size=(200, 200), grayscale=True):
    """
    Convierte la imagen recortada a escala de grises y la redimensiona a (200, 200).
    """
    if grayscale and len(img_bgr.shape) == 3:
        img_procesada = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_procesada = img_bgr

    if target_size is not None:
        img_procesada = cv2.resize(img_procesada, target_size)

    return img_procesada


def generar_aumento_opencv(img_gray):
    """
    Aplica transformaciones aleatorias (rotación, zoom, desplazamiento) usando OpenCV puro.
    No requiere dependencias externas como scipy.
    """
    h, w = img_gray.shape[:2]
    angulo = np.random.uniform(-30, 30)
    zoom = np.random.uniform(0.75, 1.25)
    dx = np.random.uniform(-0.12, 0.12) * w
    dy = np.random.uniform(-0.12, 0.12) * h

    centro = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(centro, angulo, zoom)
    M[0, 2] += dx
    M[1, 2] += dy

    return cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def preparar_dataset(origen_path, destino_path, aplicar_aumento=True, aumentos_por_imagen=3, target_size=(200, 200)):
    """
    Ejecuta el pipeline unificado de procesamiento sobre un directorio de clases.
    
    Parámetros:
      - origen_path: Ruta del dataset crudo.
      - destino_path: Ruta donde se guardará el dataset procesado.
      - aplicar_aumento: True para dataset de entrenamiento (50/50 derecha/izquierda), False para validación.
      - aumentos_por_imagen: Número de variaciones por cada orientación de mano.
    """
    origen_path = Path(origen_path)
    destino_path = Path(destino_path)

    if not origen_path.exists():
        print(f"[!] Error: La ruta de origen no existe: {origen_path}")
        return

    print("=" * 70)
    print(f"INICIANDO PIPELINE DE PREPARACIÓN DE DATASET")
    print(f"  - Origen:          {origen_path}")
    print(f"  - Destino:         {destino_path}")
    print(f"  - Aumento de datos:{' ACTIVADO (50% Derecha / 50% Izquierda)' if aplicar_aumento else ' DESACTIVADO (Solo Limpieza ROI)'}")
    print(f"  - Tamaño final:    {target_size[0]}x{target_size[1]} (Escala de Grises)")
    print("=" * 70)

    clases = sorted([c for c in os.listdir(origen_path) if (origen_path / c).is_dir()])
    total_generadas = 0

    for clase in clases:
        dir_origen = origen_path / clase
        dir_destino = destino_path / clase
        dir_destino.mkdir(parents=True, exist_ok=True)

        archivos = [f for f in os.listdir(dir_origen) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"[{clase}] Procesando {len(archivos)} imágenes...")

        for archivo in archivos:
            ruta_img = dir_origen / archivo
            img_bgr = cv2.imread(str(ruta_img))

            if img_bgr is None:
                continue

            base_name = os.path.splitext(archivo)[0]

            # 1. Segmentación de Mano (ROI)
            mano_roi = segmentar_mano(img_bgr)
            if mano_roi is None:
                mano_roi = img_bgr

            # 2. Escala de grises + Redimensión 200x200
            mano_base = procesar_y_redimensionar(mano_roi, target_size=target_size, grayscale=True)

            # 3. Guardar Base Original (Mano Derecha)
            cv2.imwrite(str(dir_destino / f"{base_name}_orig.jpg"), mano_base)
            total_generadas += 1

            # 4. Guardar Base Espejo (Mano Izquierda)
            mano_espejo = cv2.flip(mano_base, 1)
            cv2.imwrite(str(dir_destino / f"{base_name}_mirr.jpg"), mano_espejo)
            total_generadas += 1

            # 5. Aplicar Aumento de Datos si está activado (Para Entrenamiento)
            if aplicar_aumento:
                for i in range(aumentos_por_imagen):
                    # Aumento para Mano Derecha usando OpenCV (sin dependencia de scipy)
                    batch_orig_gray = generar_aumento_opencv(mano_base)

                    # Guardar Aumento Mano Derecha
                    cv2.imwrite(str(dir_destino / f"{base_name}_orig_aug{i}.jpg"), batch_orig_gray)
                    total_generadas += 1

                    # Guardar Aumento Mano Izquierda (Espejo Simétrico)
                    batch_mirr_gray = cv2.flip(batch_orig_gray, 1)
                    cv2.imwrite(str(dir_destino / f"{base_name}_mirr_aug{i}.jpg"), batch_mirr_gray)
                    total_generadas += 1

    print(f"\n[✓] Proceso completado exitosamente en: '{destino_path}'")
    print(f"[✓] Total de imágenes listas en el dataset: {total_generadas}")


if __name__ == '__main__':
    # === EJEMPLO DE USO ===
    
    # 1. Para Dataset de Entrenamiento (CON aumento 50/50):
    DIR_ENTRENAMIENTO_RAW = BASE_DIR / 'data' / 'data_training' / 'lsm'
    DIR_ENTRENAMIENTO_OUT = BASE_DIR / 'data' / 'data_training' / 'lsm_aumentado_50der_50izq'

    # 2. Para Dataset de Validación (SIN aumento de datos):
    DIR_VALIDACION_RAW = BASE_DIR / 'data' / 'data_validation' / 'lsm_entorno_semicontrolado'
    DIR_VALIDACION_OUT = BASE_DIR / 'data' / 'data_validation' / 'lsm_entorno_semicontrolado_preprocesado'

    # --- Descomentar la línea que desees ejecutar ---
    
    # Procesar Dataset de Entrenamiento (Aumento ACTIVADO)
    preparar_dataset(DIR_ENTRENAMIENTO_RAW, DIR_ENTRENAMIENTO_OUT, aplicar_aumento=True, aumentos_por_imagen=3)

    # Procesar Dataset de Validación (Aumento DESACTIVADO)
    # preparar_dataset(DIR_VALIDACION_RAW, DIR_VALIDACION_OUT, aplicar_aumento=False)
