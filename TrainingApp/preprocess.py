import os
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp


# CONFIGURACIÓN Y RUTAS (CONSTANTES)

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

# Rutas de origen y destino
RUTA_INPUT = BASE_DIR / 'data' / 'data_validation' / 'lsm_entorno_semicontrolado'
RUTA_OUTPUT = BASE_DIR / 'data' / 'data_validation' / 'lsm_entorno_semicontrolado_preprocesado'

# Parámetros de procesamiento
ESCALA_GRISES = True
TAMANO_IMG = (200, 200)


# CONFIGURACIÓN DE MEDIAPIPE (Detección de Manos)
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
        print("[*] Descargando hand_landmarker.task para MediaPipe Tasks...")
        os.makedirs(os.path.dirname(TASK_PATH), exist_ok=True)
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, TASK_PATH)
        print("[✓] Modelo hand_landmarker.task descargado exitosamente.")

    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector_manos = vision.HandLandmarker.create_from_options(options)


def segmentar_mano(imagen_bgr):
    """
    Detecta la mano en la imagen y devuelve el recorte (ROI) con un margen de seguridad.
    Retorna None si no se detecta ninguna mano.
    """
    alto, ancho, _ = imagen_bgr.shape
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

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

    # Obtener coordenadas de los puntos clave de la mano
    x_coords = [int(lm.x * ancho) for lm in landmarks_list]
    y_coords = [int(lm.y * alto) for lm in landmarks_list]

    # Calcular caja delimitadora con margen (padding)
    x_min = max(min(x_coords) - 40, 0)
    x_max = min(max(x_coords) + 40, ancho)
    y_min = max(min(y_coords) - 40, 0)
    y_max = min(max(y_coords) + 60, alto)

    return imagen_bgr[y_min:y_max, x_min:x_max]


def preprocesar(img_bgr, grayscale=True, target_size=(200, 200)):
    """
    Aplica conversión a escala de grises y/o redimensionamiento a la imagen segmentada.
    """
    imagen_procesada = img_bgr

    if grayscale:
        imagen_procesada = cv2.cvtColor(imagen_procesada, cv2.COLOR_BGR2GRAY)

    if target_size is not None:
        imagen_procesada = cv2.resize(imagen_procesada, target_size)

    return imagen_procesada


def procesar_carpetas(base_path, dest_path, grayscale=True, target_size=(200, 200)):
    """
    Recorre los subdirectorios del dataset (carpetas 00 a 20),
    segmenta la mano en cada imagen y guarda el resultado en la carpeta de destino.
    """
    base_path = Path(base_path)
    dest_path = Path(dest_path)

    # Procesar clases numeradas del 00 al 20
    for i in range(21):
        subcarpeta = f"{i:02}"
        origen = base_path / subcarpeta
        destino = dest_path / subcarpeta

        # Crear carpeta de destino si no existe
        destino.mkdir(parents=True, exist_ok=True)

        if not origen.exists():
            print(f"[!] Carpeta de origen no encontrada: {origen}")
            continue

        archivos = [f for f in os.listdir(origen) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"[{subcarpeta}] Procesando {len(archivos)} imágenes...")

        for archivo in archivos:
            ruta_img = origen / archivo
            img = cv2.imread(str(ruta_img))

            if img is None:
                print(f"[!] No se pudo leer la imagen: {archivo}")
                continue

            mano = segmentar_mano(img)
            if mano is None:
                print(f"[!] Mano no detectada: {archivo}")
                continue

            procesada = preprocesar(mano, grayscale=grayscale, target_size=target_size)
            salida = destino / archivo

            if not cv2.imwrite(str(salida), procesada):
                print(f"[X] Error al guardar: {salida}")

    print(f"\n[✓] Proceso completado exitosamente en: '{dest_path}'")



# EJECUCIÓN DEL SCRIPT

if __name__ == '__main__':
    print("=" * 60)
    print("INICIANDO SEGMENTACIÓN Y PREPROCESAMIENTO")
    print(f"  - Origen:      {RUTA_INPUT}")
    print(f"  - Destino:     {RUTA_OUTPUT}")
    print(f"  - Formato:     {'Escala de grises (1 canal)' if ESCALA_GRISES else 'Color BGR (3 canales)'}")
    print(f"  - Redimensión: {f'{TAMANO_IMG[0]}x{TAMANO_IMG[1]}' if TAMANO_IMG else 'Ninguna'}")
    print("=" * 60)

    procesar_carpetas(RUTA_INPUT, RUTA_OUTPUT, grayscale=ESCALA_GRISES, target_size=TAMANO_IMG)
