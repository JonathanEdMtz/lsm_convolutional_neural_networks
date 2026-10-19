import os
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp


# CONFIGURACIÓN Y RUTAS (CONSTANTES)

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

# Rutas de origen y destino por defecto (Validación)
RUTA_INPUT = BASE_DIR / 'data' / 'data_validation' / 'lsm_entorno_semicontrolado'
RUTA_OUTPUT = BASE_DIR / 'data' / 'data_validation' / 'lsm_entorno_semicontrolado_preprocesado'

# Parámetros de procesamiento
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


# ==============================================================================
# PREPROCESAMIENTO EN ORDEN SECUENCIAL:
# Entrada: Imagen de entrada
# 1.- Segmentación de la mano
# 2.- Redimensionamiento
# 3.- Conversión a escala de grises
# 4.- Normalización
# Salida: Imagen de salida
# ==============================================================================

# 1. Segmentación de la mano
def segmentar_mano(imagen_bgr):
    """
    1.- Segmentación de la mano:
    Detecta la mano en la imagen usando MediaPipe y recorta la región de interés (ROI)
    con un margen de seguridad. Retorna None si no se detecta la mano.
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

    # Obtener coordenadas de los puntos clave de la mano
    x_coords = [int(lm.x * ancho) for lm in landmarks_list]
    y_coords = [int(lm.y * alto) for lm in landmarks_list]

    # Calcular caja delimitadora con margen (padding)
    x_min = max(min(x_coords) - 40, 0)
    x_max = min(max(x_coords) + 40, ancho)
    y_min = max(min(y_coords) - 40, 0)
    y_max = min(max(y_coords) + 60, alto)

    return imagen_bgr[y_min:y_max, x_min:x_max]


# 2. Redimensionamiento
def redimensionar(imagen, target_size=(200, 200)):
    """
    2.- Redimensionamiento:
    Redimensiona la imagen al tamaño objetivo especificado (por defecto 200x200).
    """
    if target_size is None:
        return imagen
    return cv2.resize(imagen, target_size)


# 3. Conversión a escala de grises
def convertir_escala_grises(imagen):
    """
    3.- Conversión a escala de grises:
    Convierte la imagen de formato BGR a escala de grises (1 canal).
    """
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return imagen


# 4. Normalización
def normalizar(imagen):
    """
    4.- Normalización:
    Normaliza los valores de los píxeles al rango numérico [0.0, 1.0].
    """
    return imagen.astype(np.float32) / 255.0


def preprocesar(imagen_bgr, target_size=TAMANO_IMG, normalizar_salida=True):
    """
    Ejecuta el pipeline completo y ordenado:
    Entrada: Imagen de entrada
      1.- Segmentación de la mano
      2.- Redimensionamiento
      3.- Conversión a escala de grises
      4.- Normalización
    Salida: Imagen de salida
    """
    # 1.- Segmentación de la mano
    mano = segmentar_mano(imagen_bgr)
    if mano is None:
        return None

    # 2.- Redimensionamiento
    img_redimensionada = redimensionar(mano, target_size=target_size)

    # 3.- Conversión a escala de grises
    img_gris = convertir_escala_grises(img_redimensionada)

    # 4.- Normalización
    salida = normalizar(img_gris) if normalizar_salida else img_gris

    return salida


def procesar_carpetas(base_path, dest_path, target_size=TAMANO_IMG):
    """
    Recorre los subdirectorios del dataset, ejecuta el pipeline ordenado
    (Segmentación -> Redimensionamiento -> Escala de grises -> Normalización)
    y guarda las imágenes procesadas en la carpeta de destino.
    """
    base_path = Path(base_path)
    dest_path = Path(dest_path)

    # Detectar clases existentes (ej. 00 al 20)
    subdirs = sorted([d.name for d in base_path.iterdir() if d.is_dir()]) if base_path.exists() else []
    clases = subdirs if subdirs else [f"{i:02}" for i in range(21)]

    for subcarpeta in clases:
        origen = base_path / subcarpeta
        destino = dest_path / subcarpeta
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

            # Pipeline ordenado: 1. Segmentación -> 2. Redimensión -> 3. Grises -> 4. Normalización
            procesada = preprocesar(img, target_size=target_size, normalizar_salida=True)
            if procesada is None:
                print(f"[!] Mano no detectada: {archivo}")
                continue

            salida = destino / archivo

            # Al guardar a archivo de imagen (JPG/PNG), convertimos el rango [0, 1] a uint8 [0, 255]
            if procesada.dtype != np.uint8 and procesada.max() <= 1.0:
                img_guardar = (procesada * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img_guardar = procesada

            if not cv2.imwrite(str(salida), img_guardar):
                print(f"[X] Error al guardar: {salida}")

    print(f"\n[✓] Proceso completado exitosamente en: '{dest_path}'")


# EJECUCIÓN DEL SCRIPT

if __name__ == '__main__':
    print("=" * 60)
    print("INICIANDO PREPROCESAMIENTO SECUENCIAL")
    print("  1. Segmentación de la mano")
    print("  2. Redimensionamiento")
    print("  3. Conversión a escala de grises")
    print("  4. Normalización")
    print(f"  - Origen:      {RUTA_INPUT}")
    print(f"  - Destino:     {RUTA_OUTPUT}")
    print(f"  - Tamaño:      {TAMANO_IMG[0]}x{TAMANO_IMG[1]}")
    print("=" * 60)

    procesar_carpetas(RUTA_INPUT, RUTA_OUTPUT, target_size=TAMANO_IMG)
