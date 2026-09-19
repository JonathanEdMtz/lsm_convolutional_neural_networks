import os
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from pathlib import Path

# Cargar el modelo entrenado
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'build' / 'train_cnn_v1.h5'

if not MODEL_PATH.exists():
    print(f"[!] Error: No se encontró el modelo entrenado en {MODEL_PATH}")
    exit(1)

modelo = tf.keras.models.load_model(MODEL_PATH)

# Mapeo de índices a letras del abecedario LSM
CLASES_LSM = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

# Conexiones de la mano para dibujo
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
    (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)
]

# Inicializar Mediapipe según versión de Python/MediaPipe
USE_TASKS_API = False

if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
else:
    USE_TASKS_API = True
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    TASK_PATH = str(BASE_DIR / 'build' / 'hand_landmarker.task')
    if not os.path.exists(TASK_PATH):
        print(f"[*] Descargando hand_landmarker.task...")
        os.makedirs(os.path.dirname(TASK_PATH), exist_ok=True)
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, TASK_PATH)
        print("[✓] Modelo descargado exitosamente.")

    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    hand_detector = vision.HandLandmarker.create_from_options(options)

# Función para segmentar la mano sin eliminar fondo (recorte interno)
def segmentar_mano(frame, hand_landmarks):
    h, w, _ = frame.shape
    x_coords = [int(lm.x * w) for lm in hand_landmarks]
    y_coords = [int(lm.y * h) for lm in hand_landmarks]

    x_min, x_max = max(min(x_coords) - 40, 0), min(max(x_coords) + 40, w)
    y_min, y_max = max(min(y_coords) - 40, 0), min(max(y_coords) + 60, h)

    return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

# Dibujar conexiones y puntos de la mano
def dibujar_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    for start_idx, end_idx in HAND_CONNECTIONS:
        p1 = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
        p2 = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
        cv2.line(frame, p1, p2, (0, 255, 0), 2)

# Función para preparar imagen para el modelo
def preparar_imagen_mano(img_mano):
    img_mano = cv2.cvtColor(img_mano, cv2.COLOR_BGR2GRAY)
    img_mano = cv2.resize(img_mano, (200, 200))
    img_mano = img_mano / 255.0
    img_mano = np.expand_dims(img_mano, axis=-1)  # (200, 200, 1)
    img_mano = np.expand_dims(img_mano, axis=0)   # (1, 200, 200, 1)
    return img_mano

# Captura de video en tiempo real
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[!] Error: No se pudo acceder a la cámara web.")
    exit()

print("[*] Iniciando Clasificador LSM en tiempo real...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mano_landmarks_list = []

    if USE_TASKS_API:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = hand_detector.detect(mp_image)
        if result.hand_landmarks:
            mano_landmarks_list = result.hand_landmarks
    else:
        resultados = hands.process(img_rgb)
        if resultados.multi_hand_landmarks:
            mano_landmarks_list = [h.landmark for h in resultados.multi_hand_landmarks]

    if mano_landmarks_list:
        for hand_landmarks in mano_landmarks_list:
            mano_recortada, (x1, y1, x2, y2) = segmentar_mano(frame, hand_landmarks)

            if mano_recortada.size != 0:
                entrada = preparar_imagen_mano(mano_recortada)
                prediccion = modelo.predict(entrada, verbose=0)
                clase = np.argmax(prediccion)
                letra = CLASES_LSM[clase]

                # Mostrar letra y caja
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, letra, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

                # Mostrar ventana auxiliar con la mano recortada
                mano_display = cv2.resize(mano_recortada, (300, 300))
                cv2.imshow('Region cuadrada de la mano', mano_display)

            # Dibujar landmarks sobre el frame original
            dibujar_landmarks(frame, hand_landmarks)

    else:
        black_screen = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.imshow('Region cuadrada de la mano', black_screen)

    cv2.imshow('Clasificador LSM en tiempo real', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

