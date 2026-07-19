import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model('train_cnn_v1.h5')

# Mapeo de índices a letras del abecedario LSM
CLASES_LSM = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

# Inicializar Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Función para segmentar la mano sin eliminar fondo (recorte interno)
def segmentar_mano(frame, hand_landmarks):
    h, w, _ = frame.shape
    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

    x_min, x_max = max(min(x_coords) - 40, 0), min(max(x_coords) + 40, w)
    y_min, y_max = max(min(y_coords) - 40, 0), min(max(y_coords) + 60, h)

    return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resultados = hands.process(img_rgb)

    if resultados.multi_hand_landmarks:
        for mano_landmarks in resultados.multi_hand_landmarks:
            mano_recortada, (x1, y1, x2, y2) = segmentar_mano(frame, mano_landmarks)

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
            mp_drawing.draw_landmarks(frame, mano_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        black_screen = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.imshow('Region cuadrada de la mano', black_screen)

    cv2.imshow('Clasificador LSM en tiempo real', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
