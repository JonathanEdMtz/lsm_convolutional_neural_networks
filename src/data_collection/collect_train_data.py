import cv2
import os
import time
import mediapipe as mp

# === Configuración general ===
FRAMES_OBJETIVO = 500
TAMANO_IMG = 200
CLASES = range(21)
CLASES_LSM = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L',
              'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

# === Rutas configurables ===
REF_DIR = os.path.join(os.getcwd(), 'datasets', 'imgs_referencias')
CARPETA_DESTINO = os.path.join(os.getcwd(), 'data', 'lsm3')  # <-- CAMBIA AQUÍ

# === Inicialización de cámara ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise Exception("No se pudo acceder a la cámara.")

# === Inicialización de MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# === Función para segmentar la mano ===
def segmentar_mano_procesada(imagen_bgr):
    h, w, _ = imagen_bgr.shape
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    resultado = hands.process(imagen_rgb)

    if not resultado.multi_hand_landmarks:
        return None

    landmarks = resultado.multi_hand_landmarks[0].landmark
    x_coords = [int(lm.x * w) for lm in landmarks]
    y_coords = [int(lm.y * h) for lm in landmarks]
    x_min, x_max = max(min(x_coords) - 40, 0), min(max(x_coords) + 40, w)
    y_min, y_max = max(min(y_coords) - 40, 0), min(max(y_coords) + 60, h)

    mano_segmentada = imagen_bgr[y_min:y_max, x_min:x_max]
    mano_gris = cv2.cvtColor(mano_segmentada, cv2.COLOR_BGR2GRAY)
    return cv2.resize(mano_gris, (TAMANO_IMG, TAMANO_IMG))

# === Captura por clase ===
for clase in CLASES:
    letra = CLASES_LSM[clase]
    ruta_ref = os.path.join(REF_DIR, f"{clase}.jpg")
    carpeta_guardado = os.path.join(CARPETA_DESTINO, str(clase))
    os.makedirs(carpeta_guardado, exist_ok=True)

    # Cargar imagen de referencia
    img_ref = cv2.imread(ruta_ref)
    if img_ref is None:
        print(f"[!] Imagen de referencia no encontrada: {ruta_ref}")
        continue
    img_ref = cv2.resize(img_ref, (TAMANO_IMG, TAMANO_IMG))

    # Contador de fotos ya existentes
    archivos = [f for f in os.listdir(carpeta_guardado) if f.startswith("foto_") and f.endswith(".jpg")]
    indices = [int(f.split("_")[1].split(".")[0]) for f in archivos if "_" in f]
    foto_contador = max(indices) + 1 if indices else 0

    print(f"\n>>> Clase {clase} ({letra}) lista para capturar {FRAMES_OBJETIVO} frames.")
    print("Presiona ESPACIO una vez para empezar la grabación. Presiona ESC para saltar clase.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_mostrar = frame.copy()
        frame_mostrar[20:220, 20:220] = img_ref
        cv2.putText(frame_mostrar, f'Seña: {letra}', (240, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        cv2.putText(frame_mostrar, f'Frames guardados: {foto_contador}', (240, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Captura Dataset LSM (Segmentado)', frame_mostrar)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            print(f"[!] Clase {clase} saltada.")
            break
        elif key == 32:  # ESPACIO
            print(f"[*] Grabando frames segmentados para clase {clase} ({letra})...")
            frames_capturados = 0

            while frames_capturados < FRAMES_OBJETIVO:
                ret, frame_actual = cap.read()
                if not ret:
                    print("[!] Error al capturar frame.")
                    break

                mano = segmentar_mano_procesada(frame_actual)
                if mano is None:
                    continue

                nombre_foto = os.path.join(carpeta_guardado, f'foto_{foto_contador}.png')
                cv2.imwrite(nombre_foto, mano)
                foto_contador += 1
                frames_capturados += 1

                vista_previa = cv2.resize(mano, (400, 400))
                cv2.putText(vista_previa, f'Rec {frames_capturados}/{FRAMES_OBJETIVO}', (10, 380),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Captura Dataset LSM (Segmentado)', vista_previa)

                if cv2.waitKey(1) == 27:
                    print("[!] Grabación cancelada.")
                    break

            print(f"[✓] {frames_capturados} frames capturados para clase {clase} ({letra}).")
            break

cap.release()
cv2.destroyAllWindows()
