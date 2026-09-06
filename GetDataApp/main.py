import os
import time
import cv2
from pathlib import Path
from config import (
    REF_DIR,
    TEST_OUTPUT_DIR,
    NUM_FOTOS,
    TIEMPO_ENTRE_FOTOS,
    CLASES,
    CLASES_LSM,
    REF_IMG_SIZE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)

# === Inicialización de cámara ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    raise Exception("No se pudo acceder a la cámara.")

print("=" * 60)
print("INICIANDO CAPTURA DE DATASET LSM")
print(f"  - Imágenes de referencia: {REF_DIR}")
print(f"  - Carpeta de destino:     {TEST_OUTPUT_DIR}")
print("=" * 60)

for clase in CLASES:
    letra = CLASES_LSM[clase]
    ruta_ref = REF_DIR / f"{clase}.jpg"
    carpeta_guardado = TEST_OUTPUT_DIR / str(clase)
    carpeta_guardado.mkdir(parents=True, exist_ok=True)

    img_ref = cv2.imread(str(ruta_ref))
    if img_ref is None:
        print(f"[!] No se encontró imagen de referencia: {ruta_ref}")
        continue
    img_ref = cv2.resize(img_ref, REF_IMG_SIZE)

    # Calcular índice inicial para evitar sobrescribir fotos existentes
    archivos_existentes = [f for f in os.listdir(carpeta_guardado) if f.startswith("foto_") and f.endswith(".jpg")]
    indices = [
        int(f.split("_")[1].split(".")[0])
        for f in archivos_existentes
        if "_" in f and f.split("_")[1].split(".")[0].isdigit()
    ]
    foto_contador = max(indices) + 1 if indices else 0

    print(f"\n>>> Clase {clase} ({letra}) lista para capturar {NUM_FOTOS} fotos.")
    print("Presiona ESPACIO una sola vez para capturar automáticamente las fotos.")
    print("Presiona ESC para saltar a la siguiente clase.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Error al leer de la cámara.")
            break

        frame_mostrar = frame.copy()
        # Pegar imagen de referencia en la esquina superior izquierda
        frame_mostrar[20:220, 20:220] = img_ref

        # Mostrar etiquetas en vivo
        cv2.putText(frame_mostrar, f'Seña: {letra}', (240, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        cv2.putText(frame_mostrar, f'Fotos guardadas: {foto_contador}', (240, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Captura Dataset LSM (HD)', frame_mostrar)

        key = cv2.waitKey(1)
        if key == 27:  # Tecla ESC
            print(f"[!] Clase {clase} saltada por el usuario.")
            break
        elif key == 32:  # Tecla ESPACIO
            print(f"[*] Capturando {NUM_FOTOS} fotos para clase {clase} ({letra})...")
            for i in range(NUM_FOTOS):
                ret, frame_actual = cap.read()
                if not ret:
                    print("[!] Error al capturar frame.")
                    break

                nombre_foto = carpeta_guardado / f'foto_{foto_contador}.jpg'
                cv2.imwrite(str(nombre_foto), frame_actual)
                print(f'[✓] Foto {i+1}/{NUM_FOTOS} guardada: {nombre_foto}')
                foto_contador += 1

                # Mostrar retroalimentación en la ventana
                cv2.putText(frame_actual, f'Guardada {i+1}/{NUM_FOTOS}', (50, 650),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow('Captura Dataset LSM (HD)', frame_actual)
                cv2.waitKey(500)
                if i < NUM_FOTOS - 1:
                    time.sleep(TIEMPO_ENTRE_FOTOS)

            print(f"[✓] Captura de clase {clase} ({letra}) completada.")
            break

cap.release()
cv2.destroyAllWindows()
