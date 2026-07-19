import cv2
import os
import time


NUM_FOTOS = 5
TIEMPO_ENTRE_FOTOS = 1
CLASES = range(21)
REF_DIR = os.path.join(os.getcwd(), 'datasets', 'imgs_referencias')

CLASES_LSM = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L',
              'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']


cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise Exception("No se pudo acceder a la cámara.")

for clase in CLASES:
    letra = CLASES_LSM[clase]
    ruta_ref = os.path.join(REF_DIR, f"{clase}.jpg")
    #carpeta_guardado = f'../data/new_test/{clase}/'
    #os.makedirs(carpeta_guardado, exist_ok=True)

    carpeta_base_guardado = os.path.join(os.getcwd(), 'test', 'eval_v1')
    carpeta_guardado = os.path.join(carpeta_base_guardado, str(clase))
    os.makedirs(carpeta_guardado, exist_ok=True)

    img_ref = cv2.imread(ruta_ref)
    if img_ref is None:
        print(f"[!] No se encontró imagen de referencia: {ruta_ref}")
        continue
    img_ref = cv2.resize(img_ref, (200, 200)) 

    archivos_existentes = [f for f in os.listdir(carpeta_guardado) if f.startswith("foto_") and f.endswith(".jpg")]
    indices = [int(f.split("_")[1].split(".")[0]) for f in archivos_existentes if "_" in f]
    foto_contador = max(indices) + 1 if indices else 0

    print(f"\n>>> Clase {clase} ({letra}) lista para capturar {NUM_FOTOS} fotos.")
    print("Presiona ESPACIO una sola vez para capturar automáticamente las fotos.")
    print("Presiona ESC para saltar a la siguiente clase.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_mostrar = frame.copy()
        # Pegar imagen de referencia
        frame_mostrar[20:220, 20:220] = img_ref

        # Mostrar etiquetas
        cv2.putText(frame_mostrar, f'Seña: {letra}', (240, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        cv2.putText(frame_mostrar, f'Fotos guardadas: {foto_contador}', (240, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Captura Dataset LSM (HD)', frame_mostrar)

        key = cv2.waitKey(1)
        if key == 27:
            print(f"[!] Clase {clase} saltada por el usuario.")
            break
        elif key == 32:
            print(f"[*] Capturando {NUM_FOTOS} fotos para clase {clase} ({letra})...")
            for i in range(NUM_FOTOS):
                ret, frame_actual = cap.read()
                if not ret:
                    print("[!] Error al capturar frame.")
                    break

                nombre_foto = os.path.join(carpeta_guardado, f'foto_{foto_contador}.jpg')
                cv2.imwrite(nombre_foto, frame_actual)
                print(f'[✓] Foto {i+1}/{NUM_FOTOS} guardada: {nombre_foto}')
                foto_contador += 1

                # Mostrar retroalimentación
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
