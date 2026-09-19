import os
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
import argparse

# --- MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def segmentar_mano(imagen_bgr):
    """
    Detecta los puntos clave de la mano usando MediaPipe y recorta la caja delimitadora.
    """
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
    return imagen_bgr[y_min:y_max, x_min:x_max]

def preprocesar(img_bgr, grayscale=True, target_size=None):
    """
    Preprocesa la imagen aplicando conversión a escala de grises y/o redimensionamiento.
    """
    img = img_bgr
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if target_size is not None:
        img = cv2.resize(img, target_size)
    return img

def procesar_carpetas(base_path, dest_path, grayscale=True, target_size=(200, 200)):
    """
    Recorre los subdirectorios numerados del dataset, segmenta y guarda las imágenes procesadas.
    """
    base_path = Path(base_path)
    dest_path = Path(dest_path)
    
    # Recorrer carpetas de clases (00 a 20)
    for i in range(21):
        subcarpeta = f"{i:02}"
        origen = base_path / subcarpeta
        destino = dest_path / subcarpeta

        # Crear carpeta de destino si no existe
        os.makedirs(destino, exist_ok=True)

        if not origen.exists():
            print(f"[!] Carpeta no encontrada: {origen}")
            continue

        archivos = [f for f in os.listdir(origen) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"[{subcarpeta}] Procesando {len(archivos)} imágenes...")

        for archivo in archivos:
            ruta_img = origen / archivo
            img = cv2.imread(str(ruta_img))

            if img is None:
                print(f"[!] No se pudo leer: {ruta_img.name}")
                continue

            mano = segmentar_mano(img)
            if mano is None:
                print(f"[!] Mano no detectada: {archivo}")
                continue

            procesada = preprocesar(mano, grayscale=grayscale, target_size=target_size)
            salida = destino / archivo

            if not cv2.imwrite(str(salida), procesada):
                print(f"[X] Error al guardar: {salida}")

    print(f"\n✅ Proceso completado. Revisa la carpeta '{dest_path}'")

if __name__ == '__main__':
    APP_DIR = Path(__file__).resolve().parent
    BASE_DIR = APP_DIR.parent

    default_input = BASE_DIR / 'data' / 'data_validation' / 'lsm_entorno_semicontrolado'
    default_output = BASE_DIR / 'data' / 'data_validation' / 'lsm_entorno_semicontrolado_prueba'

    parser = argparse.ArgumentParser(description="Segmentación y preprocesamiento genérico de manos para LSM.")
    parser.add_argument('--input', type=str, default=str(default_input), help="Ruta de la carpeta original")
    parser.add_argument('--output', type=str, default=str(default_output), help="Ruta de la carpeta de destino")
    parser.add_argument('--rgb', action='store_true', help="Guardar a color (por defecto convierte a escala de grises)")
    parser.add_argument('--no-resize', action='store_true', help="Mantener el tamaño original (por defecto redimensiona a 200x200)")
    parser.add_argument('--size', type=int, default=200, help="Tamaño de redimensión (ej. 200 para 200x200)")
    
    args = parser.parse_args()
    
    grayscale = not args.rgb
    target_size = None if args.no_resize else (args.size, args.size)
    
    print("=" * 50)
    print("INICIANDO SEGMENTACIÓN GENÉRICA")
    print(f"  - Entrada:     {args.input}")
    print(f"  - Salida:      {args.output}")
    print(f"  - Formato:     {'Escala de grises (1 canal)' if grayscale else 'Color BGR (3 canales)'}")
    print(f"  - Redimensión: {f'{args.size}x{args.size}' if target_size else 'Ninguna'}")
    print("=" * 50)
    
    procesar_carpetas(args.input, args.output, grayscale=grayscale, target_size=target_size)
