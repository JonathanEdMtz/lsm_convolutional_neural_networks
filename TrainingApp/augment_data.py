import os
from pathlib import Path
import cv2
import numpy as np

# Rutas base del proyecto
DIR_ACTUAL = Path(__file__).resolve().parent
BASE_DIR = DIR_ACTUAL.parent

# Carpeta con las imágenes originales a aumentar
RUTA_ORIGEN = BASE_DIR / "data" / "data_training" / "lsm"

# Carpeta donde se guardará el nuevo dataset aumentado
RUTA_DESTINO = BASE_DIR / "data" / "data_training" / "lsm_aumentado_50der_50izq"

# Cantidad de aumentos por orientación
# 3 variaciones derecha + 3 variaciones izquierda (+ 1 base der + 1 base izq) = 8 imágenes por foto
AUMENTOS_POR_ORIENTACION = 3


def generar_aumento_opencv(imagen):
    """
    Aplica transformaciones geométricas aleatorias a la imagen recibida:
      - Rotación: entre -30° y +30°
      - Zoom: escala entre 75% y 125%
      - Desplazamiento horizontal (dx) y vertical (dy): hasta ±12%
    
    Usa OpenCV puro (warpAffine), rápido y sin dependencias externas.
    """
    alto, ancho = imagen.shape[:2]

    # 1. Definir valores aleatorios de rotación, zoom y traslación
    angulo = np.random.uniform(-30, 30)
    zoom = np.random.uniform(0.75, 1.25)
    dx = np.random.uniform(-0.12, 0.12) * ancho
    dy = np.random.uniform(-0.12, 0.12) * alto

    # 2. Calcular la matriz de rotación y zoom tomando como referencia el centro
    centro = (ancho / 2, alto / 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, zoom)

    # 3. Sumar el desplazamiento horizontal y vertical en la matriz
    matriz[0, 2] += dx
    matriz[1, 2] += dy

    # 4. Aplicar la transformación a la imagen
    return cv2.warpAffine(
        imagen,
        matriz,
        (ancho, alto),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


# ==============================================================================
# PIPELINE DE AUMENTO DE DATOS BALANCEADO (50% DERECHA / 50% IZQUIERDA)
# ==============================================================================

def aumentar_dataset(origen_path=RUTA_ORIGEN, destino_path=RUTA_DESTINO,
                     aumentos_por_orientacion=AUMENTOS_POR_ORIENTACION):
    """
    Lee las imágenes del directorio de origen y genera exactamente 8 imágenes por cada toma:
      1. Base original intacta (Mano Derecha)
      2. Base en espejo horizontal (Mano Izquierda)
      3. Tres aumentos con variaciones geométricas para Mano Derecha
      4. Tres aumentos simétricos en espejo para Mano Izquierda

    TOTAL: 8 imágenes generadas por toma (4 derechas y 4 izquierdas: 50% / 50%).
    """
    origen_path = Path(origen_path)
    destino_path = Path(destino_path)

    # Validar que la carpeta de origen exista
    if not origen_path.exists():
        print(f"[!] Error: La ruta de origen no existe: {origen_path}")
        return

    print("=" * 70)
    print("INICIANDO AUMENTO DE DATOS BALANCEADO (8 IMÁGENES POR TOMA)")
    print(f"  - Origen:        {origen_path}")
    print(f"  - Destino:       {destino_path}")
    print("  - Distribución:  4 Mano Derecha (50%) vs 4 Mano Izquierda (50%)")
    print("=" * 70)

    # Obtener las subcarpetas de clases existentes (00, 01, ..., 20)
    clases = sorted([c for c in os.listdir(origen_path) if (origen_path / c).is_dir()])
    total_generadas = 0

    for clase in clases:
        dir_origen = origen_path / clase
        dir_destino = destino_path / clase
        dir_destino.mkdir(parents=True, exist_ok=True)

        # Filtrar solo archivos de imagen
        archivos = [f for f in os.listdir(dir_origen) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"[{clase}] Generando aumentos para {len(archivos)} imágenes...")

        for archivo in archivos:
            ruta_img = dir_origen / archivo

            # Leer la imagen
            img = cv2.imread(str(ruta_img), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            nombre_base = Path(archivo).stem

            # ------------------------------------------------------------------
            # 1. Base Original (Mano Derecha)
            # ------------------------------------------------------------------
            cv2.imwrite(str(dir_destino / f"{nombre_base}_orig.jpg"), img)
            total_generadas += 1

            # ------------------------------------------------------------------
            # 2. Base Espejo (Mano Izquierda)
            # ------------------------------------------------------------------
            img_espejo = cv2.flip(img, 1)  # 1 = volteo horizontal
            cv2.imwrite(str(dir_destino / f"{nombre_base}_mirr.jpg"), img_espejo)
            total_generadas += 1

            # ------------------------------------------------------------------
            # 3. Generar las 6 variaciones aumentadas (3 Derechas + 3 Izquierdas)
            # ------------------------------------------------------------------
            for i in range(aumentos_por_orientacion):
                # Variación geométrica para Mano Derecha
                aumento_der = generar_aumento_opencv(img)
                cv2.imwrite(str(dir_destino / f"{nombre_base}_orig_aug{i}.jpg"), aumento_der)
                total_generadas += 1

                # Variación simétrica para Mano Izquierda
                aumento_izq = cv2.flip(aumento_der, 1)
                cv2.imwrite(str(dir_destino / f"{nombre_base}_mirr_aug{i}.jpg"), aumento_izq)
                total_generadas += 1

    print(f"\n[✓] Aumento de datos completado exitosamente en: '{destino_path}'")
    print(f"[✓] Total de imágenes generadas: {total_generadas}")


if __name__ == '__main__':
    aumentar_dataset(
        origen_path=RUTA_ORIGEN,
        destino_path=RUTA_DESTINO,
        aumentos_por_orientacion=AUMENTOS_POR_ORIENTACION
    )
