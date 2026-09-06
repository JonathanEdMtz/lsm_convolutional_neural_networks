from pathlib import Path

# === Rutas base ===
GETDATA_APP_DIR = Path(__file__).resolve().parent
BASE_DIR = GETDATA_APP_DIR.parent

# === Rutas de archivos ===
# Las imágenes de referencia (0.jpg a 20.jpg) están en GetDataApp/references
REF_DIR = GETDATA_APP_DIR / "references"

# Carpeta de salida para las imágenes recolectadas
TEST_OUTPUT_DIR = BASE_DIR / "getData" / "abecedario"
#TRAIN_OUTPUT_DIR = BASE_DIR / "data" / "raw"

# === Configuración de captura ===
NUM_FOTOS = 5
TIEMPO_ENTRE_FOTOS = 1
REF_IMG_SIZE = (200, 200)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# === Clases LSM ===
CLASES = range(21)
CLASES_LSM = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L',
    'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y'
]
