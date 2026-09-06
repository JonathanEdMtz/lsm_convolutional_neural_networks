# GetDataApp - Recolector de Datos LSM 🖐️🇲🇽

**GetDataApp** es una herramienta interactiva en Python para la captura y recolección asistida de imágenes del abecedario en **Lengua de Señas Mexicana (LSM)** a través de la cámara web.

---

## 📁 Estructura del Módulo

```text
GetDataApp/
├── main.py          # Script principal de captura interactiva por cámara web
├── config.py        # Módulo centralizado de configuraciones y rutas
├── references/      # Imágenes guía de referencia (0.jpg a 20.jpg) para cada seña
└── README.md        # Documentación del módulo
```

---

## 🛠️ Archivos Principales

* **`config.py`**:
  Contiene las constantes globales y rutas dinámicas del proyecto (calculadas con `pathlib.Path`). Aquí se definen:
  * `REF_DIR`: Ruta de imágenes guía de referencia.
  * `TEST_OUTPUT_DIR`: Ruta de destino de las fotos capturadas (`/test/eval_v1/`).
  * `NUM_FOTOS`: Cantidad de fotos por clase (por defecto `5`).
  * `TIEMPO_ENTRE_FOTOS`: Intervalo en segundos entre cada captura.
  * `CLASES_LSM`: Lista de las 21 letras/clases del abecedario.

* **`main.py`**:
  Ejecuta la ventana en vivo de OpenCV. Muestra la señal de vídeo de la cámara web junto con la imagen guía de referencia en la esquina superior izquierda.

---

## ⚙️ Requisitos

* Python 3.8+
* Cámara web funcional.
* Librerías requeridas:
  * `opencv-python`

---

## 🚀 Modo de Uso

### 1. Ejecutar la aplicación
Puedes iniciar la captura ejecutando el archivo `main.py` desde cualquier ubicación dentro de la terminal:

```bash
# Opción A: Desde la raíz del proyecto
python GetDataApp/main.py

# Opción B: Entrando al directorio del módulo
cd GetDataApp
python main.py
```

### 2. Controles interactivos

| Tecla | Acción |
| :--- | :--- |
| **`ESPACIO`** | Inicia la secuencia de captura automática para la seña actual. |
| **`ESC`** | Salta la seña actual y avanza a la siguiente clase. |

---

## 📝 Salida de Datos

Las imágenes capturadas se almacenarán organizadas en subcarpetas por el número de clase (`0` a `20`) en la ruta configurada en `config.py` (`/test/eval_v1/<clase>/foto_N.jpg`), preservando las imágenes previamente guardadas sin sobrescribirlas.
