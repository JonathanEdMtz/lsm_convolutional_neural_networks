# Configuración del proyecto

Este proyecto utiliza un entorno virtual de Python para aislar las dependencias y facilitar la reproducción del entorno de desarrollo.

## 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_PROYECTO>
```

---

## 2. Crear el entorno virtual

Si es la primera vez que se ejecuta el proyecto:

```bash
python -m venv .venv

```

---

## 3. Activar el entorno virtual

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Si el entorno está activo, se mostrará algo similar a:

```text
(.venv)
```

---

## 4. Instalar las dependencias

Instalar todas las librerías necesarias desde el archivo `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

## 5. Ejecutar el proyecto

Ejemplo:

```bash
python main.py
```

*(Modificar este comando según el punto de entrada del proyecto.)*

---

## 6. Añadir nueva dependencia

Si durante el desarrollo se instala una nueva librería:

```bash
pip install nombre_libreria
```

Actualizar el archivo de dependencias:

```bash
pip freeze > requirements.txt
```

---

## 🛠️ Requisitos de MediaPipe (Python 3.13+)

La aplicación utiliza la nueva API de MediaPipe Tasks y requiere el modelo `hand_landmarker.task`.

1. **Descarga automática:** Al ejecutar `python main.py`, el sistema detectará si falta el archivo y lo descargará automáticamente.
2. **Descarga manual (Opcional):** Si prefieres descargarlo manualmente, colócalo en la carpeta `build/`:
   - URL: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task


Con estos el entorno quedará configurado.
