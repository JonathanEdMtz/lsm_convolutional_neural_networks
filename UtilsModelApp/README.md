# UtilsModelApp

Módulo utilitario para inspección y conversión de modelos entrenados.

## Herramientas

* **`inspect_model.py`**: Muestra la arquitectura, capas, formas de salida y parámetros del modelo `.h5`.
* **`convert_tflite.py`**: Convierte y optimiza el modelo `.h5` a formato TensorFlow Lite (`.tflite`).

## Uso

```bash
# Inspeccionar modelo
python UtilsModelApp/inspect_model.py

# Convertir a TFLite
python UtilsModelApp/convert_tflite.py
```
