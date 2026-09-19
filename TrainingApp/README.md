# TrainingApp

Módulo para el preprocesamiento, aumento de datos (*data augmentation*) y entrenamiento de la red neuronal convolucional (CNN) en Lengua de Señas Mexicana (LSM).

---

## 📁 Archivos del módulo

* **`preprocess.py`**: Recorta el área de la mano (ROI) usando MediaPipe (compatible con versión Legacy y Tasks API en Python 3.13+) y convierte las imágenes a escala de grises y tamaño $200 \times 200$.
* **`augment_data.py`**: Aplica técnicas de aumento de datos balanceadas (50% mano derecha / 50% mano izquierda) sobre el dataset original.
* **`training.py`**: Define la arquitectura de la red CNN y realiza el proceso de entrenamiento.

---

## 🎨 Aumento de Datos Balanceado (`augment_data.py`)

Para evitar que el modelo aprenda con sesgo hacia una sola mano (por ejemplo, si todas las fotos de captura se hicieron con la mano derecha) y garantizar que reconozca señas hechas tanto con la **mano derecha** como con la **mano izquierda** en tiempo real, se implementa una estrategia de aumento de datos geométrico y espejado simétrico.

### Tabla de distribución de imágenes generadas por cada toma original

Por cada foto de origen leída, el script genera **8 imágenes** distribuidas equitativamente:

| Tipo de Imagen | Cantidad | Orientación / Descripción |
| :--- | :---: | :--- |
| **Imagen Original intacta** (`_original.jpg`) | 1 | Mano Derecha (Base) |
| **Imagen Espejo limpia** (`_mirror.jpg`) | 1 | Mano Izquierda (Base) |
| **Aumentos de la Original** (`_orig_aug0.jpg` a `2`) | 3 | Mano Derecha con variaciones geométricas |
| **Aumentos del Espejo** (`_mirr_aug0.jpg` a `2`) | 3 | Mano Izquierda con variaciones espejo exactas |
| **TOTAL POR IMAGEN** | **8** | **4 Derechas (50%) vs 4 Izquierdas (50%)** |

---

### ⚙️ Parámetros de transformación (`ImageDataGenerator`)

A cada imagen original se le aplican transformaciones aleatorias dentro de los siguientes rangos:

* **Rotación aleatoria:** $\pm 30^\circ$ (`rotation_range=30`)
* **Desplazamiento horizontal:** $20\%$ del ancho (`width_shift_range=0.2`)
* **Desplazamiento vertical:** $20\%$ del alto (`height_shift_range=0.2`)
* **Deformación / Sesgo angular:** $15^\circ$ (`shear_range=15`)
* **Zoom aleatorio:** $70\%$ a $140\%$ del tamaño (`zoom_range=[0.7, 1.4]`)
* **Espejeo simétrico:** A cada lote transformado de la mano derecha (`batch_orig`) se le aplica `cv2.flip(batch_orig, 1)` para garantizar que cada rotación, zoom o deformación de la mano derecha tenga su par idéntico en la mano izquierda.

---

## 🚀 Ejemplo de Uso

```bash
# 1. Preprocesar y segmentar manos
python TrainingApp/preprocess.py

# 2. Generar aumentos de datos balanceados 50/50
python TrainingApp/augment_data.py

# 3. Entrenar la Red Neuronal CNN
python TrainingApp/training.py
```
