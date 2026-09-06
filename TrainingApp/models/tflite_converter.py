import tensorflow as tf

# Ruta del archivo .h5 (modelo entrenado)
modelo_h5 = "train_cnn_v1.h5"

# Cargar el modelo desde el archivo .h5
modelo_keras = tf.keras.models.load_model(modelo_h5)

# Crear el convertidor desde el modelo Keras cargado
converter = tf.lite.TFLiteConverter.from_keras_model(modelo_keras)

# Activar la optimización para tamaño/velocidad (ideal para móviles)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convertir el modelo a TFLite
tflite_model = converter.convert()

# Guardar el archivo .tflite
with open("train_cnn_v1.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Modelo optimizado y convertido exitosamente a train_cnn_v1.tflite")
