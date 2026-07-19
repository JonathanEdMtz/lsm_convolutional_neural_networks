import tensorflow as tf

# Cargar el modelo
model = tf.keras.models.load_model("train_cnn_v1.h5")

# Mostrar resumen
model.summary()

# Imprimir detalles capa por capa
for layer in model.layers:
    print("Nombre:", layer.name)
    print("Tipo:", layer.__class__.__name__)
    try:
        print("Forma de salida:", layer.output_shape)
    except:
        pass
    if hasattr(layer, 'activation'):
        print("Activación:", layer.activation.__name__)
    if 'Dropout' in layer.__class__.__name__:
        print("Dropout rate:", layer.rate)
    print("Parámetros entrenables:", layer.count_params())
    print("-" * 30)
