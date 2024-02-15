import tensorflow as tf
from tensorflow.keras import layers
def create_resnet_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # Convolutional layers
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # Residual blocks
    num_res_blocks = 1
    for _ in range(num_res_blocks):
        # Identity block
        x_temp = x
        x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Add()([x, x_temp])

    # Global average pooling and output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create the model
model = create_resnet_model()

# Print model summary
model.summary()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
