import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

sam_dataset = 500
image_height, image_width, num_channels = 224, 224, 3
images = np.random.rand(sam_dataset, image_height, image_width, num_channels)
labels = np.random.randint(0, 4, size=(sam_dataset,))

images = (images - 0.5) / 0.5  

# ResNet model
def resnet_model():
    inputs = tf.keras.Input(shape=(image_height, image_width, num_channels))

    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # Residual blocks
    num_res_blocks = 1
    for _ in range(num_res_blocks):
        x_temp = x
        x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Add()([x, x_temp])

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(4, activation="softmax")(x) 

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = resnet_model()

model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])

history = model.fit(images, labels, epochs=15, batch_size=32)

plt.plot(history.history['accuracy'])
plt.title('Model Training Progress')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

model.summary()

tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True)
