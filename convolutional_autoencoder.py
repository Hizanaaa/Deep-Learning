import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from keras.datasets import mnist


# Load Dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize images
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

# Add channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("Training Shape:", x_train.shape)
print("Testing Shape :", x_test.shape)


# Define Autoencoder

class ConvAutoencoder(Model):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = tf.keras.Sequential([

            layers.Input(shape=(28,28,1)),
            layers.Conv2D(
                filters=32,
                kernel_size=(3,3),
                activation='relu',
                padding='same'
            ),
            layers.MaxPooling2D((2,2), padding='same'),
            layers.Conv2D(
                filters=16,
                kernel_size=(3,3),
                activation='relu',
                padding='same'
            ),
            layers.MaxPooling2D((2,2), padding='same')

        ])

        # Decoder
        self.decoder = tf.keras.Sequential([

            layers.Conv2DTranspose(
                filters=16,
                kernel_size=3,
                strides=2,
                activation='relu',
                padding='same'
            ),
            layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=2,
                activation='relu',
                padding='same'
            ),
            layers.Conv2D(
                filters=1,
                kernel_size=3,
                activation='sigmoid',
                padding='same'
            )
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Build Model

autoencoder = ConvAutoencoder()
autoencoder.compile(
    optimizer='adam',
    loss=losses.BinaryCrossentropy()
)

# Train
history = autoencoder.fit(

    x_train,
    x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test,x_test)

)

# Encode and Decode
encoded_imgs = autoencoder.encoder.predict(x_test)
decoded_imgs = autoencoder.decoder.predict(encoded_imgs)

# Display Results
n = 6
plt.figure(figsize=(12,6))

for i in range(n):

    # Original
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title("Original")
    plt.axis("off")

    # Reconstructed
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis("off")

plt.show()