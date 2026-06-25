import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from keras.datasets import mnist


# Load Dataset
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Add Gaussian Noise
noise_factor = 0.4

x_train_noisy = x_train + noise_factor * np.random.normal(
    loc=0.0,
    scale=1.0,
    size=x_train.shape
)

x_test_noisy = x_test + noise_factor * np.random.normal(
    loc=0.0,
    scale=1.0,
    size=x_test.shape
)

# Keep pixel values between 0 and 1
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Define Denoising Autoencoder

class DenoisingAutoencoder(Model):
    def __init__(self):
        super().__init__()
        self.encoder = tf.keras.Sequential([

            layers.Input(shape=(28,28,1)),
            layers.Conv2D(
                32,
                3,
                activation='relu',
                padding='same'
            ),
            layers.MaxPooling2D(
                2,
                padding='same'
            ),
            layers.Conv2D(
                16,
                3,
                activation='relu',
                padding='same'
            ),
            layers.MaxPooling2D(
                2,
                padding='same'
            )
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(
                16,
                3,
                strides=2,
                activation='relu',
                padding='same'
            ),
            layers.Conv2DTranspose(
                32,
                3,
                strides=2,
                activation='relu',
                padding='same'
            ),
            layers.Conv2D(
                1,
                3,
                activation='sigmoid',
                padding='same'
            )
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Create Model
autoencoder = DenoisingAutoencoder()
autoencoder.compile(
    optimizer='adam',
    loss=losses.BinaryCrossentropy()
)

# Train
history = autoencoder.fit(
    x_train_noisy,
    x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)


# Predict
decoded_imgs = autoencoder.predict(x_test_noisy)

# Plot
n = 6
plt.figure(figsize=(15,7))

for i in range(n):

    # Noisy Input
    ax = plt.subplot(3,n,i+1)
    plt.imshow(
        x_test_noisy[i].reshape(28,28),
        cmap='gray'
    )

    plt.title("Noisy")
    plt.axis("off")

    # Reconstructed
    ax = plt.subplot(3,n,i+1+n)
    plt.imshow(
        decoded_imgs[i].reshape(28,28),
        cmap='gray'
    )

    plt.title("Denoised")
    plt.axis("off")

    # Original
    ax = plt.subplot(3,n,i+1+2*n)
    plt.imshow(
        x_test[i].reshape(28,28),
        cmap='gray'
    )
    plt.title("Original")
    plt.axis("off")

plt.show()