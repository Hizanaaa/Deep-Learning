import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from keras.datasets import mnist

# -------------------------------------
# Load Dataset
# -------------------------------------

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# -------------------------------------
# Parameters
# -------------------------------------

latent_dim = 16

# -------------------------------------
# Sampling Layer
# -------------------------------------

class Sampling(layers.Layer):

    def call(self, inputs):

        z_mean, z_log_var = inputs

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        epsilon = tf.random.normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# -------------------------------------
# Encoder
# -------------------------------------

encoder_inputs = tf.keras.Input(shape=(28,28,1))

x = layers.Conv2D(
    32,
    3,
    activation="relu",
    strides=2,
    padding="same"
)(encoder_inputs)

x = layers.Conv2D(
    64,
    3,
    activation="relu",
    strides=2,
    padding="same"
)(x)

x = layers.Flatten()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

z_mean = layers.Dense(
    latent_dim,
    name="z_mean"
)(x)

z_log_var = layers.Dense(
    latent_dim,
    name="z_log_var"
)(x)

z = Sampling()([z_mean, z_log_var])

encoder = tf.keras.Model(
    encoder_inputs,
    [z_mean, z_log_var, z],
    name="Encoder"
)

encoder.summary()

# -------------------------------------
# Decoder
# -------------------------------------

latent_inputs = tf.keras.Input(shape=(latent_dim,))

x = layers.Dense(
    7*7*64,
    activation="relu"
)(latent_inputs)

x = layers.Reshape((7,7,64))(x)

x = layers.Conv2DTranspose(
    64,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

x = layers.Conv2DTranspose(
    32,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

decoder_outputs = layers.Conv2D(
    1,
    3,
    activation="sigmoid",
    padding="same"
)(x)

decoder = tf.keras.Model(
    latent_inputs,
    decoder_outputs,
    name="Decoder"
)

decoder.summary()

# -------------------------------------
# CVAE Model
# -------------------------------------

class CVAE(tf.keras.Model):

    def __init__(self, encoder, decoder):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):

        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self.encoder(data)

            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(

                tf.reduce_sum(

                    tf.keras.losses.binary_crossentropy(
                        data,
                        reconstruction
                    ),

                    axis=(1,2)

                )

            )

            kl_loss = -0.5 * tf.reduce_mean(

                tf.reduce_sum(

                    1
                    + z_log_var
                    - tf.square(z_mean)
                    - tf.exp(z_log_var),

                    axis=1

                )

            )

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(
            total_loss,
            self.trainable_weights
        )

        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights)
        )

        return {

            "loss": total_loss,

            "reconstruction_loss": reconstruction_loss,

            "kl_loss": kl_loss

        }

# -------------------------------------
# Train
# -------------------------------------

vae = CVAE(
    encoder,
    decoder
)

vae.compile(
    optimizer="adam"
)

vae.fit(
    x_train,
    epochs=20,
    batch_size=256
)

# -------------------------------------
# Generate Images
# -------------------------------------

random_latent_vectors = np.random.normal(
    size=(10, latent_dim)
)

generated = decoder.predict(
    random_latent_vectors
)

plt.figure(figsize=(15,3))

for i in range(10):

    plt.subplot(1,10,i+1)

    plt.imshow(
        generated[i].squeeze(),
        cmap="gray"
    )

    plt.axis("off")

plt.show()