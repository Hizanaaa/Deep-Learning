import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from keras.datasets import mnist

# =====================================================
# Load MNIST Dataset
# =====================================================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images (0-255 --> 0-1)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
# (60000,28,28) -> (60000,28,28,1)

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Convert labels into One-Hot Encoding

num_classes = 10

y_train = tf.keras.utils.to_categorical(
    y_train,
    num_classes
)

y_test = tf.keras.utils.to_categorical(
    y_test,
    num_classes
)

print("Training Images :", x_train.shape)
print("Training Labels :", y_train.shape)

# =====================================================
# Hyperparameters
# =====================================================

image_height = 28
image_width = 28
channels = 1

latent_dim = 16

# =====================================================
# Sampling Layer
# =====================================================

class Sampling(layers.Layer):

    def call(self, inputs):

        z_mean, z_log_var = inputs

        batch_size = tf.shape(z_mean)[0]
        latent_size = tf.shape(z_mean)[1]

        epsilon = tf.random.normal(
            shape=(batch_size, latent_size)
        )

        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        return z

# =====================================================
# Encoder
# =====================================================

# Image Input

image_input = tf.keras.Input(
    shape=(28,28,1),
    name="Image_Input"
)

# Label Input

label_input = tf.keras.Input(
    shape=(num_classes,),
    name="Label_Input"
)

# Convert label vector into an image

label_embedding = layers.Dense(
    28 * 28,
    activation="relu"
)(label_input)

label_embedding = layers.Reshape(
    (28,28,1)
)(label_embedding)

# Merge Image + Label

x = layers.Concatenate()(
    [
        image_input,
        label_embedding
    ]
)

# --------------------------------------
# Convolution Block 1
# --------------------------------------

x = layers.Conv2D(

    filters=32,

    kernel_size=3,

    strides=2,

    padding="same",

    activation="relu"

)(x)

# Output:
# 14 x 14 x 32

# --------------------------------------
# Convolution Block 2
# --------------------------------------

x = layers.Conv2D(

    filters=64,

    kernel_size=3,

    strides=2,

    padding="same",

    activation="relu"

)(x)

# Output:
# 7 x 7 x 64

# --------------------------------------
# Flatten
# --------------------------------------

x = layers.Flatten()(x)

# --------------------------------------
# Dense Layer
# --------------------------------------

x = layers.Dense(

    128,

    activation="relu"

)(x)

# --------------------------------------
# Mean Vector
# --------------------------------------

z_mean = layers.Dense(

    latent_dim,

    name="z_mean"

)(x)

# --------------------------------------
# Log Variance
# --------------------------------------

z_log_var = layers.Dense(

    latent_dim,

    name="z_log_var"

)(x)

# --------------------------------------
# Sample Latent Vector
# --------------------------------------

z = Sampling()(

    [z_mean, z_log_var]

)

# --------------------------------------
# Build Encoder
# --------------------------------------

encoder = tf.keras.Model(

    inputs=[image_input, label_input],

    outputs=[z_mean, z_log_var, z],

    name="Conditional_Encoder"

)

encoder.summary()

# =====================================================
# Decoder
# =====================================================

# Latent Vector Input

latent_inputs = tf.keras.Input(
    shape=(latent_dim,),
    name="Latent_Input"
)

# Label Input

decoder_label_input = tf.keras.Input(
    shape=(num_classes,),
    name="Decoder_Label_Input"
)

# -----------------------------------------------------
# Combine Latent Vector and Label
# -----------------------------------------------------

x = layers.Concatenate()(
    [
        latent_inputs,
        decoder_label_input
    ]
)

# Output shape:
# latent_dim + num_classes
# = 16 + 10 = 26

# -----------------------------------------------------
# Dense Layer
# -----------------------------------------------------

x = layers.Dense(
    7 * 7 * 64,
    activation="relu"
)(x)

# -----------------------------------------------------
# Reshape
# -----------------------------------------------------

x = layers.Reshape(
    (7, 7, 64)
)(x)

# -----------------------------------------------------
# Transposed Convolution 1
# -----------------------------------------------------

x = layers.Conv2DTranspose(

    filters=64,

    kernel_size=3,

    strides=2,

    padding="same",

    activation="relu"

)(x)

# Output:
# 14 × 14 × 64

# -----------------------------------------------------
# Transposed Convolution 2
# -----------------------------------------------------

x = layers.Conv2DTranspose(

    filters=32,

    kernel_size=3,

    strides=2,

    padding="same",

    activation="relu"

)(x)

# Output:
# 28 × 28 × 32

# -----------------------------------------------------
# Final Output Layer
# -----------------------------------------------------

decoder_outputs = layers.Conv2D(

    filters=1,

    kernel_size=3,

    padding="same",

    activation="sigmoid"

)(x)

# Output:
# 28 × 28 × 1

# -----------------------------------------------------
# Build Decoder
# -----------------------------------------------------

decoder = tf.keras.Model(

    inputs=[
        latent_inputs,
        decoder_label_input
    ],

    outputs=decoder_outputs,

    name="Conditional_Decoder"

)

decoder.summary()

# =====================================================
# Conditional Variational Autoencoder
# =====================================================

class CVAE(tf.keras.Model):

    def __init__(self, encoder, decoder):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    # -------------------------------------------------
    # Custom Training Step
    # -------------------------------------------------

    def train_step(self, data):

        images, labels = data

        with tf.GradientTape() as tape:

            # ----------------------------
            # Encode
            # ----------------------------

            z_mean, z_log_var, z = self.encoder(
                [images, labels],
                training=True
            )

            # ----------------------------
            # Decode
            # ----------------------------

            reconstruction = self.decoder(
                [z, labels],
                training=True
            )

            # ----------------------------
            # Reconstruction Loss
            # ----------------------------

            reconstruction_loss = tf.reduce_mean(

                tf.reduce_sum(

                    tf.keras.losses.binary_crossentropy(
                        images,
                        reconstruction
                    ),

                    axis=(1,2)

                )

            )

            # ----------------------------
            # KL Divergence Loss
            # ----------------------------

            kl_loss = -0.5 * tf.reduce_mean(

                tf.reduce_sum(

                    1
                    + z_log_var
                    - tf.square(z_mean)
                    - tf.exp(z_log_var),

                    axis=1

                )

            )

            # ----------------------------
            # Total Loss
            # ----------------------------

            total_loss = reconstruction_loss + kl_loss

        # ----------------------------
        # Compute Gradients
        # ----------------------------

        gradients = tape.gradient(

            total_loss,

            self.trainable_variables

        )

        # ----------------------------
        # Update Weights
        # ----------------------------

        self.optimizer.apply_gradients(

            zip(

                gradients,

                self.trainable_variables

            )

        )

        return {

            "loss": total_loss,

            "reconstruction_loss": reconstruction_loss,

            "kl_loss": kl_loss

        }
# =====================================================
# Create the CVAE Model
# =====================================================

cvae = CVAE(
    encoder=encoder,
    decoder=decoder
)

# =====================================================
# Compile
# =====================================================

cvae.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    )
)

# =====================================================
# Train
# =====================================================

history = cvae.fit(

    x=x_train,

    y=y_train,

    epochs=20,

    batch_size=256,

    shuffle=True

)

# =====================================================
# Function to Generate Any Digit
# =====================================================

def generate_digit(digit):

    # Convert digit into one-hot vector

    label = tf.keras.utils.to_categorical(
        [digit],
        num_classes=10
    )

    # Random latent vector

    z = np.random.normal(
        size=(1, latent_dim)
    )

    # Generate image

    generated = decoder.predict(
        [z, label],
        verbose=0
    )

    plt.figure(figsize=(3,3))

    plt.imshow(
        generated[0].squeeze(),
        cmap="gray"
    )

    plt.title(f"Generated Digit : {digit}")

    plt.axis("off")

    plt.show()