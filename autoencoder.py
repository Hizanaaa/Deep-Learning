'''
        MNIST Image
        (28×28)
            ↓
        Flatten
        784
            ↓
        Encoder Dense
        784 → 64
            ↓
        Latent Space
        64
            ↓
        Decoder Dense
        64 → 784
            ↓
        Reshape
        28×28
            ↓
        Reconstructed Image
'''

'''
numpy for numerical operations, array manipulation, 
matrix operations and data handling
'''
import numpy as np


'''
matplotlib for data visualization
used later for visualizing the original and reconstructed images from 
the autoencoder
'''
import matplotlib.pyplot as plt

'''
Imports TensorFlow, the deep learning framework.

TensorFlow provides:
- Neural networks
- Automatic differentiation
- GPU support
- Model training
'''
import tensorflow as tf

'''
Imports Keras from TensorFlow.
Keras is a high-level API for building neural networks.
'''
from tensorflow import keras

'''
Imports commonly used components.

layers: Contains neural network layers:
    >>> layers.Dense()
    >>> layers.Conv2D()
    >>> layers.Flatten()

losses: Contains loss functions:
    >>> losses.MeanSquaredError()
'''
from tensorflow.keras import layers, losses

'''
Imports the base Model class.

Your autoencoder will inherit from this class.
    >>> class SimpleAutoencoder(Model):
This gives access to:
    >>> compile()
    >>> fit()
    >>> predict()
'''
from tensorflow.keras.models import Model

'''
Imports the MNIST dataset.

MNIST contains:
    60,000 training images
    10,000 testing images
    Handwritten digits (0–9)

Each image: 28 × 28 pixels
'''
from keras.datasets import mnist


# LOAD DATASET 

'''
Normally: 
    >>> (x_train, y_train), (x_test, y_test)
But autoencoders dont need labels, 
    so '_' means ignore that value
'''
(x_train, _), (x_test, _) = mnist.load_data()

# NORMALISE DATA 

'''
Original Pixel values: 0 - 255
Neural networks train better when value are small. 
so it converts, 
    0 → 0.0
    255 → 1.0
'''
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# RESHAPE IMAGES

'''
Originally : (60000, 28,28)
tensorflow expects: (samples, height, width, channels)
since its grayscale: Channels --> 1 
Result : 
    >>> for train :(60000, 28, 28, 1)
    >>> for test : (10000, 28, 28, 1)
'''
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


# AUTOENCODER CLASS DEFINITION

'''
autoencoder has 2 parts : Encoder and Decoder
'''
class SimpleAutoencoder(Model):
    '''
    Constructor function runs when 
        >>> autoencoder = SimpleAutoencoder(64) 
    '''
    def __init__(self, latent_dimensions):
        '''
        Calls the parent Model constructor.
        Without this, TensorFlow's internal model setup won't happen.
        '''
        super(SimpleAutoencoder, self).__init__()

        # ENCODER 
        '''
        Creates encoder network.
        Encoder compresses image into a smaller representation

        >>> layers.Input(shape=(28, 28, 1)) 
            defines the input shape 
        >>> layers.Flatten(),
            converts 28 x 28 x 1 into 784

            [[1,2],
             [3,4]]

            becomes [1,2,3,4]
        >>> layers.Dense(latent_dimensions, activation='relu'),
            Compresses 784 numbers into 64 numbers.
            if 
                latent_dimensions = 64
            then 
                784 --> 64 
            
            This compressed representation is called latent vector or latent space
        '''
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(latent_dimensions, activation='relu'),
        ])
        
        # DECODER

        '''
            >>> self.decoder = tf.keras.Sequential([
                Creates decoder network.
                Decoder reconstructs image from compressed data.
            >>> layers.Dense(28 * 28, activation='sigmoid'),
                Expands 64 --> 784
                sigmoid outputs: 0 to 1  
                which matches normalised pixel values 
            >>> layers.Reshape((28, 28, 1))
                converts 784 to 28 x 28 x 1 image format
        '''
        self.decoder = tf.keras.Sequential([
            layers.Dense(28 * 28, activation='sigmoid'),
            layers.Reshape((28, 28, 1))
        ])
    
    # FORWARD PASS

    '''
        >>> layers.Reshape((28, 28, 1))
            Defines how data flows through network.
        >>> encoded = self.encoder(input_data)
            compress image: 28 x 28 x 1 --> 64 
        >>> decoded = self.decoder(encoded)
            Reconstruct image : 64 → 28×28×1
        >>> return decoded
            Returns reconstructed image.
    '''
    def call(self, input_data):
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)
        return decoded


# CREATE MODEL

'''
sets compression size
    original : 784 values
    compressed: 64 values
    compression ratio: 784/64 = 12.25
'''
latent_dimensions = 64

'''
creates the model object
'''
autoencoder = SimpleAutoencoder(latent_dimensions)

# COMPILE MODEL 

'''
configures the training
    >>> optimizer='adam' 
        updates weights efficiently : most common optimizer 
    >>> loss=losses.MeanSquaredError()
        measures reconstruction error
            MSE = 1/n E( y - y^) ^ 2 
        smaller MSE = better reconstruction 

'''
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

# TRAIN MODEL 

'''
    >>> autoencoder.fit(
            x_train, --> input 
            x_train, --> target
        autoencoder learns : Input --> same output
    >>> epochs = 10 
        entire dataset is seen 10 times 
    >>> batch_size = 256
        Processes 256 images before updating weights.
    >>> shuffle = True 
        Randomizes image order and Improves learning.
    >>> validation_data=(x_test, x_test)
        Measures performance on unseen images.
'''
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# ENCODE TEST IMAGES 
'''
Obtains compressed 64-dimensional vectors : Shape (10000, 64)
'''
encoded_imgs = autoencoder.encoder(x_test).numpy()

#DECODE
'''
Reconstruct images from latent vectors. : Shape (10000, 28,28, 1)
'''
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


# VISUALIZATION 

''' Display 6 examples '''
n = 6

''' Creates figure Window'''
plt.figure(figsize=(12, 6))

''' Loop through the first 6 images '''
for i in range(n):
    '''
        >>> ax = plt.subplot(2, n, i + 1)
            Creates position in first row
        >>> plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
            Diaplays original digit
        >>> plt.title("Original")
            Adds title
        >>> plt.axis('off')
            Hides axes 
    '''
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    '''
        >>> ax = plt.subplot(2, n, i + 1 + n)
            Creates second-row position.
        >>> plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
            Displays reconstructed image.
        >>> plt.title("Reconstructed")
            Adds title
        >>> plt.axis('off')
            Hides axes 
    '''
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

# SHOW FIGURE
plt.show()

