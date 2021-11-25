import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, Conv2DTranspose, UpSampling2D, Flatten
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(128, 128, 3)),
            layers.Conv2D(filters=128, kernel_size=(5, 5), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2D(filters=512, kernel_size=(5, 5), activation='relu', strides=(2, 2), padding='same'),
            Flatten(),
            Dense(128)
        ])

        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=128),
            layers.Dense(8 * 8 * 512),
            layers.Reshape(target_shape=(8, 8, 512)),
            layers.Conv2DTranspose(filters=256, kernel_size=(5, 5), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(filters=256, kernel_size=(5, 5), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(filters=3, kernel_size=(5, 5), activation='linear', strides=(2, 2), padding='same'),
        ])

    def __call__(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


Autoencoder = Denoise()
