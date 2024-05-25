import tensorflow as tf
from tensorflow.keras import layers

def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv3D(64, (4, 4, 4), strides=(2, 2, 2), padding='same', input_shape=(32, 32, 32, 1)))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv3D(128, (4, 4, 4), strides=(2, 2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model
