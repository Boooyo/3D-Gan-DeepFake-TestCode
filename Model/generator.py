import tensorflow as tf
from tensorflow.keras import layers

def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128 * 4 * 4 * 4, activation="relu", input_dim=100))
    model.add(layers.Reshape((4, 4, 4, 128)))
    
    model.add(layers.Conv3DTranspose(128, (4, 4, 4), strides=(2, 2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv3DTranspose(64, (4, 4, 4), strides=(2, 2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv3DTranspose(32, (4, 4, 4), strides=(2, 2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv3DTranspose(1, (4, 4, 4), strides=(2, 2, 2), padding='same', activation='tanh'))
    
    return model