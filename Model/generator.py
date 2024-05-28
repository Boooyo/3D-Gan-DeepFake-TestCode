import tensorflow as tf
from tensorflow.keras import layers

def build_generator():
    noise_dim = 100
    init_shape = (4, 4, 4, 128)  # Dense 레이어 이후 초기 모양

    model = tf.keras.Sequential(name='Generator')
    
    # 완전 연결층: 입력 노이즈를 투영하고 재구성
    model.add(layers.Dense(tf.math.reduce_prod(init_shape), activation="relu", input_dim=noise_dim))
    model.add(layers.Reshape(init_shape))

    # Deconvolution (Conv3DTranspose) 레이어
    filters = [128, 64, 32, 1]
    kernel_size = (4, 4, 4)
    strides = (2, 2, 2)
    padding = 'same'
    
    for i in range(len(filters) - 1):
        model.add(layers.Conv3DTranspose(filters[i], kernel_size, strides=strides, padding=padding))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
    
    # 출력 레이어
    model.add(layers.Conv3DTranspose(filters[-1], kernel_size, strides=strides, padding=padding, activation='tanh'))
    
    return model

# 생성기 모델 인스턴스화 및 구조 출력
generator = build_generator()
generator.summary()
