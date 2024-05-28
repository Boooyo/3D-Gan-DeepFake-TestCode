import tensorflow as tf
from tensorflow.keras import layers

def build_discriminator():
    input_shape = (32, 32, 32, 1)
    alpha = 0.2  # LeakyReLU의 음수 슬로프 계수
    
    model = tf.keras.Sequential(name='Discriminator')
    
    # 첫 번째 Conv3D 레이어
    model.add(layers.Conv3D(64, (4, 4, 4), strides=(2, 2, 2), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(alpha=alpha))
    
    # 두 번째 Conv3D 레이어
    model.add(layers.Conv3D(128, (4, 4, 4), strides=(2, 2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=alpha))
    
    # 평탄화 레이어 및 출력 레이어
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

# 디스크리미네이터 모델 인스턴스화 및 구조 출력
discriminator = build_discriminator()
discriminator.summary()
