# 3D GAN 프로젝트

이 프로젝트는 3D GAN (Generative Adversarial Network)을 사용하여 3D 데이터를 생성하는 모델을 구축합니다. 이 README 파일은 프로젝트의 개요, 설치 방법, 사용 방법을 설명합니다.

## 프로젝트 구성

- `build_discriminator.py`: 디스크리미네이터 모델을 구축하는 코드
- `train.py`: 모델을 훈련시키는 코드
- `requirements.txt`: 필요한 패키지 목록
- `voxel_data.npy`: 훈련에 사용될 3D 데이터 (가상의 파일로 예시입니다)

## 요구 사항

- Python 3.7 이상
- TensorFlow 2.14.1
- NumPy 1.21.0
- Matplotlib 3.4.2
- Pillow 8.2.0
- Gensim 4.3.0
- FuzzyTM 0.4.0 이상

## 설치 방법

1. 가상 환경을 만듭니다 (권장).

    ```bash
    python -m venv myenv
    ```

2. 가상 환경을 활성화합니다.

    - Windows:

        ```bash
        myenv\Scripts\activate
        ```

    - Mac/Linux:

        ```bash
        source myenv/bin/activate
        ```

3. 필요한 패키지를 설치합니다.

    ```bash
    pip install -r requirements.txt
    ```

## 실행 방법

1. 디스크리미네이터 모델 구축:

    `build_discriminator.py` 파일을 실행하여 디스크리미네이터 모델을 구축합니다.

    ```python
    import tensorflow as tf
    from tensorflow.keras import layers

    def build_discriminator():
        model = tf.keras.Sequential()
        model.add(layers.Conv3D(64, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='same', input_shape=(32, 32, 32, 1)))
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Conv3D(128, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        
        return model

    if __name__ == "__main__":
        discriminator = build_discriminator()
        discriminator.summary()
    ```

2. 모델 훈련:

    `train.py` 파일을 실행하여 모델을 훈련시킵니다. `train.py` 파일은 다음과 같이 작성될 수 있습니다.

    ```python
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers
    from build_discriminator import build_discriminator

    def build_generator():
        model = tf.keras.Sequential()
        model.add(layers.Dense(256 * 4 * 4 * 4, activation="relu", input_shape=(100,)))
        model.add(layers.Reshape((4, 4, 4, 256)))
        
        model.add(layers.Conv3DTranspose(128, (4, 4, 4), strides=(2, 2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Conv3DTranspose(64, (4, 4, 4), strides=(2, 2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Conv3DTranspose(1, (4, 4, 4), strides=(2, 2, 2), padding='same', activation='tanh'))
        
        return model

    def train():
        discriminator = build_discriminator()
        generator = build_generator()
        
        # Load data
        voxel_data = np.load('voxel_data.npy')
        
        # Preprocess data
        voxel_data = (voxel_data - 0.5) * 2
        
        # Compile models
        discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        generator.compile(loss='binary_crossentropy', optimizer='adam')
        
        # Training parameters
        batch_size = 64
        epochs = 10000
        half_batch = batch_size // 2
        
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, voxel_data.shape[0], half_batch)
            real_voxels = voxel_data[idx]
            noise = np.random.normal(0, 1, (half_batch, 100))
            gen_voxels = generator.predict(noise)
            
            d_loss_real = discriminator.train_on_batch(real_voxels, np.ones((half_batch, 1)))
            d_loss_fake = discriminator.train_on_batch(gen_voxels, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, 100))
            valid_y = np.array([1] * batch_size)
            g_loss = generator.train_on_batch(noise, valid_y)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {d_loss[1]*100}%] [G loss: {g_loss}]")

    if __name__ == "__main__":
        train()
    ```

## 데이터 준비

`voxel_data.npy` 파일은 훈련에 사용될 3D 데이터를 포함합니다. 이 파일을 프로젝트 디렉토리에 추가해야 합니다.

## 기여

기여를 원하시면 포크를 하고 풀 리퀘스트를 제출해 주세요. 버그 보고 및 기능 요청은 이슈 트래커를 이용해 주세요.
