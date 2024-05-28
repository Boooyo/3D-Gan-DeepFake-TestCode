import numpy as np
import tensorflow as tf
from models.generator import build_generator
from models.discriminator import build_discriminator
from scripts.utils import load_voxel_data, save_voxel_images

# 모델 컴파일 및 설정
def compile_models():
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

    generator = build_generator()
    discriminator.trainable = False

    z = tf.keras.Input(shape=(100,))
    voxel = generator(z)
    valid = discriminator(voxel)

    combined = tf.keras.Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
    
    return generator, discriminator, combined

generator, discriminator, combined = compile_models()

# 데이터 로드
def load_data(file_path):
    try:
        voxel_data = load_voxel_data(file_path)
        print(f"데이터 로드 완료: {voxel_data.shape}")
        return voxel_data
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return None

voxel_data = load_data('data/voxel_data.npy')

# 학습 함수
def train(epochs, batch_size=128, save_interval=50):
    if voxel_data is None:
        print("학습할 데이터가 없습니다.")
        return
    
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # 실제 데이터 가져오기
        idx = np.random.randint(0, voxel_data.shape[0], batch_size)
        real_voxels = voxel_data[idx]
        
        # 가짜 데이터 생성
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_voxels = generator.predict(noise)
        
        # 디스크리미네이터 학습
        d_loss_real = discriminator.train_on_batch(real_voxels, real_labels)
        d_loss_fake = discriminator.train_on_batch(gen_voxels, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # 생성기 학습
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = combined.train_on_batch(noise, real_labels)
        
        # 진행 상황 출력 및 이미지 저장
        if epoch % save_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")
            save_voxel_images(epoch, gen_voxels)

# 모델 학습
train(epochs=10000, batch_size=32, save_interval=200)
