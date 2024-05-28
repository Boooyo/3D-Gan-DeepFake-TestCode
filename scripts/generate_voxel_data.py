import numpy as np
import os

def generate_dummy_voxel_data(num_samples=100, voxel_dim=32):
    """
    더미 3D 복셀 데이터를 생성합니다.

    Args:
    num_samples (int): 생성할 데이터 샘플 수
    voxel_dim (int): 각 샘플의 복셀 차원 (voxel_dim x voxel_dim x voxel_dim)

    Returns:
    np.ndarray: 생성된 더미 복셀 데이터
    """
    # 무작위 데이터 생성 및 이진화
    voxel_data = np.random.rand(num_samples, voxel_dim, voxel_dim, voxel_dim)
    voxel_data = (voxel_data > 0.5).astype(np.float32)  # 0과 1로 이진화
    return voxel_data

def save_voxel_data(data, file_path):
    """
    복셀 데이터를 파일로 저장합니다.

    Args:
    data (np.ndarray): 저장할 복셀 데이터
    file_path (str): 데이터가 저장될 파일 경로
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 디렉토리가 없으면 생성
    np.save(file_path, data)

# 더미 데이터 생성
voxel_data = generate_dummy_voxel_data()

# 데이터 저장 경로 설정
save_path = 'data/voxel_data.npy'

# 데이터 저장
save_voxel_data(voxel_data, save_path)

print(f"생성된 더미 데이터의 형태: {voxel_data.shape}")
print(f"더미 데이터가 '{save_path}'에 저장되었습니다.")
