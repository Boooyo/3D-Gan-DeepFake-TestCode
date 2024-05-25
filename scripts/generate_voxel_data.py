import numpy as np

def generate_dummy_voxel_data(num_samples=100, voxel_dim=32):
    """
    더미 3D 복셀 데이터를 생성합니다.

    Args:
    num_samples (int): 생성할 데이터 샘플 수
    voxel_dim (int): 각 샘플의 복셀 차원 (voxel_dim x voxel_dim x voxel_dim)

    Returns:
    np.ndarray: 생성된 더미 복셀 데이터
    """
    voxel_data = np.random.rand(num_samples, voxel_dim, voxel_dim, voxel_dim)
    voxel_data = (voxel_data > 0.5).astype(np.float32)  # 이진화
    return voxel_data

# 더미 데이터 생성
voxel_data = generate_dummy_voxel_data()

# 데이터 저장
np.save('data/voxel_data.npy', voxel_data)

print(f"생성된 더미 데이터의 형태: {voxel_data.shape}")
