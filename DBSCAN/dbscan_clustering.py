import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from KeypointDatasetForClustering import KeypointDatasetForClustering
import matplotlib.pyplot as plt

# 가중치를 조정하기 위해 keypoint 인덱스 정의
arm_leg_indices = [7, 8, 9, 10, 15, 16]  # 팔과 다리 부분
torso_thigh_indices = [5, 6, 11, 12, 13, 14]  # 몸통과 허벅지 부분
face_indices = [0, 1, 2, 3, 4]  # 얼굴 부분

def apply_weights(data):
    """특정 keypoint에 가중치를 적용하는 함수"""
    num_keypoints = data.shape[1] // 2
    weights = np.ones(num_keypoints)

    # 가중치 설정
    weights[arm_leg_indices] = 1.5  # 팔과 다리 부분: 높은 가중치
    weights[torso_thigh_indices] = 1.0  # 몸통과 허벅지 부분: 중간 가중치
    weights[face_indices] = 0.5  # 얼굴 부분: 낮은 가중치

    # x, y 좌표에 동일한 가중치 적용
    weighted_data = data.copy()
    for i in range(num_keypoints):
        weighted_data[:, i * 2] *= weights[i]
        weighted_data[:, i * 2 + 1] *= weights[i]

    return weighted_data

def main():
    # 데이터셋 로드
    json_dir = '../3/json'
    dataset = KeypointDatasetForClustering(json_dir=json_dir)

    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    data = np.stack(dataset.data_list)

    # 가중치 적용
    data_weighted = apply_weights(data)

    # MinMaxScaler 사용
    print("\n=== Applying MinMaxScaler ===")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_weighted)

    # 차원 축소 (PCA)
    pca = PCA(n_components=10)
    data_pca = pca.fit_transform(data_scaled)

    best_eps = None
    best_min_samples = None
    best_score = -1
    best_labels = None

    eps_values = np.arange(0.5, 2.0, 0.1)
    min_samples_values = range(2, 10)

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data_pca)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            if n_clusters <= 1:
                continue

            try:
                score = silhouette_score(data_pca, labels)
            except:
                continue

            if score > best_score:
                best_eps = eps
                best_min_samples = min_samples
                best_score = score
                best_labels = labels

    if best_eps is None:
        print("No suitable clustering found.")
        return

    print(f"Best eps: {best_eps}")
    print(f"Best min_samples: {best_min_samples}")
    print(f"Best silhouette score: {best_score}")

    # 최적의 DBSCAN 모델로 최종 클러스터링 수행
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    labels = dbscan.fit_predict(data_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f'Number of clusters: {n_clusters}')
    print(f'Number of noise points: {n_noise}')

    # PCA 2D 시각화
    pca_2d = PCA(n_components=2)
    data_2d = pca_2d.fit_transform(data_pca)

    plt.figure(figsize=(10, 7))
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black for noise
        class_member_mask = (labels == k)
        xy = data_2d[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=f'Cluster {k}' if k != -1 else 'Noise', edgecolors='k', s=50)

    plt.title('DBSCAN Clustering Result (MinMaxScaler)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig("dbscan_clustering_result_MinMaxScaler.png")
    print("MinMaxScaler result saved as dbscan_clustering_result_MinMaxScaler.png")

if __name__ == '__main__':
    main()
