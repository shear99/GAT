# KeypointDataset.py
import os
import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np

class KeypointDataset(Dataset):
    def __init__(self, json_dir):
        self.json_dir = json_dir
        self.file_list = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.data_list = []
        self.load_data()

    def load_data(self):
        for file_name in self.file_list:
            file_path = os.path.join(self.json_dir, file_name)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                poses = json_data.get('pose_estimation_info', [])
                for pose in poses:
                    keypoints = pose.get('keypoints', [])
                    bbox = pose['bbox']
                    kp_array = self.process_keypoints(keypoints, bbox)
                    data = self.create_data_object(kp_array)
                    self.data_list.append(data)

    def process_keypoints(self, keypoints, bbox):
        # 바운딩 박스에서 x_min, y_min, x_max, y_max 추출
        x_min = bbox['top_left']['x']
        y_min = bbox['top_left']['y']
        x_max = bbox['bottom_right']['x']
        y_max = bbox['bottom_right']['y']

        width = x_max - x_min
        height = y_max - y_min

        kp_coords = []
        for kp in keypoints:
            x = kp['coordinates']['x']
            y = kp['coordinates']['y']
            # Normalize keypoints to [0,1] within the bounding box
            x_norm = (x - x_min) / width
            y_norm = (y - y_min) / height
            kp_coords.append([x_norm, y_norm])
        kp_array = np.array(kp_coords)
        return kp_array

    def create_edge_index(self):
        # 스켈레톤 연결 (0-based indexing)
        edge_index = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
            [7, 9], [8, 10], [3, 5], [4, 6],
            [0, 1], [0, 2], [1, 2], [1, 3], [2, 4]
            # 추가된 엣지: [0,1], [0,2], [1,2], [1,3], [2,4]
        ]
        # edge_index를 텐서로 변환
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index

    def create_skeleton_for_features(self):
        # 각도 계산에 사용할 스켈레톤
        skeleton_for_features = [
            [15,13], [13,11], [16,14], [14,12], [11,12],
            [5,11], [6,12], [5,6], [5,7], [6,8],
            [7,9], [8,10], [3,5], [4,6],
            # 제외된 엣지: [0,1], [0,2], [1,2], [1,3], [2,4]
            # 추가된 엣지 (학습에만 사용): [4,10], [3,9], [6,10]
            [4,10], [3,9], [6,10]
        ]
        skeleton_for_features = torch.tensor(skeleton_for_features, dtype=torch.long)
        return skeleton_for_features

    def create_data_object(self, kp_array):
        x = torch.zeros((17, 2), dtype=torch.float)
        num_keypoints = kp_array.shape[0]
        x[:num_keypoints] = torch.tensor(kp_array, dtype=torch.float)
        # Define edge_index for GAT message passing
        edge_index = self.create_edge_index()
        # Skeleton for angle computation
        skeleton_for_features = self.create_skeleton_for_features()
        data = self.create_data_object_with_features(x, edge_index, skeleton_for_features)
        return data

    def create_data_object_with_features(self, x, edge_index, skeleton_for_features):
        num_nodes = x.size(0)

        # 각 노드별 연결 정보 생성
        adjacency_list = [[] for _ in range(num_nodes)]
        for edge in skeleton_for_features:
            i, j = edge
            adjacency_list[i].append(j)
            adjacency_list[j].append(i)  # 무방향 그래프 가정

        # 각 노드별로 단위 벡터 계산
        features = []
        max_connections = 4  # 노드당 최대 연결 수 (필요에 따라 조정)
        for i in range(num_nodes):
            connected_nodes = adjacency_list[i]
            unit_vectors = []
            for j in connected_nodes:
                if x[j].sum() == 0 or x[i].sum() == 0:
                    # 키포인트가 누락된 경우
                    unit_vec = torch.zeros(2)
                else:
                    vec = x[j] - x[i]
                    norm = torch.norm(vec)
                    if norm != 0:
                        unit_vec = vec / norm
                    else:
                        unit_vec = torch.zeros_like(vec)
                unit_vectors.append(unit_vec)
            # 최대 연결 수에 맞게 패딩 또는 자르기
            if len(unit_vectors) < max_connections:
                unit_vectors += [torch.zeros(2) for _ in range(max_connections - len(unit_vectors))]
            else:
                unit_vectors = unit_vectors[:max_connections]
            # 단위 벡터를 펼치기
            unit_vectors_flat = torch.cat(unit_vectors)
            # 노드 좌표와 단위 벡터를 결합
            node_feature = torch.cat([x[i], unit_vectors_flat])
            features.append(node_feature)
        x = torch.stack(features)  # Shape: [num_nodes, feature_size]

        data = Data(x=x, edge_index=edge_index)
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
