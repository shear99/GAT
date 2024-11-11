import os
import json
import numpy as np
from torch.utils.data import Dataset

class KeypointDatasetForClustering(Dataset):
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
                    bbox = pose.get('bbox')
                    kp_array = self.process_keypoints(keypoints, bbox)
                    if kp_array is not None:
                        self.data_list.append(kp_array)

    def process_keypoints(self, keypoints, bbox):
        if not bbox or len(keypoints) != 17:
            return None

        # 바운딩 박스 좌표 추출
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
            if x == 0 and y == 0:
                x_norm, y_norm = 0, 0
            else:
                x_norm = (x - x_min) / width if width != 0 else 0
                y_norm = (y - y_min) / height if height != 0 else 0
            kp_coords.append([x_norm, y_norm])
        
        if len(kp_coords) != 17:
            return None

        kp_array = np.array(kp_coords)  # shape: (17, 2)

        # 관절 벡터 계산
        # 관절 연결 정보 (COCO 키포인트 순서 기준)
        limbs = [
            (5, 7),  # 왼쪽 어깨-팔꿈치
            (7, 9),  # 왼쪽 팔꿈치-손목
            (6, 8),  # 오른쪽 어깨-팔꿈치
            (8, 10), # 오른쪽 팔꿈치-손목
            (11, 13),# 왼쪽 엉덩이-무릎
            (13, 15),# 왼쪽 무릎-발목
            (12, 14),# 오른쪽 엉덩이-무릎
            (14, 16) # 오른쪽 무릎-발목
        ]

        limb_vectors = []
        for (i, j) in limbs:
            x1, y1 = kp_array[i]
            x2, y2 = kp_array[j]
            vec = [x2 - x1, y2 - y1]
            limb_vectors.extend(vec)

        # 키포인트 좌표와 관절 벡터를 하나의 벡터로 결합
        feature_vector = np.concatenate([kp_array.flatten(), limb_vectors])

        return feature_vector

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
