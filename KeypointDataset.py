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
                    if len(keypoints) != 17:
                        # 키포인트가 17개가 아닌 경우 패스
                        print(f"Skipping {file_name}: Keypoints count is {len(keypoints)}, expected 17.")
                        continue
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

    def create_data_object(self, kp_array):
        x = torch.tensor(kp_array, dtype=torch.float)
        # Define edge_index
        edge_index = self.create_edge_index()
        data = Data(x=x, edge_index=edge_index)
        return data

    def create_edge_index(self):
        # COCO skeleton connections (converted to 0-based indexing)
        skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
            [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
            [1, 3], [2, 4], [3, 5], [4, 6]
        ]
        # Convert to edge_index tensor
        edge_index = torch.tensor(skeleton, dtype=torch.long).t().contiguous()
        return edge_index

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
