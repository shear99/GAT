# Evaluator.py
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from GAT import GAT
from KeypointDataset import KeypointDataset

class Evaluator:
    def __init__(self, model_path, json_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 샘플 데이터를 이용하여 in_channels 결정
        sample_dataset = KeypointDataset(json_dir=json_dir)
        sample_data = sample_dataset[0]
        in_channels = sample_data.x.size(1)

        # 모델 초기화
        self.model = GAT(in_channels=in_channels, hidden_channels=8, out_channels=in_channels, num_heads=8).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def evaluate(self, data):
        # data는 torch_geometric.data.Data 객체입니다.
        data = data.to(self.device)
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            # 좌표 부분만을 사용하여 유사도 계산
            valid_mask = (data.x[:, :2].sum(dim=1) != 0)
            if valid_mask.sum() > 0:
                similarity = F.cosine_similarity(out[valid_mask, :2], data.x[valid_mask, :2], dim=1).mean().item()
            else:
                similarity = 0.0
        return similarity, out.cpu().numpy()  # 시각화를 위해 예측된 특징 벡터를 반환합니다

def visualize_pose(json_data, predicted_features, similarity_score, filename):
    # JSON에서 이미지의 가로와 세로 크기를 가져옵니다.
    image_width = json_data['image_width']
    image_height = json_data['image_height']

    # 이미지 크기에 맞는 빈 그림을 생성합니다.
    fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=100)

    # 플롯의 한계를 이미지 크기에 맞춥니다.
    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)
    ax.invert_yaxis()  # y축을 반전하여 이미지 좌표계와 일치시킵니다.

    # 바운딩 박스와 키포인트를 추출합니다.
    pose_info = json_data['pose_estimation_info'][0]
    bbox = pose_info['bbox']
    keypoints = pose_info['keypoints']

    # 원본 키포인트 좌표
    original_keypoints = []
    for kp in keypoints:
        x = kp['coordinates']['x']
        y = kp['coordinates']['y']
        original_keypoints.append([x, y])
    original_keypoints = np.array(original_keypoints)

    # 예측된 키포인트 좌표 (재스케일링 필요)
    # 바운딩 박스 좌표 추출
    x_min = bbox['top_left']['x']
    y_min = bbox['top_left']['y']
    x_max = bbox['bottom_right']['x']
    y_max = bbox['bottom_right']['y']

    width_bbox = x_max - x_min
    height_bbox = y_max - y_min

    # 예측된 특징 벡터에서 좌표 부분만 추출
    predicted_keypoints = predicted_features[:, :2]

    # 좌표를 원본 이미지 크기에 맞게 스케일링
    predicted_keypoints_rescaled = predicted_keypoints.copy()
    predicted_keypoints_rescaled[:, 0] = predicted_keypoints_rescaled[:, 0] * width_bbox + x_min
    predicted_keypoints_rescaled[:, 1] = predicted_keypoints_rescaled[:, 1] * height_bbox + y_min

    # 원본 및 예측된 키포인트와 바운딩 박스를 그립니다.
    plot_and_save(ax, original_keypoints, predicted_keypoints_rescaled, x_min, y_min, width_bbox, height_bbox, similarity_score, filename, image_width, image_height)

def plot_and_save(ax, original_keypoints, predicted_keypoints, x_min, y_min, width_bbox, height_bbox, similarity_score, filename, image_width, image_height):
    # 디렉토리가 존재하지 않으면 생성합니다.
    os.makedirs("5/original", exist_ok=True)
    os.makedirs("5/eval", exist_ok=True)

    # 스켈레톤 연결 정보
    skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
        [7, 9], [8, 10], [3, 5], [4, 6],
        [0, 1], [0, 2], [1, 2], [1, 3], [2, 4]
    ]

    # 바운딩 박스 그리기
    rect = patches.Rectangle((x_min, y_min), width_bbox, height_bbox, linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # 원본 키포인트와 스켈레톤 그리기
    for edge in skeleton:
        if edge[0] < len(original_keypoints) and edge[1] < len(original_keypoints):
            x_coords = [original_keypoints[edge[0], 0], original_keypoints[edge[1], 0]]
            y_coords = [original_keypoints[edge[0], 1], original_keypoints[edge[1], 1]]
            ax.plot(x_coords, y_coords, 'b-', linewidth=2)
    ax.scatter(original_keypoints[:, 0], original_keypoints[:, 1], c='blue', s=50)
    ax.set_title("Original Keypoints")
    ax.axis('off')
    original_path = os.path.join("5/original", f"{filename}_original.png")
    plt.savefig(original_path, bbox_inches='tight', pad_inches=0)
    print(f"Original plot saved as {original_path}")
    ax.cla()  # 다음 플롯을 위해 축을 초기화합니다.

    # 예측된 키포인트와 스켈레톤 그리기
    # 동일한 축을 재사용하여 크기를 일관되게 유지합니다.
    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)
    ax.invert_yaxis()
    ax.add_patch(rect)
    for edge in skeleton:
        if edge[0] < len(predicted_keypoints) and edge[1] < len(predicted_keypoints):
            x_coords = [predicted_keypoints[edge[0], 0], predicted_keypoints[edge[1], 0]]
            y_coords = [predicted_keypoints[edge[0], 1], predicted_keypoints[edge[1], 1]]
            ax.plot(x_coords, y_coords, 'r--', linewidth=2)
    ax.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], c='red', s=50)
    ax.set_title(f"Predicted Keypoints (Similarity: {similarity_score:.4f})")
    ax.axis('off')
    eval_path = os.path.join("5/eval", f"{filename}_eval.png")
    plt.savefig(eval_path, bbox_inches='tight', pad_inches=0)
    print(f"Evaluation plot saved as {eval_path}")
    plt.close()

if __name__ == '__main__':
    # 테스트 JSON 폴더 경로 정의
    test_json_dir = './test_img_5'

    # 데이터셋 및 평가자 초기화
    dataset = KeypointDataset(json_dir=test_json_dir)
    evaluator = Evaluator(model_path='gat_model.pth', json_dir=test_json_dir)

    # 테스트 폴더에서 JSON 파일 목록 가져오기
    json_files = [f for f in os.listdir(test_json_dir) if f.endswith('.json')]

    # 데이터셋의 각 샘플을 평가하고 시각화
    for i, (sample_data, json_file) in enumerate(zip(dataset, json_files)):
        similarity_score, predicted_features = evaluator.evaluate(sample_data)

        # 원본 JSON 파일명을 기반으로 파일명 정의
        filename = os.path.splitext(json_file)[0]

        # 원본 JSON 파일의 전체 경로
        original_json_path = os.path.join(test_json_dir, json_file)

        # JSON 데이터 로드
        with open(original_json_path, 'r') as f:
            json_data = json.load(f)

        # 결과를 시각화하고 저장
        visualize_pose(json_data, predicted_features, similarity_score, filename)
