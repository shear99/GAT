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

        # 샘플 데이터를 이용하여 in_channels를 결정합니다.
        sample_dataset = KeypointDataset(json_dir=json_dir)
        sample_data = sample_dataset[0]
        in_channels = sample_data.x.size(1)

        # 모델 초기화 (hidden_channels를 훈련 시와 동일하게 설정)
        self.model = GAT(in_channels=in_channels, hidden_channels=64, out_channels=in_channels, num_heads=8).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def evaluate(self, data):
        # data는 torch_geometric.data.Data 객체입니다.
        data = data.to(self.device)
        with torch.no_grad():
            out = self.model(data.x, data.edge_index, data.edge_weight)
            # 좌표 부분만을 사용하여 유사도를 계산합니다.
            # 누락된 키포인트를 고려하여 마스크를 적용합니다.
            valid_mask = (data.x[:, 0] != 0) | (data.x[:, 1] != 0)
            similarity = F.cosine_similarity(out[valid_mask, :2], data.x[valid_mask, :2], dim=1).mean().item()
        return similarity, out.cpu().numpy()  # 시각화를 위해 예측된 특징 벡터를 반환합니다.


def calculate_keypoint_differences(original_keypoints, predicted_keypoints, image_width, image_height):
    # 키포인트 좌표를 이미지 크기로 정규화
    original_normalized = original_keypoints / np.array([image_width, image_height])
    predicted_normalized = predicted_keypoints / np.array([image_width, image_height])
    
    # 키포인트별 좌표 차이 계산
    differences = original_normalized - predicted_normalized
    # 유클리드 거리 계산
    distances = np.linalg.norm(differences, axis=1)
    # 평균 거리 계산
    mean_distance = np.mean(distances)
    return distances, mean_distance

def save_predicted_json(json_data, predicted_keypoints, filename):
    # 원본 JSON 데이터에서 필요한 정보 추출
    output_data = {
        "image_filename": json_data.get("image_filename", ""),
        "image_height": json_data.get("image_height", 0),
        "image_width": json_data.get("image_width", 0),
        "image_detect_count": json_data.get("image_detect_count", 1),
        "model_name": json_data.get("model_name", ""),
        "device_used": json_data.get("device_used", ""),
        "arguments": json_data.get("arguments", {}),
        "pose_estimation_info": []
    }

    # 예측된 keypoints와 바운딩 박스 정보 구성
    pose_info = {}
    # 예측된 바운딩 박스는 원본과 동일하다고 가정
    pose_info["bbox"] = json_data["pose_estimation_info"][0]["bbox"]

    # 예측된 keypoints 리스트 생성
    keypoints_list = []
    for idx, (x, y) in enumerate(predicted_keypoints):
        keypoint = {
            "index": idx,
            "coordinates": {
                "x": float(x),
                "y": float(y)
            },
            "score": None,  # score는 없으므로 None으로 설정
            "type": None    # type도 없으므로 None으로 설정
        }
        keypoints_list.append(keypoint)

    pose_info["keypoints"] = keypoints_list
    output_data["pose_estimation_info"].append(pose_info)

    # JSON 저장 디렉토리 생성
    os.makedirs("4/predicted_json", exist_ok=True)
    json_path = os.path.join("4/predicted_json", f"{filename}_predicted.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"Predicted keypoints JSON saved as {json_path}")

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
    x_min = bbox['top_left']['x']
    y_min = bbox['top_left']['y']
    x_max = bbox['bottom_right']['x']
    y_max = bbox['bottom_right']['y']

    width_bbox = x_max - x_min
    height_bbox = y_max - y_min

    predicted_keypoints = predicted_features[:, :2]

    # 좌표를 원본 이미지 크기에 맞게 스케일링
    predicted_keypoints_rescaled = predicted_keypoints.copy()
    predicted_keypoints_rescaled[:, 0] = predicted_keypoints_rescaled[:, 0] * width_bbox + x_min
    predicted_keypoints_rescaled[:, 1] = predicted_keypoints_rescaled[:, 1] * height_bbox + y_min

    # 키포인트 차이 계산
    distances, mean_distance = calculate_keypoint_differences(
        original_keypoints, predicted_keypoints_rescaled, image_width, image_height)

    # 결과 출력
    print(f"Mean keypoint distance (normalized): {mean_distance:.4f}")
    for idx, distance in enumerate(distances):
        print(f"Keypoint {idx}: Distance = {distance:.4f}")

    # 예측된 키포인트를 JSON으로 저장
    save_predicted_json(json_data, predicted_keypoints_rescaled, filename)

    # 시각화
    plot_and_save(ax, original_keypoints, predicted_keypoints_rescaled, x_min, y_min, width_bbox, height_bbox, similarity_score, filename, image_width, image_height, distances, mean_distance)

def plot_and_save(ax, original_keypoints, predicted_keypoints, x_min, y_min, width_bbox, height_bbox, similarity_score, filename, image_width, image_height, distances, mean_distance):
    # 디렉토리가 존재하지 않으면 생성합니다.
    os.makedirs("4/original", exist_ok=True)
    os.makedirs("4/eval", exist_ok=True)
    os.makedirs("4/differences", exist_ok=True)

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
    original_path = os.path.join("4/original", f"{filename}_original.png")                         #############                     
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
    eval_path = os.path.join("4/eval", f"{filename}_eval.png")                                      ##########
    plt.savefig(eval_path, bbox_inches='tight', pad_inches=0)
    print(f"Evaluation plot saved as {eval_path}")
    ax.cla()

    # 원본 및 예측된 키포인트와 차이 벡터 그리기
    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)
    ax.invert_yaxis()
    ax.add_patch(rect)
    ax.scatter(original_keypoints[:, 0], original_keypoints[:, 1], c='blue', s=50, label='Original')
    ax.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], c='red', s=50, marker='x', label='Predicted')

    # 키포인트 간 차이 벡터 그리기
    for i in range(len(original_keypoints)):
        ax.plot([original_keypoints[i, 0], predicted_keypoints[i, 0]],
                [original_keypoints[i, 1], predicted_keypoints[i, 1]],
                'g--', linewidth=1)
        # 각 키포인트에 거리 표시
        ax.text(predicted_keypoints[i, 0], predicted_keypoints[i, 1], f'{distances[i]:.2f}', color='green')

    ax.set_title(f"Keypoint Differences (Mean Distance: {mean_distance:.4f})")
    ax.legend()
    ax.axis('off')
    diff_path = os.path.join("4/differences", f"{filename}_differences.png")
    plt.savefig(diff_path, bbox_inches='tight', pad_inches=0)
    print(f"Differences plot saved as {diff_path}")
    plt.close()

if __name__ == '__main__':
    # 테스트 JSON 폴더 경로 정의
    test_json_dir = './test_img_4'

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
