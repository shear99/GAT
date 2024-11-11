import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from GAT import GAT
from KeypointDataset import KeypointDataset

MODEL_PATH = 'gat_model.pth'                # 학습된 모델 파일 경로
TEST_JSON_DIR = './test_data'               # 테스트 JSON 데이터 폴더 경로
PREDICTED_JSON_DIR = './output/predict'     # 예측 결과 JSON 저장 경로
ORIGINAL_IMG_DIR = './output/original/'     # 원본 키포인트 시각화 이미지 저장 경로
EVAL_IMG_DIR = './output/eval/'             # 예측된 키포인트 시각화 이미지 저장 경로
DIFF_IMG_DIR = './output/differences/'      # 키포인트 차이 시각화 이미지 저장 경로

class Evaluator:
    def __init__(self, model_path, json_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 샘플 데이터를 사용하여 in_channels 값을 결정합니다.
        sample_dataset = KeypointDataset(json_dir=json_dir)
        sample_data = sample_dataset[0]
        in_channels = sample_data.x.size(1)

        # 모델 초기화 (훈련 시 사용한 hidden_channels 설정)
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
    # 키포인트 개수가 일치하지 않으면 None 반환
    if original_keypoints.shape[0] != predicted_keypoints.shape[0]:
        print(f"Warning: 키포인트 개수가 일치하지 않습니다! 원본: {original_keypoints.shape[0]}, 예측: {predicted_keypoints.shape[0]}")
        return None, None

    # 키포인트 좌표를 이미지 크기로 정규화합니다.
    original_normalized = original_keypoints / np.array([image_width, image_height])
    predicted_normalized = predicted_keypoints / np.array([image_width, image_height])
    
    # 키포인트별 좌표 차이 계산
    differences = original_normalized - predicted_normalized
    # 유클리드 거리 계산
    distances = np.linalg.norm(differences, axis=1)
    # 평균 거리 계산
    mean_distance = np.mean(distances)
    return distances, mean_distance

def save_predicted_json(json_data, predicted_keypoints, filename, distances, vector_differences):
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
            "score": None,
            "type": None
        }
        keypoints_list.append(keypoint)

    pose_info["keypoints"] = keypoints_list

    # pose_estimation_difference 추가
    pose_info["pose_estimation_difference"] = {
        "keypoint_distances": distances.tolist(),
        "keypoint_distance_sum": float(np.sum(distances)),
        "selected_keypoint_indices": [7, 8, 9, 10, 13, 14, 15, 16],
        "selected_keypoint_distance_sum": float(np.sum(distances[[7, 8, 9, 10, 13, 14, 15, 16]])),
        "vector_pairs": [[15, 13], [13, 11], [16, 14], [14, 12], [7, 9], [8, 10], [3, 5], [4, 6]],
        "vector_differences": vector_differences,
        "vector_difference_sum": float(np.sum(vector_differences))
    }

    output_data["pose_estimation_info"].append(pose_info)

    # JSON 저장 디렉토리 생성
    os.makedirs(PREDICTED_JSON_DIR, exist_ok=True)
    json_path = os.path.join(PREDICTED_JSON_DIR, f"{filename}_predicted.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"예측된 키포인트 JSON 저장 완료: {json_path}")

def visualize_pose(json_data, predicted_features, similarity_score, filename):
    # JSON에서 이미지의 가로와 세로 크기를 가져옵니다.
    image_width = json_data['image_width']
    image_height = json_data['image_height']

    # 바운딩 박스와 키포인트를 추출합니다.
    pose_info = json_data['pose_estimation_info'][0]
    bbox = pose_info['bbox']
    keypoints = pose_info['keypoints']

    # 원본 키포인트 좌표 추출
    original_keypoints = []
    for kp in keypoints:
        x = kp['coordinates']['x']
        y = kp['coordinates']['y']
        original_keypoints.append([x, y])
    original_keypoints = np.array(original_keypoints)

    # 예측된 키포인트 좌표를 스케일 조정
    x_min = bbox['top_left']['x']
    y_min = bbox['top_left']['y']
    x_max = bbox['bottom_right']['x']
    y_max = bbox['bottom_right']['y']
    width_bbox = x_max - x_min
    height_bbox = y_max - y_min

    predicted_keypoints = predicted_features[:, :2]
    predicted_keypoints_rescaled = predicted_keypoints.copy()
    predicted_keypoints_rescaled[:, 0] = predicted_keypoints_rescaled[:, 0] * width_bbox + x_min
    predicted_keypoints_rescaled[:, 1] = predicted_keypoints_rescaled[:, 1] * height_bbox + y_min

    # 키포인트 개수가 일치하지 않으면 스킵
    if original_keypoints.shape[0] != predicted_keypoints_rescaled.shape[0]:
        print(f"{filename} 시각화 생략: 키포인트 개수 불일치")
        return

    # 키포인트 차이 계산
    distances, mean_distance = calculate_keypoint_differences(
        original_keypoints, predicted_keypoints_rescaled, image_width, image_height)

    # 차이 계산 불가능 시 스킵
    if distances is None or mean_distance is None:
        print(f"{filename} 시각화 생략: 차이 계산 불가능")
        return

    print(f"평균 키포인트 거리 (정규화됨): {mean_distance:.4f}")
    for idx, distance in enumerate(distances):
        print(f"키포인트 {idx}: 거리 = {distance:.4f}")

    # 예측된 키포인트를 JSON으로 저장
    save_predicted_json(json_data, predicted_keypoints_rescaled, filename, distances, [])

if __name__ == '__main__':
    # 테스트 JSON 폴더 경로 정의
    test_json_dir = TEST_JSON_DIR

    # 데이터셋 및 평가자 초기화
    dataset = KeypointDataset(json_dir=test_json_dir)
    evaluator = Evaluator(model_path=MODEL_PATH, json_dir=test_json_dir)

    # 테스트 폴더에서 JSON 파일 목록 가져오기
    json_files = [f for f in os.listdir(test_json_dir) if f.endswith('.json')]

    # 각 샘플을 평가하고 시각화
    for i, (sample_data, json_file) in enumerate(zip(dataset, json_files)):
        similarity_score, predicted_features = evaluator.evaluate(sample_data)
        filename = os.path.splitext(json_file)[0]
        original_json_path = os.path.join(test_json_dir, json_file)
        with open(original_json_path, 'r') as f:
            json_data = json.load(f)
        visualize_pose(json_data, predicted_features, similarity_score, filename)
