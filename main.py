# main.py

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from KeypointDataset import KeypointDataset
from GAT import GAT
import torch.optim as optim

def main():
    # CUDA 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {'CUDA' if device.type == 'cuda' else 'CPU'}")

    # 데이터셋 로드
    dataset = KeypointDataset(json_dir='3/json')  # 모든 JSON 파일이 있는 폴더 지정
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 입력 특징 크기 결정
    sample_data = dataset[0]
    in_channels = sample_data.x.size(1)

    # 모델 초기화
    model = GAT(in_channels=in_channels, hidden_channels=64, out_channels=in_channels, num_heads=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 키포인트 중요도 인덱스 설정
    num_keypoints = 17  # 키포인트 총 개수 (인덱스 0부터 16까지)

    # 각 키포인트 인덱스별로 가중치를 개별적으로 설정합니다.
    # 예시로 가중치를 설정하였으니, 필요한 값으로 수정하시면 됩니다.
    importance_weights = torch.tensor([
        0.3,  # 인덱스 0
        0.3,  # 인덱스 1
        0.3,  # 인덱스 2
        0.3,  # 인덱스 3
        0.3,  # 인덱스 4 -> 여기까지 얼굴
        0.6,  # 인덱스 5
        0.2,  # 인덱스 6
        0.8,  # 인덱스 7
        0.8,  # 인덱스 8
        0.8,  # 인덱스 9
        0.8,  # 인덱스 10
        0.2,  # 인덱스 11
        0.2,  # 인덱스 12
        0.2,  # 인덱스 13
        0.2,  # 인덱스 14
        0.8,  # 인덱스 15
        0.8   # 인덱스 16
    ], device=device)

    # 학습 루프
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_weight)

            # 좌표 손실 계산 (MSE Loss)
            coord_loss = F.mse_loss(out[:, :2], data.x[:, :2], reduction='none')  # [total_num_nodes, 2]
            coord_loss = coord_loss.sum(dim=1)  # [total_num_nodes]

            # 각도 손실 계산 (코사인 유사도 기반)
            angle_loss = torch.zeros_like(coord_loss)
            valid_angle_mask = (data.x[:, 2:].abs().sum(dim=1) > 0)
            if valid_angle_mask.any():
                cosine_similarity = F.cosine_similarity(out[valid_angle_mask, 2:], data.x[valid_angle_mask, 2:], dim=1)
                angle_loss[valid_angle_mask] = 1 - cosine_similarity  # 코사인 거리로 변환

            # 총 손실 계산 (좌표 손실 + 각도 손실)
            total_node_loss = coord_loss + angle_loss

            # 노드별 가중치 적용
            # 노드 인덱스를 키포인트 인덱스에 맞게 계산
            node_indices = torch.remainder(torch.arange(data.num_nodes, device=device), num_keypoints)
            weights_per_node = importance_weights[node_indices]
            weighted_node_loss = total_node_loss * weights_per_node

            # 배치의 평균 손실 계산
            loss = weighted_node_loss.mean()

            # 역전파 및 옵티마이저 스텝
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 에폭별 평균 손실 출력
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}')

    # 학습된 모델 저장
    torch.save(model.state_dict(), 'gat_model.pth')
    print("Model saved as gat_model.pth")

if __name__ == '__main__':
    main()
