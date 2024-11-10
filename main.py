# main.py
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader  # 업데이트된 임포트
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
    model = GAT(in_channels=in_channels, hidden_channels=8, out_channels=in_channels, num_heads=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 키포인트 중요도 인덱스
    num_keypoints = 17
    arm_leg_indices = [7, 8, 9, 10, 15, 16]  # 팔과 다리 부분
    torso_thigh_indices = [5, 6, 11, 12, 13, 14]  # 몸통과 허벅지 부분
    face_indices = [0, 1, 2, 3, 4]  # 얼굴 부분

    # 학습 루프
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)

            # 좌표 손실 계산 (MSE Loss)
            coord_loss = F.mse_loss(out[:, :2], data.x[:, :2], reduction='none')  # shape: [total_num_nodes, 2]
            coord_loss_per_node = coord_loss.mean(dim=1)  # shape: [total_num_nodes]

            # 각도 손실 계산 (코사인 유사도 기반)
            angle_loss_per_node = torch.zeros_like(coord_loss_per_node)

            # 유효한 각도 벡터를 가진 노드 선택
            valid_mask = (data.x[:, 2:].sum(dim=1) != 0)
            if valid_mask.sum() > 0:
                cosine_sim = F.cosine_similarity(out[valid_mask, 2:], data.x[valid_mask, 2:], dim=1)
                angle_loss = 1 - cosine_sim  # 유사도를 손실로 변환
                angle_loss_per_node[valid_mask] = angle_loss
            else:
                angle_loss_per_node = torch.zeros_like(coord_loss_per_node)

            # 노드별 총 손실 계산
            loss_per_node = coord_loss_per_node + 0.5 * angle_loss_per_node

            # 키포인트별 가중치 적용
            node_indices = torch.arange(data.x.size(0), device=device) % num_keypoints
            weights = torch.ones(num_keypoints, device=device)
            weights[arm_leg_indices] = 0.758
            weights[torso_thigh_indices] = 0.387
            weights[face_indices] = 0.1
            weights_per_node = weights[node_indices]

            # 가중치를 적용한 손실 계산
            weighted_loss_per_node = loss_per_node * weights_per_node
            weighted_loss = weighted_loss_per_node.mean()

            # 역전파 및 옵티마이저 스텝
            weighted_loss.backward()
            optimizer.step()
            total_loss += weighted_loss.item()

        # 에폭별 평균 손실 출력
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    # 학습된 모델 저장
    torch.save(model.state_dict(), 'gat_model.pth')
    print("Model saved as gat_model.pth")

if __name__ == '__main__':
    main()
