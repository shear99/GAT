import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from KeypointDataset import KeypointDataset
from GAT import GAT
import torch.optim as optim

MODEL_PATH = 'gat_model.pth'                # 학습된 모델 파일 경로
TRAIN_JSON_DIR = './input/3/json'              # 테스트 JSON 데이터 폴더 경로


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {'CUDA' if device.type == 'cuda' else 'CPU'}")

    dataset = KeypointDataset(json_dir=TRAIN_JSON_DIR)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    sample_data = dataset[0]
    in_channels = sample_data.x.size(1)

    model = GAT(in_channels=in_channels, hidden_channels=64, out_channels=in_channels, num_heads=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_keypoints = 17
    importance_weights = torch.tensor([
        0.2,  # 인덱스 0: 코
        0.2,  # 인덱스 1: 왼쪽 눈
        0.2,  # 인덱스 2: 오른쪽 눈
        0.2,  # 인덱스 3: 왼쪽 귀
        0.2,  # 인덱스 4: 오른쪽 귀 -> 여기까지 얼굴
        0.6,  # 인덱스 5: 왼쪽 어깨
        0.6,  # 인덱스 6: 오른쪽 어깨
        0.6,  # 인덱스 7: 왼쪽 팔꿈치
        0.6,  # 인덱스 8: 오른쪽 팔꿈치
        0.8,  # 인덱스 9: 왼쪽 손목
        0.8,  # 인덱스 10: 오른쪽 손목
        0.5,  # 인덱스 11: 왼쪽 엉덩이
        0.5,  # 인덱스 12: 오른쪽 엉덩이
        0.7,  # 인덱스 13: 왼쪽 무릎
        0.7,  # 인덱스 14: 오른쪽 무릎
        0.8,  # 인덱스 15: 왼쪽 발목
        0.8   # 인덱스 16: 오른쪽 발목
    ], device=device)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_weight)

            coord_loss = F.mse_loss(out[:, :2], data.x[:, :2], reduction='none').sum(dim=1)
            total_node_loss = coord_loss
            node_indices = torch.remainder(torch.arange(data.num_nodes, device=device), num_keypoints)
            weights_per_node = importance_weights[node_indices]
            weighted_node_loss = total_node_loss * weights_per_node

            loss = weighted_node_loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}')

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved as gat_model.pth")

if __name__ == '__main__':
    main()
