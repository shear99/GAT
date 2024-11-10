# main.py
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from KeypointDataset import KeypointDataset
from GAT import GAT
import torch.optim as optim

def main():
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {'CUDA' if device.type == 'cuda' else 'CPU'}")

    # Load dataset
    dataset = KeypointDataset(json_dir='3/json')  # 모든 JSON 파일이 있는 폴더 지정
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Determine input feature size
    sample_data = dataset[0]
    in_channels = sample_data.x.size(1)

    # Initialize model
    model = GAT(in_channels=in_channels, hidden_channels=8, out_channels=in_channels, num_heads=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Keypoint importance indices
    num_keypoints = 17
    arm_leg_indices = [7, 8, 9, 10, 15, 16]  # 팔과 다리 부분
    torso_thigh_indices = [5, 6, 11, 12, 13, 14]  # 몸통과 허벅지 부분
    face_indices = [0, 1, 2, 3, 4]  # 얼굴 부분

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)

            # Compute the MSE loss with no reduction to retain individual losses
            # Only consider the coordinate part (first two features) for loss calculation
            loss = F.mse_loss(out[:, :2], data.x[:, :2], reduction='none')  # shape: [total_num_nodes, 2]

            # Average over features (x, y) for each node
            loss_per_node = loss.mean(dim=1)  # shape: [total_num_nodes]

            # Calculate node indices within each batch to identify keypoints
            node_indices = torch.arange(data.x.size(0), device=device) % num_keypoints

            # Assign weights to keypoints based on importance
            weights = torch.ones(num_keypoints, device=device)  # 기본 가중치 1로 초기화
            weights[arm_leg_indices] = 0.8   # 팔과 다리 부분에 높은 가중치
            weights[torso_thigh_indices] = 0.4     # 몸통과 허벅지 부분에 기본 가중치 
            weights[face_indices] = 0.005      # 얼굴 부분에 매우 낮은 가중치

            # Apply weights to each node in the batch
            weights_per_node = weights[node_indices]
            weighted_loss = (loss_per_node * weights_per_node).mean()

            # Backpropagation and optimization
            weighted_loss.backward()
            optimizer.step()
            total_loss += weighted_loss.item()

        # Calculate and print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'gat_model.pth')
    print("Model saved as gat_model.pth")

if __name__ == '__main__':
    main()
