import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from KeypointDataset import KeypointDataset
from GAT import GAT
import torch.optim as optim

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {'CUDA' if device.type == 'cuda' else 'CPU'}")

    dataset = KeypointDataset(json_dir='3/json')
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    sample_data = dataset[0]
    in_channels = sample_data.x.size(1)

    model = GAT(in_channels=in_channels, hidden_channels=64, out_channels=in_channels, num_heads=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_keypoints = 17
    importance_weights = torch.tensor([
        0.3, 0.1, 0.1, 0.1, 0.1, 
        0.5, 0.5, 0.7, 0.7, 0.8, 
        0.8, 0.2, 0.2, 0.5, 0.5, 
        0.8, 0.8
    ], device=device)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)

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

    torch.save(model.state_dict(), 'gat_model.pth')
    print("Model saved as gat_model.pth")

if __name__ == '__main__':
    main()
