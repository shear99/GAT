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
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GAT(in_channels=2, hidden_channels=8, out_channels=2, num_heads=8).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def evaluate(self, data):
        # data is a torch_geometric.data.Data object
        data = data.to(self.device)
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            similarity = F.cosine_similarity(out, data.x).mean().item()
        return similarity, out.cpu().numpy()  # Return predicted keypoints for visualization

def visualize_pose(json_data, predicted_keypoints, similarity_score, filename):
    # Use image width and height from JSON
    image_width = json_data['image_width']
    image_height = json_data['image_height']

    # Create a blank image using the image dimensions
    fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=100)

    # Set the limits of the plot to match the image dimensions
    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)
    ax.invert_yaxis()  # Invert y-axis to match image coordinate system

    # Extract bounding box and keypoints
    pose_info = json_data['pose_estimation_info'][0]
    bbox = pose_info['bbox']
    keypoints = pose_info['keypoints']

    # Check if the number of keypoints is 17
    if len(keypoints) != 17:
        print(f"Skipping {filename}: Keypoints count is {len(keypoints)}, expected 17.")
        return

    # Original keypoints coordinates
    original_keypoints = np.array([[kp['coordinates']['x'], kp['coordinates']['y']] for kp in keypoints])

    # Predicted keypoints coordinates (rescaled)
    # Extract bounding box coordinates
    x_min = bbox['top_left']['x']
    y_min = bbox['top_left']['y']
    x_max = bbox['bottom_right']['x']
    y_max = bbox['bottom_right']['y']

    width_bbox = x_max - x_min
    height_bbox = y_max - y_min

    predicted_keypoints_rescaled = predicted_keypoints.copy()
    predicted_keypoints_rescaled[:, 0] = predicted_keypoints_rescaled[:, 0] * width_bbox + x_min
    predicted_keypoints_rescaled[:, 1] = predicted_keypoints_rescaled[:, 1] * height_bbox + y_min

    # Plotting original keypoints and bounding box
    plot_and_save(ax, original_keypoints, predicted_keypoints_rescaled, x_min, y_min, width_bbox, height_bbox, similarity_score, filename, image_width, image_height)

def plot_and_save(ax, original_keypoints, predicted_keypoints, x_min, y_min, width_bbox, height_bbox, similarity_score, filename, image_width, image_height):
    # Create the directories if they do not exist
    os.makedirs("original", exist_ok=True)
    os.makedirs("eval", exist_ok=True)

    # Skeleton connections
    skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
        [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
        [1, 3], [2, 4], [3, 5], [4, 6]
    ]

    # Draw bounding box
    rect = patches.Rectangle((x_min, y_min), width_bbox, height_bbox, linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # Plot original keypoints and skeleton
    for edge in skeleton:
        x_coords = [original_keypoints[edge[0], 0], original_keypoints[edge[1], 0]]
        y_coords = [original_keypoints[edge[0], 1], original_keypoints[edge[1], 1]]
        ax.plot(x_coords, y_coords, 'b-', linewidth=2)
    ax.scatter(original_keypoints[:, 0], original_keypoints[:, 1], c='blue', s=50)
    ax.set_title("Original Keypoints")
    ax.axis('off')
    original_path = os.path.join("original", f"{filename}_original.png")
    plt.savefig(original_path, bbox_inches='tight', pad_inches=0)
    print(f"Original plot saved as {original_path}")
    ax.cla()  # Clear the axis for the next plot

    # Plot predicted keypoints and skeleton
    # Re-use the same axis to ensure consistent sizing
    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)
    ax.invert_yaxis()
    ax.add_patch(rect)
    for edge in skeleton:
        x_coords = [predicted_keypoints[edge[0], 0], predicted_keypoints[edge[1], 0]]
        y_coords = [predicted_keypoints[edge[0], 1], predicted_keypoints[edge[1], 1]]
        ax.plot(x_coords, y_coords, 'r--', linewidth=2)
    ax.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], c='red', s=50)
    ax.set_title(f"Predicted Keypoints (Similarity: {similarity_score:.4f})")
    ax.axis('off')
    eval_path = os.path.join("eval", f"{filename}_eval.png")
    plt.savefig(eval_path, bbox_inches='tight', pad_inches=0)
    print(f"Evaluation plot saved as {eval_path}")
    plt.close()

if __name__ == '__main__':
    # Define the test JSON folder path
    test_json_dir = './test_img_3'

    # Initialize dataset and evaluator
    dataset = KeypointDataset(json_dir=test_json_dir)
    evaluator = Evaluator(model_path='gat_model.pth')

    # Get list of JSON files from the test folder
    json_files = [f for f in os.listdir(test_json_dir) if f.endswith('.json')]

    # Evaluate and visualize each sample in the dataset
    for i, (sample_data, json_file) in enumerate(zip(dataset, json_files)):
        # Check if the sample has 17 keypoints
        if sample_data.x.size(0) != 17:
            print(f"Skipping {json_file}: Keypoints count is {sample_data.x.size(0)}, expected 17.")
            continue

        similarity_score, predicted_keypoints = evaluator.evaluate(sample_data)

        # Define filename based on original JSON filename
        filename = os.path.splitext(json_file)[0]

        # Full path to the original JSON file
        original_json_path = os.path.join(test_json_dir, json_file)

        # Load JSON data
        with open(original_json_path, 'r') as f:
            json_data = json.load(f)

        # Visualize and save the result
        visualize_pose(json_data, predicted_keypoints, similarity_score, filename)
