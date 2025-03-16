import torch
import sys
# sys.path.append("./posecnn")
# import posecnn_cuda
import torch.nn as nn
import torch.nn.functional as F
# from posecnn.hough_voting import HoughVoting
import numpy as np
import scipy.ndimage


class HoughVotingCPU:
    def __init__(self, label_threshold=100, inlier_threshold=0.9):
        self.label_threshold = label_threshold
        self.inlier_threshold = inlier_threshold

    def forward(self, label, direction_vectors):
        """
        Implements Hough Voting on CPU.

        Args:
            label: (B, H, W) - Object instance labels per pixel.
            direction_vectors: (B, 2, H, W) - Direction vectors (dx, dy) for center prediction.

        Returns:
            instance_centers: List of detected instance centers per batch.
        """
        B, H, W = label.shape
        instance_centers = []

        for b in range(B):  # Process each batch separately
            current_label = label[b].detach().cpu().numpy()
            dx, dy = direction_vectors[b, 0].detach().cpu().numpy(), direction_vectors[b, 1].detach().cpu().numpy()
            
            # Find unique object labels
            unique_labels = np.unique(current_label)
            unique_labels = unique_labels[unique_labels > 0]  # Ignore background (label = 0)

            centers = []
            for obj_label in unique_labels:
                mask = current_label == obj_label  # Pixels belonging to this object
                
                # Get voting points
                y_idx, x_idx = np.where(mask)
                if len(y_idx) == 0:
                    continue

                # Compute votes for center locations
                vote_x = np.clip(x_idx + dx[y_idx, x_idx], 0, W - 1).astype(int)
                vote_y = np.clip(y_idx + dy[y_idx, x_idx], 0, H - 1).astype(int)

                # Accumulate votes in a Hough space
                hough_space = np.zeros((H, W), dtype=np.float32)
                for vx, vy in zip(vote_x, vote_y):
                    hough_space[vy, vx] += 1

                # Find local maxima as object centers
                max_vote = hough_space.max()
                if max_vote > self.label_threshold:
                    center_y, center_x = np.unravel_index(hough_space.argmax(), hough_space.shape)
                    centers.append((center_x, center_y))

            instance_centers.append(centers)

        return instance_centers


class CenterVotingDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=2, is_train=0):
        """
        in_channels: Number of input channels (from backbone features).
        out_channels: 2 (for center direction vectors: dx, dy).
        is_train: Whether the model is in training mode (affects Hough Voting behavior).
        """
        super(CenterVotingDecoder, self).__init__()
        
        # Basic CNN layers for processing features
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, out_channels, kernel_size=1)  # Output (dx, dy)
        
        # Hough Voting Module (CPU Version)
        self.hough_voting = HoughVotingCPU()

    def forward(self, x, label):
        """
        x: Feature map from backbone
        label: Object segmentation mask
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        direction_vectors = self.conv4(x)  # Shape: (B, 2, H, W) -> (dx, dy) per pixel
        
        # Normalize vectors to unit length
        direction_vectors = F.normalize(direction_vectors, dim=1)
        
        # Apply Hough Voting to get object centers
        top_box = self.hough_voting.forward(label, direction_vectors)

        return direction_vectors, top_box


# torch.cuda.empty_cache()

# device = torch.cuda.current_device()
# device_props = torch.cuda.get_device_properties(device)


# print("Max threads per block:", device_props.max_threads_per_multi_processor)

# # Initialize model
# decoder = CenterVotingDecoder(in_channels=256)

# # Dummy inputs
# features = torch.randn(1, 256, 64, 64)  # Backbone feature map
# label = torch.randint(0, 2, (1, 64, 64)).to(torch.int32)  # Binary segmentation mask
# meta_data = torch.randn(1, 3, 3)  # Camera intrinsics
# extents = torch.randn(1, 3)  # Object dimensions




# # Run forward pass
# direction_vectors, top_box, top_pose = decoder(features, label, meta_data, extents)

# print("Direction Vectors Shape:", direction_vectors.shape)  # (B, 2, H, W)
# print("Top Box:", top_box)  # Predicted bounding boxes
# print("Top Pose:", top_pose)  # Predicted poses

B, H, W = 1, 128, 128  # Batch size 1, 128x128 image
label = torch.randint(0, 3, (B, H, W))  # Random labels (0 = background, 1,2 = objects)
feature_map = torch.randn(B, 64, H, W)  # Simulated backbone features

decoder = CenterVotingDecoder(in_channels=64)
direction_vectors, centers = decoder(feature_map, label)

print("Direction Vectors Shape:", direction_vectors.shape)
print("Detected Centers:", centers)