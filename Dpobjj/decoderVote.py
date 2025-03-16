import torch.nn.functional as F
import torch.nn as nn
import torch

from Dpobjj.HoughVoter import HoughVoterPyTorch  # Ensure correct import

class HoughVotingDecoder(nn.Module):
    def __init__(self, num_classes, image_height, image_width):
        super().__init__()
        self.num_classes = num_classes
        self.image_height = image_height
        self.image_width = image_width

        # Upsampling Layers
        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(256 + 256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64 + 64, 64, kernel_size=4, stride=2, padding=1)
        self.upconv1 = nn.ConvTranspose2d(64 + 64, 64, kernel_size=4, stride=2, padding=1)

        # Final Layers
        self.vertex_out = nn.Conv2d(64, 3 * num_classes, kernel_size=1)  # (dx, dy, d) per class
        self.class_out = nn.Conv2d(64, num_classes, kernel_size=1)  # Class logits

        # Hough Voting Module
        self.hough_voter = HoughVoterPyTorch(image_height, image_width,0.5)

    def forward(self, features, depth_predictions, segmentation_map, fx, fy , px, py):
        """
        Args:
            features (tuple of tensors): Feature maps from encoder.(5 tensors from resnet18)
            depth_predictions (Tensor): Depth output from another decoder.
            segmentation_map (Tensor): Segmentation output from another decoder.
        """
        x5, x4, x3, x2, x1 = features

        # Upsampling
        x = F.relu(self.upconv5(x1))
        x = F.relu(self.upconv4(torch.cat([x, x2], dim=1)))
        x = F.relu(self.upconv3(torch.cat([x, x3], dim=1)))
        x = F.relu(self.upconv2(torch.cat([x, x4], dim=1)))
        x = F.relu(self.upconv1(torch.cat([x, x5], dim=1)))

        # Final Outputs
        bottom_vertex = self.vertex_out(x)  # (dx, dy, d)
        bottom_label_logits = self.class_out(x)  # Class logits
        bottom_label = torch.argmax(bottom_label_logits, dim=1)

        # Prepare inputs for Hough voting
        center_directions = bottom_vertex[:, :2, :, :].permute(0, 2, 3, 1).contiguous()  # (B, H, W, 2)

        # Ensure tensors remain on the same device and require gradients
        depth_predictions = depth_predictions.to(torch.float32).requires_grad_()
        segmentation_map = segmentation_map.to(torch.float32).requires_grad_()

        # Apply Hough voting
        B, num_classes, H, W = segmentation_map.shape
        vote_maps = torch.zeros((B, H, W), dtype=torch.float32, device=segmentation_map.device)
        batch_centers = []
        batch_translations = []

        for b in range(B):
            # Convert segmentation to binary mask per image
            seg_map = torch.clamp(segmentation_map[b].sum(dim=0), 0, 1)  # Keep it in tensor

            # Process using HoughVoter (Modify HoughVoter to support PyTorch tensors)
            vote_map, centers, translations = self.hough_voter.cast_votes(seg_map, center_directions[b], depth_predictions[b, 0], fx, fy , px, py  )

            vote_maps[b] = vote_map
            batch_centers.append(centers)
            batch_translations.append(translations)

        return bottom_vertex, bottom_label, vote_maps, batch_centers, batch_translations
