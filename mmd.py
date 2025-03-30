import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(-1).unsqueeze(-1)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class MMD(nn.Module):
    def __init__(self, depth_channels, seg_channels, vector_channels):
        super(MMD, self).__init__()

        # Channel Attention for different feature types
        self.channel_attn_depth = ChannelAttention(depth_channels)
        self.channel_attn_seg = ChannelAttention(seg_channels)
        self.channel_attn_vector = ChannelAttention(vector_channels)

        # Spatial Attention for combined features
        self.spatial_attn = SpatialAttention()
        
        # Fusion convolution
        self.fusion_conv = nn.Conv2d(
            depth_channels + seg_channels + vector_channels, 
            256, 
            kernel_size=3, 
            padding=1
        )
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth_features, seg_features, vector_features):
        # Apply channel attention
        depth_features = depth_features * self.channel_attn_depth(depth_features)
        seg_features = seg_features * self.channel_attn_seg(seg_features)
        vector_features = vector_features * self.channel_attn_vector(vector_features)

        # Concatenate features
        combined_features = torch.cat([depth_features, seg_features, vector_features], dim=1)

        # Apply spatial attention
        spatial_weight = self.spatial_attn(combined_features)
        combined_features = combined_features * spatial_weight
        
        # Final fusion
        fused_features = self.relu(self.bn(self.fusion_conv(combined_features)))

        return fused_features

class KeypointPoseEstimation(nn.Module):
    def __init__(self, feature_dim=256, num_keypoints=8):
        super(KeypointPoseEstimation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(feature_dim, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
        self.fc_pos = nn.Linear(512, 3)  # Predicts object position (x, y, z)
        self.fc_keypoints = nn.Linear(512, num_keypoints * 2)  # Predicts keypoints (u, v)
    
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc(x))
        x = self.dropout(x)
        
        position = self.fc_pos(x)
        keypoints = self.fc_keypoints(x).view(x.size(0), -1, 2)  # Reshape to (batch, num_keypoints, 2)
        return position, keypoints

class PoseEstimationModel(nn.Module):
    def __init__(self, depth_channels=1, seg_channels=19, vector_channels=2, feature_dim=256, num_keypoints=8):
        super(PoseEstimationModel, self).__init__()
        self.mmd = MMD(depth_channels, seg_channels, vector_channels)
        self.pose_estimation = KeypointPoseEstimation(feature_dim, num_keypoints)
    
    def forward(self, depth_features, seg_features, vector_features):
        fused_features = self.mmd(depth_features, seg_features, vector_features)
        position, keypoints = self.pose_estimation(fused_features)
        return position, keypoints