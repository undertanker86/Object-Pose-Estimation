import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)  # Global Average Pooling
        attention = self.mlp(avg_pool).view(b, c, 1, 1)
        return x * attention  # Element-wise multiplication

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # Average pooling over channels
        attention = self.sigmoid(self.conv(avg_pool))
        return x * attention  # Element-wise multiplication

class MultiModalDistillation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(MultiModalDistillation, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, Fs, Fd, Fv):
        # Concatenation of multi-modal features
        F_fused = torch.cat([Fs, Fd, Fv], dim=1)
        
        # Apply channel attention
        F_focused = self.channel_attention(F_fused)
        
        # Apply spatial attention
        F_enhanced = self.spatial_attention(F_focused)
        
        return F_enhanced
