import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, l2_scale=0.0
    ):
        super(MaskedConv2D, self).__init__()

        # Feature convolution
        self.feature_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            bias=False,
        )

        # Mask convolution (normalization)
        self.norm_conv = nn.Conv2d(
            1, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=False
        )
        nn.init.constant_(
            self.norm_conv.weight, 1.0
        )  # Initialize weights to all ones (as in the diagram)
        self.norm_conv.requires_grad_(False)  # Keep mask convolution non-trainable

        # Learnable bias
        self.bias = nn.Parameter(
            torch.zeros(out_channels)
        )  # Learnable bias (as in diagram)

    def forward(self, x, mask=None):
        if mask is None:  # First layer, generate binary mask
            mask = (
                x[:, :1, :, :] != 0
            ).float()  # Assume 1st channel represents mask info

        # Apply the mask to input features
        masked_features = x * mask

        # Perform feature convolution
        conv_features = self.feature_conv(masked_features)

        # Perform mask convolution to get normalization factor
        norm = self.norm_conv(mask)

        # Normalize the features
        norm = torch.where(
            norm == 0, torch.zeros_like(norm), 1.0 / norm
        )  # Avoid division by zero
        output = conv_features * norm + self.bias.view(1, -1, 1, 1)  # Apply bias

        # Update mask using max pooling
        new_mask = F.max_pool2d(
            mask, kernel_size=3, stride=1, padding=1
        )  # Mask pooling

        return output, new_mask


# Example usage
if __name__ == "__main__":
    x = torch.randn(1, 2, 5, 5)  # (Batch, Channels, Height, Width)
    mask = torch.randint(0, 2, (1, 1, 5, 5)).float()  # Binary mask

    model = MaskedConv2D(2, 4, kernel_size=3, stride=1)  # Example 2->4 channels conv
    features, new_mask = model(x, mask)

    print("Output Features:", features.shape)
    print("Updated Mask:", new_mask.shape)
