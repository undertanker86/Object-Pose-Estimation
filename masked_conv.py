import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
# Note using SpConv https://github.com/traveller59/spconv can have better performance
class InstanceConv(nn.Module):
    """
    Sparse convolution layer that handles masked input data.
    Masks invalid regions during convolution and normalizes outputs based on valid pixels.
    """
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bn: bool = False,
        if_bias: bool = True,
        mask_invalid_value: float = -1.0,
    ):
        super(InstanceConv, self).__init__()
        self.k = kernel_size
        self.in_c = in_channel
        self.out_c = out_channel
        self.stride = stride
        self.padding = padding
        self.bn = bn
        self.if_bias = if_bias
        self.mask_invalid_value = mask_invalid_value

        # Define learnable parameters
        self.conv = nn.Parameter(torch.Tensor(self.out_c, self.in_c, self.k, self.k))
        self.bias = nn.Parameter(torch.zeros(self.out_c)) if self.if_bias else None

        # Weight initialization
        nn.init.kaiming_uniform_(self.conv, a=math.sqrt(5))
        if self.if_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def pad_and_unfold(
        x: torch.Tensor, kernel_size: int, stride: int, padding: int, invalid_value: float
    ) -> torch.Tensor:
        """Helper function to pad and unfold a tensor."""
        x = F.pad(x, [padding] * 4, mode="constant", value=invalid_value)
        return F.unfold(x, (kernel_size, kernel_size), stride=stride)

    def forward(self, inp: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for sparse convolution.
        Args:
            inp: Input tensor of shape [B, C, H, W].
            mask: Mask tensor of shape [B, H, W], where invalid regions are marked with `mask_invalid_value`.
        Returns:
            out: Output tensor after sparse convolution.
            out_mask: Updated mask tensor of shape [B, 1, H_out, W_out].
        """
        assert inp.shape[2:] == mask.shape[1:], "Input and mask must have the same spatial dimensions."

        k, stride, padding = self.k, self.stride, self.padding
        h_in, w_in = inp.shape[2], inp.shape[3]
        center = (k - 1) // 2  # Adjusted center calculation for even/odd kernels
        print(f"inp.shape{inp.shape}")
        # Pad and unfold input and mask
        inp_padded = self.pad_and_unfold(inp, k, stride, padding, invalid_value=self.mask_invalid_value)
        mask_padded = self.pad_and_unfold(
            mask.unsqueeze(1), k, stride, padding, invalid_value=self.mask_invalid_value
        )
        print(f"inp_padded.shape{inp_padded.shape}")
        batch_size = inp.shape[0]

        # Reshape unfolded tensors
        inp_patched = inp_padded.view(batch_size, self.in_c, k, k, -1)  # [B, C, K, K, patches]
        mask_patched = mask_padded.view(batch_size, k, k, -1)           # [B, K, K, patches]

        # Create mask for valid regions
        mask_center_equals = (
            mask_patched == mask_patched[:, center : center + 1, center : center + 1, :]
        ).float()  # [B, K, K, patches]
        print(f"inp_patched.shape{inp_patched.shape}")
        print(f"mask_center_equals.shape{mask_center_equals.shape}")
        # Apply mask to input patches
        masked_input = inp_patched * mask_center_equals.unsqueeze(1)  # [B, C, K, K, patches]

        # Normalize by number of valid pixels
        m_norm = mask_center_equals.view(batch_size, k * k, -1)  # [B, K*K, patches]
        m_norm = k * k / torch.clamp(torch.sum(m_norm, dim=1), min=1e-8)  # Robust normalization

        # Perform convolution
        # out_unfolded = torch.einsum("bckp,ockl->bop", masked_input, self.conv)  # [B, C_out, patches]
         # Reshape tensors for einsum
        masked_input = masked_input.view(batch_size, self.in_c, -1, masked_input.size(-1))  # [B, C, K*K, patches]
        conv_weights = self.conv.view(self.out_c, self.in_c, -1)  # [out_c, in_c, K*K]

        # Corrected einsum equation
        out_unfolded = torch.einsum("bcxp,ocx->bop", masked_input, conv_weights)  # [B, C_out, patches]
        # Apply normalization
        out_unfolded_normalized = out_unfolded * m_norm.unsqueeze(1)  # [B, C_out, patches]

        # Reshape output to expected size
        h_out = (h_in + 2 * padding - (k - 1) - 1) // stride + 1
        w_out = (w_in + 2 * padding - (k - 1) - 1) // stride + 1
        out_ = out_unfolded_normalized.view(batch_size, self.out_c, h_out, w_out)  # [B, C_out, H_out, W_out]

        # Update mask
        mask_pooled = mask_patched[:, center, center, :]  # [B, patches]
        mask_out = mask_pooled.view(batch_size, 1, h_out, w_out)  # [B, 1, H_out, W_out]

        # Add bias
        if self.if_bias:
            out_ = out_ + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(out_)

        return out_, mask_out


class InstanceDeconv(nn.Module):
    """
    Deconvolution layer that upsamples features and applies sparse convolution.
    """
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bn: bool = False,
        mask_invalid_value: float = -1.0,
    ):
        super(InstanceDeconv, self).__init__()
        self.sparse_conv = InstanceConv(
            in_channel, out_channel, kernel_size, stride, padding, bn=bn, mask_invalid_value=mask_invalid_value
        )
        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x_guide: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for deconvolution.
        Args:
            x_guide: Feature tensor to upsample.
            mask: Mask tensor to upsample.
        Returns:
            x: Upsampled feature tensor after sparse convolution.
            m: Upsampled mask tensor.
        """
        x = self.up_sample(x_guide)
        m = F.interpolate(mask, scale_factor=2, mode="nearest")  # Upsample mask
        x, m = self.sparse_conv(x, m)
        return x, m


class CenterPool(nn.Module):
    """
    Pooling layer that downsamples masks by selecting the center pixel of each patch.
    """
    def __init__(self, kernel_size: int = 1, stride: int = 2, padding: int = 0):
        super(CenterPool, self).__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for center pooling.
        Args:
            mask: Mask tensor of shape [B, H, W].
        Returns:
            mask_out: Downsampled mask tensor.
        """
        k, stride, padding = self.k, self.stride, self.padding
        h_in, w_in = mask.shape[1], mask.shape[2]
        center = (k - 1) // 2  # Adjusted center calculation

        # Pad and unfold mask
        mask_padded = InstanceConv.pad_and_unfold(
            mask.unsqueeze(1), k, stride, padding, invalid_value=-1.0
        )
        mask_patched = mask_padded.view(mask.shape[0], k, k, -1)  # [B, K, K, patches]

        # Select center pixel
        mask_pooled = mask_patched[:, center, center, :]  # [B, patches]

        # Reshape to expected size
        h_out = (h_in + 2 * padding - (k - 1) - 1) // stride + 1
        w_out = (w_in + 2 * padding - (k - 1) - 1) // stride + 1
        mask_out = mask_pooled.view(mask.shape[0], 1, h_out, w_out)  # [B, 1, H_out, W_out]

        return mask_out


class SparseConvNet(nn.Module):
    """
    Sparse convolutional network for tasks like depth completion.
    """
    def __init__(self, in_channel ,activation=nn.ReLU):
        # in_channel = 1
        super(SparseConvNet, self).__init__()
        self.conv1 = InstanceConv(in_channel=in_channel, out_channel=80, kernel_size=3, stride=1, padding=1)
        self.seq1 = nn.Sequential(
            nn.BatchNorm2d(80),
            activation(inplace=True),
        )
        self.depth_pred = InstanceConv(in_channel=80, out_channel=1, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sparse convolutional network.
        Args:
            x: Input tensor of shape [B, C, H, W].
            mask: Mask tensor of shape [B, H, W].
        Returns:
            pred: Predicted output tensor (e.g., depth map).
        """
        out, out_mask = self.conv1(x, mask)
        out = self.seq1(out)
        print(out.shape, out_mask.shape)
        pred, _ = self.depth_pred(out, out_mask.squeeze(1))  # Squeeze mask to [B, H, W]
        return pred


if __name__ == "__main__":
    # Test the model with random inputs
    image = torch.randn(1, 19, 480, 640)  # Example input (batch, channels, height, width)
    mask = torch.randint(0, 2, (1, 480, 640)).float()  # Binary mask (0=invalid, 1=valid)
    print(mask.shape)
    model = SparseConvNet()
    output = model(image, mask)
    print(output.shape)  # Expected output shape: [1, 1, 480, 640]