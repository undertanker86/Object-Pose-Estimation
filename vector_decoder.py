import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorFieldDecoder(nn.Module):
    def __init__(self, in_channels_list, num_units=512, out_channels=1, target_resolution=(480, 640)):
        """
        Args:
            in_channels_list (list): List of input channels for each feature map.
            num_units (int): Number of units in intermediate layers.
            out_channels (int): Number of output channels for the vector field.
            target_resolution (tuple): Target spatial resolution (height, width) for the output.
        """
        super(VectorFieldDecoder, self).__init__()

        # Encoder feature embedding layers
        self.embed_layers = nn.ModuleList([
            self._make_conv(in_channels, num_units // (2 ** i), kernel_size=1)
            for i, in_channels in enumerate(reversed(in_channels_list))
        ])

        # Upsampling layers
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                self._make_conv(num_units // (2 ** i), num_units // (2 ** (i + 1)), kernel_size=1)  # Adjust channels
            )
            for i in range(len(in_channels_list) - 1)
        ])

        # Final upsampling to match target resolution
        self.final_upsample = nn.Upsample(size=target_resolution, mode='bilinear', align_corners=False)

        # Final convolution for vector field prediction
        self.final_conv = self._make_conv(num_units // (2 ** (len(in_channels_list) - 1)), out_channels, kernel_size=1, relu=False)

    def _make_conv(self, in_channels, out_channels, kernel_size=3, relu=True):
        """Helper function to create convolutional blocks."""
        padding = (kernel_size - 1) // 2
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, input_features):
        """
        Args:
            input_features (list of tensors): Multi-scale feature maps from the encoder.

        Returns:
            Tensor: Predicted vector field tensor with shape [B, out_channels, H, W].
        """
        # Reverse input features (from deepest to shallowest)
        input_features = input_features[::-1]

        # Process deepest feature first
        x = self.embed_layers[0](input_features[0])

        # Progressive upsampling and feature fusion
        for i in range(1, len(input_features)):
            x = self.upsample_layers[i - 1](x)  # Upsample and adjust channels
            skip = self.embed_layers[i](input_features[i])  # Embed skip connection
            x = x + skip  # Fuse features (now dimensions match)

        # Final upsampling to match target resolution
        x = self.final_upsample(x)

        # Final output
        vector_field = self.final_conv(x)
        return vector_field


# Example Usage
if __name__ == "__main__":
    # Input feature maps (example dimensions)
    input1 = torch.randn(1, 64, 240, 320)   # Shallow features
    input2 = torch.randn(1, 256, 120, 160)
    input3 = torch.randn(1, 512, 60, 80)    # Mid-level features
    input4 = torch.randn(1, 1024, 30, 40)   # High-level features
    input5 = torch.randn(1, 2048, 15, 20)   # Deepest features

    input_features = [input1, input2, input3, input4, input5]

    # Initialize decoder
    in_channels_list = [64, 256, 512, 1024, 2048]  # Channels for each feature map
    target_resolution = (480, 640)  # Original image size
    decoder = VectorFieldDecoder(
        in_channels_list=in_channels_list,
        num_units=512,
        out_channels=1,
        target_resolution=target_resolution
    )

    # Forward pass
    vector_field = decoder(input_features)
    print("Vector Field Shape:", vector_field.shape)  # Expected: [1, 1, 480, 640]