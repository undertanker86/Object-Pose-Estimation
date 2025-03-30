try:
    from FSREDepth.utils.depth_utils import *
    from FSREDepth.networks.depth_decoder import DepthDecoder
    from FSREDepth.networks.multi_embedding import MultiEmbedding
    from FSREDepth.networks.seg_decoder import SegDecoder
except:
    import sys
    sys.path.append("..")
    from FSREDepth.utils.depth_utils import *
    from FSREDepth.networks.depth_decoder import DepthDecoder
    from FSREDepth.networks.multi_embedding import MultiEmbedding
    from FSREDepth.networks.seg_decoder import SegDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CMA(nn.Module):
    def __init__(self, num_ch_enc=None, scales=None, cma_layers=None, num_head=4, head_ratio=2, sgt=0.1, num_output_channels_depth=1, num_output_channels_seg=19):
        """
        Args:
            num_ch_enc (list): List of encoder channel dimensions.
            scales (list): Scales for depth decoder outputs.
            cma_layers (list): Layers where Cross-Modal Attention is applied.
            num_head (int): Number of attention heads in MultiEmbedding.
            head_ratio (int): Ratio for MultiEmbedding.
            sgt (float): Flag to enable/disable saving intermediate features.
            num_output_channels_depth (int): Number of output channels for depth decoder.
            num_output_channels_seg (int): Number of output channels for segmentation decoder.
        """
        super(CMA, self).__init__()
        

        # Hyperparameters
        self.scales = scales if scales is not None else [0, 1, 2, 3]
        self.cma_layers = cma_layers if cma_layers is not None else [2, 1]
        self.num_head = num_head
        self.head_ratio = head_ratio
        self.sgt = sgt

        # Decoder channel dimensions
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        in_channels_list = [32, 64, 128, 256, 16]

        # Initialize decoders
        self.depth_decoder = DepthDecoder(
            num_ch_enc=num_ch_enc,
            num_output_channels=num_output_channels_depth,
            scales=self.scales
        )
        self.seg_decoder = SegDecoder(
            num_ch_enc=num_ch_enc,
            num_output_channels=num_output_channels_seg,
            scales=[0]
        )

        # Cross-Modal Attention layers
        att_d_to_s = {}
        att_s_to_d = {}
        for i in self.cma_layers:
            att_d_to_s[str(i)] = MultiEmbedding(
                in_channels=in_channels_list[i],
                num_head=self.num_head,
                ratio=self.head_ratio
            )
            att_s_to_d[str(i)] = MultiEmbedding(
                in_channels=in_channels_list[i],
                num_head=self.num_head,
                ratio=self.head_ratio
            )
        self.att_d_to_s = nn.ModuleDict(att_d_to_s)
        self.att_s_to_d = nn.ModuleDict(att_s_to_d)

    def forward(self, input_features):
        depth_outputs = {}
        seg_outputs = {}
        x = input_features[-1]
        x_d = None
        x_s = None

        for i in range(4, -1, -1):
            # Depth decoder
            if x_d is None:
                x_d = self.depth_decoder.decoder[-2 * i + 8](x)
            else:
                x_d = self.depth_decoder.decoder[-2 * i + 8](x_d)

            # Segmentation decoder
            if x_s is None:
                x_s = self.seg_decoder.decoder[-2 * i + 8](x)
            else:
                x_s = self.seg_decoder.decoder[-2 * i + 8](x_s)

            # Upsample and concatenate skip connections
            x_d = [F.interpolate(x_d, scale_factor=2, mode='nearest')]
            x_s = [F.interpolate(x_s, scale_factor=2, mode='nearest')]

            if i > 0:
                x_d += [input_features[i - 1]]
                x_s += [input_features[i - 1]]

            x_d = torch.cat(x_d, dim=1)
            x_s = torch.cat(x_s, dim=1)

            x_d = self.depth_decoder.decoder[-2 * i + 9](x_d)
            x_s = self.seg_decoder.decoder[-2 * i + 9](x_s)

            # Apply Cross-Modal Attention
            if (i - 1) in self.cma_layers:
                x_d_att = self.att_d_to_s[str(i - 1)](x_d, x_s)
                x_s_att = self.att_s_to_d[str(i - 1)](x_s, x_d)
                x_d = x_d_att
                x_s = x_s_att

            # Save intermediate features if sgt is enabled
            if self.sgt:
                depth_outputs[('d_feature', i)] = x_d
                seg_outputs[('s_feature', i)] = x_s

            # Generate outputs at specified scales
            if i in self.scales:
                outs = self.depth_decoder.decoder[10 + i](x_d)
                depth_outputs[("disp", i)] = torch.sigmoid(outs[:, :1, :, :])
                if i == 0:
                    outs = self.seg_decoder.decoder[10 + i](x_s)
                    seg_outputs[("seg_logits", i)] = outs[:, :19, :, :]

        return depth_outputs, seg_outputs

if __name__ == '__main__':
    # Example input feature maps
    input1 = torch.randn(1, 64, 240, 320)
    input2 = torch.randn(1, 256, 120, 160)
    input3 = torch.randn(1, 512, 60, 80)
    input4 = torch.randn(1, 1024, 30, 40)
    input5 = torch.randn(1, 2048, 15, 20)
    input_features = [input1, input2, input3, input4, input5]

    # Encoder channel dimensions
    num_ch_enc = [64, 256, 512, 1024, 2048]

    # Initialize CMA with custom hyperparameters
    cma = CMA(
        num_ch_enc=num_ch_enc,
        scales=[0, 1, 2, 3],
        cma_layers=[2, 1],
        num_head=4,
        head_ratio=2,
        sgt=0.1
    )

    # Forward pass
    depth_outputs, seg_outputs = cma(input_features)

    # Print shapes of outputs
    for key, value in depth_outputs.items():
        print(f"Depth Output {key}: {value.shape}")

    for key, value in seg_outputs.items():
        print(f"Segmentation Output {key}: {value.shape}")