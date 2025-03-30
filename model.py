import sys
sys.path.append("./FSRE-Depth")
from FSREDepth.networks.cma import CMA
# from FSREDepth.networks.depth_decoder import DepthDecoder
# from FSREDepth.networks.resnet_encoder import ResnetEncoder
# from FSREDepth.networks.seg_decoder import SegDecoder
from vector_decoder import VectorFieldDecoder
from mmd import MMD
from masked_conv import SparseConvNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import tomli

import timm
import click
from Dpobjj.HoughVoter import HoughVoterPyTorch

with open("config.toml", "rb") as f:
    config = tomli.load(f)
print(config)

class MTGOE(nn.Module):
    def __init__(self, image_height, image_width, in_channel=1):
        super(MTGOE, self).__init__()
        self.encoder = timm.create_model('tf_mobilenetv3_small_minimal_100.in1k',
                                         pretrained=True,
                                         features_only = True)
        self.cma = CMA(
            num_ch_enc=config["cma"]["num_ch_enc"],
            scales=config["cma"]["scales"],
            cma_layers=config["cma"]["cma_layers"],
            num_head=config["cma"]["num_head"],
            head_ratio=config["cma"]["head_ratio"],
            sgt=config["cma"]["sgt"]
        )
        self.vector_decoder = VectorFieldDecoder(
            in_channels_list=config["vector_decoder"]["in_channels_list"],
        )
        self.mask_conv = SparseConvNet(in_channel=in_channel)
        self.mmd = MMD
        self.HoughVoter = HoughVoterPyTorch(image_height, image_width)
    
    @staticmethod
    def get_binary_mask(segmentation_map, threshold=0.4):
        """
        Convert softmax probabilities to a binary mask.
        Args:
            segmentation_map (Tensor): Softmax probabilities [B, C, H, W].
            threshold (float): Threshold for binarization.
        Returns:
            Tensor: Binary mask [B, H, W].
        """
        # Apply softmax to get probabilities
        probs = F.softmax(segmentation_map, dim=1)
        
        # Select the probabilities for the first class (object of interest)
        object_probs = probs[:, 0]  # Shape: [B, H, W]
        
        # Apply threshold to create binary mask
        binary_mask = (object_probs > threshold).float()  # Shape: [B, H, W]
        
        return binary_mask
    
    @staticmethod
    def get_segmentation_map(segmentation_logits, threshold=0.4):
        """
        Convert segmentation logits or probabilities to a binary mask.
        Args:
            segmentation_logits (Tensor): Logits or probabilities from the segmentation model [H, W] or [C, H, W].
            threshold (float): Threshold for binarization.
        Returns:
            Tensor: Binary segmentation map [H, W].
        """
        # If segmentation_logits has multiple channels (e.g., logits for each class), select the first class
        if segmentation_logits.dim() == 3:  # Shape: [C, H, W]
            # Apply softmax to convert logits to probabilities
            probs = torch.softmax(segmentation_logits, dim=0)
            # Select the probabilities for the first class (object of interest)
            object_probs = probs[0]  # Shape: [H, W]
        elif segmentation_logits.dim() == 2:  # Shape: [H, W] (already probabilities or single-class logits)
            object_probs = segmentation_logits
        else:
            raise ValueError(f"Unsupported input dimensions: {segmentation_logits.shape}")

        # Apply threshold to create binary mask
        binary_mask = (object_probs > threshold).float()  # Shape: [H, W]

        return binary_mask


    def forward(self, x, fx, fy, px, py):
        features = self.encoder(x)
        depth_features, seg_features = self.cma(features)
        vector_field = self.vector_decoder(features)
        
        # Get segmentation map and depth map
        segmentation_map = seg_features[('seg_logits', 0)]  # Bx19xHxW
        depth_map = depth_features[('disp', 0)]  # Bx1xHxW
        
        # Get binary mask from segmentation
        mask = self.get_binary_mask(segmentation_map)
        
        # Apply mask-based convolution on depth map
        refined_depth = self.mask_conv(depth_map, mask)
        
        # Get segmentation map for a single instance
        instance_seg_map = self.get_segmentation_map(segmentation_map[0])
        
        # Cast votes using Hough Voter
        vote_map, centers, translations = self.HoughVoter.cast_votes(
            instance_seg_map,
            vector_field[0], 
            refined_depth[0],
            fx=fx, fy=fy, px=px, py=py
        )
        
        #Khởi tạo mmd_instance và chuyển nó đến cùng thiết bị
        device = refined_depth.device
        mmd_instance = MMD(
            depth_channels=refined_depth.shape[1],
            seg_channels=segmentation_map.shape[1],
            vector_channels=vector_field.shape[1]
        ).to(device)
        
        # Apply MMD
        fused_features = mmd_instance(refined_depth, segmentation_map, vector_field)
        
        return vote_map, centers, translations, refined_depth, segmentation_map, vector_field, fused_features

if __name__ == '__main__':
    model = MTGOE(480,640)
    x = torch.randn(1,3,480,640)
    out=model(x, fx=500, fy=500, px=320, py=240)
    for i in out:
        print(i.shape)