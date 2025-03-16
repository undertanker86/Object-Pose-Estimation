import Dpobjj.vote as EncoderDecoder
import FSREDepth.networks.depth_decoder as depth_decoder
import FSREDepth.networks.resnet_encoder as resnet_encoder
import FSREDepth.networks.seg_decoder as seg_decoder
import torch
from pprint import pprint

model = {}
model['encoder'] = resnet_encoder.ResnetEncoder(num_layers=18, pretrained=False)

model['depth_decoder'] = depth_decoder.DepthDecoder(num_ch_enc=model['encoder'].num_ch_enc, scales=range(4))
model['seg_decoder'] = seg_decoder.SegDecoder(num_ch_enc=model['encoder'].num_ch_enc, scales=[0])

# print(model)


image = torch.randn(1, 3, 256, 256)
features = model['encoder'](image)
depth = model['depth_decoder'](features)
seg = model['seg_decoder'](features)

print(depth)


