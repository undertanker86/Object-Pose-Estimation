import torch
import sys
sys.path.append("./posecnn")
import posecnn_cuda

# Example: Running a function from posecnn_cuda
bottom_prob = torch.randn((1, 3, 224, 224), device="cuda")  # Example tensor
bottom_label = torch.randint(0, 10, (1, 3, 224, 224), device="cuda")
bottom_rand = torch.rand((1, 3, 224, 224), device="cuda")

threshold = 0.5
sample_percentage = 0.2

# Assuming `hard_label_cuda_forward` is exposed in `posecnn_cuda`
output = posecnn_cuda.hard_label_forward(threshold, sample_percentage, bottom_prob, bottom_label, bottom_rand)
print(output)
