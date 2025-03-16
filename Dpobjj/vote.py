import torch
import hough_voting_cuda
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define tensor dimensions
batch_size = 1
num_classes = 3
height, width = 480, 640
VERTEX_CHANNELS = 3
num_meta_data = 10
MAX_ROI = 1000

# Create dummy input tensors
bottom_label = torch.randint(0, num_classes, (batch_size, height, width), dtype=torch.int32, device=device)
bottom_vertex = torch.randn(batch_size, num_classes * VERTEX_CHANNELS, height, width, device=device)
bottom_meta_data = torch.randn(batch_size, num_meta_data, device=device)
extents = torch.randn(num_classes, 3, device=device)

# Define function parameters
is_train = 1
skip_pixels = 5
labelThreshold = 10
inlierThreshold = 0.1
votingThreshold = 0.5
perThreshold = 0.2

# print(torch.cuda.memory_summary())





# Call the CUDA function
top_box_final, top_pose_final = hough_voting_cuda.hough_voting_cuda_forward(
    bottom_label, bottom_vertex, bottom_meta_data, extents,
    is_train, skip_pixels, labelThreshold, inlierThreshold, votingThreshold, perThreshold
)

# Print output
print("Top Box Final Shape:", top_box_final.shape)
print("Top Pose Final Shape:", top_pose_final.shape)
print("Top Box Final Data:", top_box_final)
print("Top Pose Final Data:", top_pose_final)

# print(torch.cuda.memory_summary())
import torch
import torch.nn as nn
 # The compiled CUDA extension

class HoughVotingLayer(nn.Module):
    def __init__(self, skip_pixels=10, label_threshold=5, inlier_threshold=0.5, 
                 voting_threshold=0.3, per_threshold=0.1, is_train=True):
        """ This is not gradient enabled layer
        """
        super(HoughVotingLayer, self).__init__()
        self.skip_pixels = skip_pixels
        self.label_threshold = label_threshold
        self.inlier_threshold = inlier_threshold
        self.voting_threshold = voting_threshold
        self.per_threshold = per_threshold
        self.is_train = is_train

    def forward(self, bottom_label, bottom_vertex, bottom_meta_data, extents):
        bottom_label = bottom_label.to(torch.int)  # Convert bottom_label to Int
        # bottom_vertex = bottom_vertex.to(torch.int)  # Convert bottom_vertex to Int
        # bottom_label = bottom_label.to(torch.float)  # Convert to Float
        bottom_vertex = bottom_vertex.to(torch.float)  # Convert to Float
        bottom_meta_data = bottom_meta_data.to(torch.float)  # Convert to Float
        extents = extents.to(torch.float)  # Convert to Float

        return hough_voting_cuda.hough_voting_cuda_forward(
            bottom_label, bottom_vertex, bottom_meta_data, extents,
            int(self.is_train), self.skip_pixels, self.label_threshold,
            self.inlier_threshold, self.voting_threshold, self.per_threshold
        )

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        
        # Decoder (Feature Upsampling + Hough Voting)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 20, kernel_size=3, stride=1, padding=1),  # Assume 20 output channels
            nn.ReLU()
        )

        # Hough Voting Layer
        self.hough_voting = HoughVotingLayer()

    def forward(self, x, bottom_label, bottom_meta_data, extents):
        # Encoder
        # x = self.encoder(x)  # Feature extraction

        # Decoder
        bottom_vertex = self.decoder(x)  # Dense regression (e.g., vertex field)

        # Apply Hough Voting
        top_box, top_pose = self.hough_voting(bottom_label, bottom_vertex, bottom_meta_data, extents)
        # top_box = 0
        # top_pose = 0
        return top_box, top_pose, bottom_vertex


