import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
import os
from collections import defaultdict
import trimesh
import Dpobjj.vote as hough_voting_cuda_1
import FSREDepth.networks.depth_decoder as depth_decoder
import FSREDepth.networks.resnet_encoder as resnet_encoder
import FSREDepth.networks.seg_decoder as seg_decoder
import Dpobjj.decoderVote as hough_voting_cpu
import torch
from pprint import pprint
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
class BOPDataset(Dataset):
    def __init__(self, dataset_root, split='train', obj_id=1, transforms=None,
    model_path='./models',
    
    ):
        self.dataset_root = dataset_root
        self.split = split
        self.obj_id = obj_id
        self.transforms = transforms
        self.model_path = model_path
        self.data = []  # Store data in a list of dictionaries
        self.cam_model_path = os.path.join(self.dataset_root,self.model_path, f'obj_{self.obj_id:06d}.ply')
        self.model = self.load_ply_model(self.cam_model_path)
        
        scene_dir = os.path.join(self.dataset_root, self.split)
        for scene_id in sorted(os.listdir(scene_dir)):
            scene_path = os.path.join(scene_dir, scene_id)
            rgb_path = os.path.join(scene_path, 'rgb')
            mask_path = os.path.join(scene_path, 'mask')
            mask_visib_path = os.path.join(scene_path, 'mask_visib')
            gt_path = os.path.join(scene_path, 'scene_gt.json')
            cam_params_path = os.path.join(scene_path, 'scene_camera.json')

            try:
                with open(gt_path, 'r') as f:
                    gts = json.load(f)
                with open(cam_params_path, 'r') as f:
                    cam_params = json.load(f)
            except FileNotFoundError:
                print(f"Warning: scene_gt.json or scene_camera.json not found in {scene_path}. Skipping this scene.")
                continue # Skip this scene if the file is missing.
            cam_k_values = {key: value["cam_K"] for key, value in cam_params.items()}
            for img_id, objects in gts.items():  # Iterate through image IDs
                img_id = int(img_id)
                rgb_file = os.path.join(rgb_path, f"{img_id:06d}.png")
                for obj_data in objects: #Iterate through the objects
                    if obj_data['obj_id'] == int(self.obj_id):  #Filter by object ID
                        mask_file = os.path.join(mask_path, f"{img_id:06d}_000000.png")
                        # print(mask_file)
                        mask_visib_file = os.path.join(mask_visib_path, f"{img_id:06d}_000000.png")
                        if os.path.exists(rgb_file) and os.path.exists(mask_file) and os.path.exists(mask_visib_file):
                          self.data.append({
                              'rgb_file': rgb_file,
                              'mask_file': mask_file,
                              'mask_visib_file': mask_visib_file,
                              'obj_id': self.obj_id,
                              'img_id': img_id,
                              'cam_R_m2c': obj_data.get('cam_R_m2c', None), #Handle missing keys gracefully
                              'cam_t_m2c': obj_data.get('cam_t_m2c', None), #Handle missing keys gracefully
                              'cam_K': cam_k_values[str(img_id)],
                          })
                        else:
                          print(f"Warning: Missing image or mask file for img_id {img_id}, obj_id {self.obj_id}. Skipping.")
                       


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        try:
            rgb = cv2.imread(sample['rgb_file'])
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(sample['mask_file'], cv2.IMREAD_GRAYSCALE)
            mask_visib = cv2.imread(sample['mask_visib_file'], cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f"Error loading image or mask: {e}, skipping sample {idx}.")
            return None

        rgb = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        mask_visib = torch.from_numpy(mask_visib.astype(np.float32))
        cam_param = torch.tensor(sample['cam_K'], dtype=torch.float32)
        cam_param = cam_param.view(3, 3)
        #Handle missing pose data gracefully
        cam_R_m2c = torch.tensor(sample['cam_R_m2c'], dtype=torch.float32) if sample['cam_R_m2c'] else None
        cam_t_m2c = torch.tensor(sample['cam_t_m2c'], dtype=torch.float32) if sample['cam_t_m2c'] else None
        
        #  --- Generate Ground Truth Data ---
        h, w = rgb.shape[1:]
        depth_map = self.generate_depth_map(sample['cam_R_m2c'], sample['cam_t_m2c'], cam_param, h, w)
        segmentation_mask = self.generate_segmentation_mask(mask, h, w)  #Example of on-the-fly segmentation
        vector_field = self.generate_field_vectors(cam_R_m2c, cam_t_m2c, self.model, cam_param, h, w)
        # Placeholder for Field Vectors (replace with your calculation)
        # field_vectors = np.zeros((h, w, 2), dtype=np.float32)
        extens = self.compute_extent(self.cam_model_path)
        return {
            'rgb': rgb,
            'mask': mask,
            'mask_visib': mask_visib,
            'depth_map': depth_map,
            'segmentation_mask': segmentation_mask,
            'vector_field': vector_field,
            'obj_id': sample['obj_id'],
            'img_id': sample['img_id'],
            'cam_R_m2c': cam_R_m2c,
            'cam_t_m2c': cam_t_m2c,
            'cam_K': cam_param,
            'extens': extens
            

        }
    @staticmethod
    def load_ply_model(ply_path):
        """Loads a 3D model from a PLY file."""
        try:
            model = trimesh.load_mesh(ply_path)
            return model
        except Exception as e:
            print(f"Error loading PLY model from {ply_path}: {e}")
            return None
    @staticmethod 
    def compute_extent(ply_file):
        mesh = trimesh.load_mesh(ply_file)
        min_corner = mesh.bounds[0]  # Minimum XYZ coordinates
        max_corner = mesh.bounds[1]  # Maximum XYZ coordinates
        extent = max_corner - min_corner  # Compute width, height, depth
        return extent
    
    def generate_depth_map(self, R, t, K, height, width):
        """Generates a depth map from the object's pose and 3D model."""
        if R is None or K is None:
          print("Warning: cam_R_m2c or cam_K is None. Returning zero depth map.")
          return np.zeros((height, width))

        R = np.array(R).reshape(3, 3) # Reshape R into 3x3 matrix
        t = np.array(t) #convert t to numpy array
        K = np.array(K).reshape(3, 3) # Reshape K into 3x3 matrix
        points_model = self.model.vertices
        points_camera = (R @ points_model.T).T + t
        points_image = (K @ points_camera.T).T
        points_image[:, 0] /= points_image[:, 2]
        points_image[:, 1] /= points_image[:, 2]
        depth_map = np.zeros((height, width))
        for point in points_image:
            x, y, z = point
            if 0 <= x < width and 0 <= y < height:
                depth_map[int(y), int(x)] = z
        return depth_map

    def generate_segmentation_mask(self, mask, h, w):
        """Generates a segmentation mask.  Replace this with your actual logic."""
        # This is a placeholder; you need to define the segmentation mask generation
        # here based on your annotations.
        return mask.reshape(h, w)
    
    @staticmethod
    def generate_field_vectors(cam_R_m2c, cam_t_m2c, model, K, height, width):

        K = np.array(K).reshape(3, 3)  # This should be a 3x3 matrix

        if cam_R_m2c is not None:
            cam_R_m2c = np.array(cam_R_m2c).reshape(3, 3)
        else:
            raise ValueError("cam_R_m2c is None, cannot generate field vectors.")
        
        # Ensure cam_t_m2c is a 3x1 vector
        if cam_t_m2c is not None:
            cam_t_m2c = np.array(cam_t_m2c).reshape(3,)
        else:
            raise ValueError("cam_t_m2c is None, cannot generate field vectors.")
        
        # 3D model vertices (N x 3)
        points_model = model.vertices  # Should be N x 3
        
        # Ensure points_model is in the right shape (N, 3)
        if points_model.shape[1] != 3:
            raise ValueError("points_model must have shape (N, 3).")
        
        # Transform model vertices to camera coordinates (using rotation + translation)
        points_camera = np.dot(points_model, cam_R_m2c.T) + cam_t_m2c  # Should give N x 3
        
        # Project to image plane (2D)
        points_image = np.dot(points_camera, K.T)  # This will give N x 3
        
        # Normalize points to get (x, y) image coordinates in the range [0, width] and [0, height]
        points_image[:, 0] /= points_image[:, 2]
        points_image[:, 1] /= points_image[:, 2]

        # Initialize flow field (2D array with flow vectors)
        flow_field = np.zeros((height, width, 2), dtype=np.float32)

        # Map 3D points onto the 2D image plane and generate flow vectors
        for point in points_image:
            x, y, z = point
            if 0 <= x < width and 0 <= y < height:
                # Example: flow vector from the image center to the projected point
                flow_field[int(y), int(x)] = [x - width // 2, y - height // 2]

        return flow_field
def vector_field_loss(pred_vectors, gt_vectors, mask):
    """
    Computes the vector field loss (Lvf) using Smooth L1 loss.

    Args:
        pred_vectors (torch.Tensor): Predicted vector field (B, 2, H, W)
        gt_vectors (torch.Tensor): Ground truth vector field (B, 2, H, W)
        mask (torch.Tensor): Binary mask for object pixels (B, 1, H, W)

    Returns:
        torch.Tensor: Loss scalar
    """
    gt_vectors = gt_vectors.permute(0, 3, 1, 2)
    # Ensure the loss is only computed for object pixels
    mask = mask.float()
    loss = F.smooth_l1_loss(pred_vectors * mask, gt_vectors * mask, reduction='sum')

    num_object_pixels = mask.sum() + 1e-6  # Avoid division by zero
    return loss / num_object_pixels

def train_model():
    # Setup device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset
    data = BOPDataset(dataset_root='./data/lmo', split='train', obj_id=1)

    dataloader = DataLoader(data, batch_size=1, shuffle=True)

    # Initialize the model components
    model_encoder = resnet_encoder.ResnetEncoder(num_layers=18, pretrained=True).to(device)
    model_depth_decoder = depth_decoder.DepthDecoder(num_ch_enc=model_encoder.num_ch_enc, scales=range(4)).to(device)
    model_seg_decoder = seg_decoder.SegDecoder(num_ch_enc=model_encoder.num_ch_enc, scales=[0]).to(device)
    # model_vector_field = hough_voting_cuda_1.EncoderDecoder().to(device)
    model_vector_field = hough_voting_cpu.HoughVotingDecoder(8, 480, 640).to(device)
    # Combine the models in a dictionary (or directly integrate if needed)
    model = {
        'encoder': model_encoder,
        'depth_decoder': model_depth_decoder,
        'seg_decoder': model_seg_decoder,
        'vector_field': model_vector_field
    }

    # Optimizer
    optimizer = optim.Adam(
        list(model_encoder.parameters()) + 
        list(model_depth_decoder.parameters()) + 
        list(model_seg_decoder.parameters()) + 
        list(model_vector_field.parameters()), 
        lr=1e-4
    )

    num_epochs = 20

    # Training Loop
    for epoch in range(num_epochs):
        model_encoder.train()
        model_depth_decoder.train()
        model_seg_decoder.train()
        model_vector_field.train()

        total_loss = 0
        for batch in dataloader:
         
            # Get data from the batch
            rgb = batch['rgb'].to(device)
            mask = batch['mask'].to(device)
            depth_map = batch['depth_map'].to(device)
            segmentation_mask = batch['segmentation_mask'].to(device)
            field_vectors = batch['vector_field'].to(device)
            info_cams = batch['cam_K'].to(device)
            extens = batch['extens'].to(device)
            print('info_cams: ', info_cams)
            fx, fy = info_cams[0, 0, 0], info_cams[0, 1, 1]
            px, py = info_cams[0, 0, 2], info_cams[0, 1, 2]


            # info_cams = torch.tensor(info_cams, dtype=torch.float32)
            # extens = torch.tensor(extens, dtype=torch.float32)
            # field_vectors = torch.tensor(field_vectors, dtype=torch.float32)
            # segmentation_mask = torch.tensor(segmentation_mask, dtype=torch.float32)
            # depth_map = torch.tensor(depth_map, dtype=torch.float32)
            # mask = torch.tensor(mask, dtype=torch.float32)
            # rgb = torch.tensor(rgb, dtype=torch.float32)

            # Forward pass through the encoder
            
            print('segmentation_mask: ', segmentation_mask.shape)


            features, features_2 = model['encoder'](rgb)
            print('features: ', features[-1].shape)

            # Decoder forward passes
            depth = model['depth_decoder'](features)
            for key, value in depth.items():
                print(key, value.shape)
            seg, bottom_label = model['seg_decoder'](features)
            # print('bottom_label: ', bottom_label.shape)
            depth_tensor = depth['disp', 0]
            segmenta= seg['seg_logits', 0]
            # print('features: ', features.shape)
            # print('info_cams: ', info_cams.shape)
            # print('extens: ', extens.shape)

            # Assuming the vector field decoder takes in the features and produces predictions
            bottom_vertex, bottom_label, vote_map, centers, translations = model['vector_field'](features,depth_tensor, segmenta, fx, fy , px, py )

            # Calculate the losses for each task
            depth_loss = torch.mean((depth_tensor - depth_map) ** 2)
            seg_loss = torch.mean((segmenta - segmentation_mask) ** 2)
            print('mask: ', mask.shape)
            print('bottom_vertex: ', bottom_vertex.shape)
            print('field_vectors: ', field_vectors.shape)
            bottom_vertex = F.interpolate(bottom_vertex, size=(480, 640), mode='bilinear', align_corners=False)
            bottom_vertex = bottom_vertex[:, :2, :, :]  # Use first 2 channels (x, y)
            # Vector field loss (you can define it based on your task, here using a placeholder)
            
            vector_field_loss1= vector_field_loss(bottom_vertex, field_vectors, mask)
            print('vector_field_loss1: ', vector_field_loss1)
            # Total loss combining all tasks
            total_loss = depth_loss + seg_loss + vector_field_loss1
            # print(torch.cuda.memory_summary())
            # Backpropagation and optimization
            optimizer.zero_grad()
            # print(torch.cuda.memory_summary())
            total_loss.backward()
            optimizer.step()
          

            # Print loss for this batch (optional, for debugging)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item()}")

        # Optionally, save the model after each epoch
        # torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    print(torch.cuda.memory_summary())
    train_model()
    # data = BOPDataset(dataset_root='./data/lmo', split='train', obj_id=1)
    # print(len(data))
    # import matplotlib.pyplot as plt
    # # Assuming `data` is your BOPDataset object and `sample` is a sample from it
    # sample = data[1]

    # # Extract the necessary data
    # cam_R_m2c = sample['cam_R_m2c']
    # cam_t_m2c = sample['cam_t_m2c']
    # model = data.model  # 3D model (loaded earlier in the dataset)
    # K = sample['cam_K']  # Camera intrinsic matrix
    # K = np.array(K).reshape(3, 3)  # This should be a 3x3 matrix

    # # Get the image dimensions
    # height, width = sample['rgb'].shape[1:]

    # # Generate the flow field
    # flow_field = data.generate_field_vectors(cam_R_m2c, cam_t_m2c, model, K, height, width)

    # # Optionally display the flow field (vectors as arrows)
    # import matplotlib.pyplot as plt
    # plt.quiver(flow_field[:,:,0], flow_field[:,:,1], scale=10)
    # plt.title('Flow Field')
    # plt.show()



    # segmentation_mask = sample['segmentation_mask'].numpy()
    # flow_field = data.generate_field_vectors(sample[], cam_t_m2c, model, K, height=480, width=640)
    # Display the segmentation mask using matplotlib
    # plt.imshow(segmentation_mask)
    # plt.title('Segmentation Mask')
    # plt.show()

