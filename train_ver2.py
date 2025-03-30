import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
import os
from collections import defaultdict
import trimesh
from model import MTGOE
import torch
from pprint import pprint
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import masked_conv
from TeacherRenderer import TeacherRenderer
# from mmd import PoseEstimationModel

# import FSREDepth.networks.cma as cma

# class BOPDataset(Dataset):
#     def __init__(self, dataset_root, split='train', obj_id=1, transforms=None,
#     model_path='./models',
    
#     ):
#         self.dataset_root = dataset_root
#         self.split = split
#         self.obj_id = obj_id
#         self.transforms = transforms
#         self.model_path = model_path
#         self.data = []  # Store data in a list of dictionaries
#         self.cam_model_path = os.path.join(self.dataset_root,self.model_path, f'obj_{self.obj_id:06d}.ply')
#         self.model = self.load_ply_model(self.cam_model_path)
        
#         scene_dir = os.path.join(self.dataset_root, self.split)
#         for scene_id in sorted(os.listdir(scene_dir)):
#             scene_path = os.path.join(scene_dir, scene_id)
#             rgb_path = os.path.join(scene_path, 'rgb')
#             mask_path = os.path.join(scene_path, 'mask')
#             mask_visib_path = os.path.join(scene_path, 'mask_visib')
#             gt_path = os.path.join(scene_path, 'scene_gt.json')
#             cam_params_path = os.path.join(scene_path, 'scene_camera.json')

#             try:
#                 with open(gt_path, 'r') as f:
#                     gts = json.load(f)
#                 with open(cam_params_path, 'r') as f:
#                     cam_params = json.load(f)
#             except FileNotFoundError:
#                 print(f"Warning: scene_gt.json or scene_camera.json not found in {scene_path}. Skipping this scene.")
#                 continue # Skip this scene if the file is missing.
#             cam_k_values = {key: value["cam_K"] for key, value in cam_params.items()}
#             for img_id, objects in gts.items():  # Iterate through image IDs
#                 img_id = int(img_id)
#                 rgb_file = os.path.join(rgb_path, f"{img_id:06d}.png")
#                 for obj_data in objects: #Iterate through the objects
#                     if obj_data['obj_id'] == int(self.obj_id):  #Filter by object ID
#                         mask_file = os.path.join(mask_path, f"{img_id:06d}_000000.png")
#                         # print(mask_file)
#                         mask_visib_file = os.path.join(mask_visib_path, f"{img_id:06d}_000000.png")
#                         if os.path.exists(rgb_file) and os.path.exists(mask_file) and os.path.exists(mask_visib_file):
#                           self.data.append({
#                               'rgb_file': rgb_file,
#                               'mask_file': mask_file,
#                               'mask_visib_file': mask_visib_file,
#                               'obj_id': self.obj_id,
#                               'img_id': img_id,
#                               'cam_R_m2c': obj_data.get('cam_R_m2c', None), #Handle missing keys gracefully
#                               'cam_t_m2c': obj_data.get('cam_t_m2c', None), #Handle missing keys gracefully
#                               'cam_K': cam_k_values[str(img_id)],
#                           })
#                         else:
#                           print(f"Warning: Missing image or mask file for img_id {img_id}, obj_id {self.obj_id}. Skipping.")
                       


#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         try:
#             rgb = cv2.imread(sample['rgb_file'])
#             rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
#             mask = cv2.imread(sample['mask_file'], cv2.IMREAD_GRAYSCALE)
#             mask_visib = cv2.imread(sample['mask_visib_file'], cv2.IMREAD_GRAYSCALE)
#         except Exception as e:
#             print(f"Error loading image or mask: {e}, skipping sample {idx}.")
#             return None

#         rgb = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32))
#         mask = torch.from_numpy(mask.astype(np.float32))
#         mask_visib = torch.from_numpy(mask_visib.astype(np.float32))
#         cam_param = torch.tensor(sample['cam_K'], dtype=torch.float32)
#         cam_param = cam_param.view(3, 3)
#         #Handle missing pose data gracefully
#         cam_R_m2c = torch.tensor(sample['cam_R_m2c'], dtype=torch.float32) if sample['cam_R_m2c'] else None
#         cam_t_m2c = torch.tensor(sample['cam_t_m2c'], dtype=torch.float32) if sample['cam_t_m2c'] else None
        
#         #  --- Generate Ground Truth Data ---
#         h, w = rgb.shape[1:]
#         depth_map = self.generate_depth_map(sample['cam_R_m2c'], sample['cam_t_m2c'], cam_param, h, w)
#         segmentation_mask = self.generate_segmentation_mask(mask, h, w)  #Example of on-the-fly segmentation
#         vector_field = self.generate_field_vectors(cam_R_m2c, cam_t_m2c, self.model, cam_param, h, w)
#         # Placeholder for Field Vectors (replace with your calculation)
#         # field_vectors = np.zeros((h, w, 2), dtype=np.float32)
#         extens = self.compute_extent(self.cam_model_path)
#         return {
#             'rgb': rgb,
#             'mask': mask,
#             'mask_visib': mask_visib,
#             'depth_map': depth_map,
#             'segmentation_mask': segmentation_mask,
#             'vector_field': vector_field,
#             'obj_id': sample['obj_id'],
#             'img_id': sample['img_id'],
#             'cam_R_m2c': cam_R_m2c,
#             'cam_t_m2c': cam_t_m2c,
#             'cam_K': cam_param,
#             'extens': extens
            

#         }
#     @staticmethod
#     def load_ply_model(ply_path):
#         """Loads a 3D model from a PLY file."""
#         try:
#             model = trimesh.load_mesh(ply_path)
#             return model
#         except Exception as e:
#             print(f"Error loading PLY model from {ply_path}: {e}")
#             return None
#     @staticmethod 
#     def compute_extent(ply_file):
#         mesh = trimesh.load_mesh(ply_file)
#         min_corner = mesh.bounds[0]  # Minimum XYZ coordinates
#         max_corner = mesh.bounds[1]  # Maximum XYZ coordinates
#         extent = max_corner - min_corner  # Compute width, height, depth
#         return extent
    
#     def generate_depth_map(self, R, t, K, height, width):
#         """Generates a depth map from the object's pose and 3D model."""
#         if R is None or K is None:
#           print("Warning: cam_R_m2c or cam_K is None. Returning zero depth map.")
#           return np.zeros((height, width))

#         R = np.array(R).reshape(3, 3) # Reshape R into 3x3 matrix
#         t = np.array(t) #convert t to numpy array
#         K = np.array(K).reshape(3, 3) # Reshape K into 3x3 matrix
#         points_model = self.model.vertices
#         points_camera = (R @ points_model.T).T + t
#         points_image = (K @ points_camera.T).T
#         points_image[:, 0] /= points_image[:, 2]
#         points_image[:, 1] /= points_image[:, 2]
#         depth_map = np.zeros((height, width))
#         for point in points_image:
#             x, y, z = point
#             if 0 <= x < width and 0 <= y < height:
#                 depth_map[int(y), int(x)] = z
#         return depth_map

#     def generate_segmentation_mask(self, mask, h, w):
#         """Generates a segmentation mask.  Replace this with your actual logic."""
#         # This is a placeholder; you need to define the segmentation mask generation
#         # here based on your annotations.
#         return mask.reshape(h, w)
    
#     @staticmethod
#     def generate_field_vectors(cam_R_m2c, cam_t_m2c, model, K, height, width):

#         K = np.array(K).reshape(3, 3)  # This should be a 3x3 matrix

#         if cam_R_m2c is not None:
#             cam_R_m2c = np.array(cam_R_m2c).reshape(3, 3)
#         else:
#             raise ValueError("cam_R_m2c is None, cannot generate field vectors.")
        
#         # Ensure cam_t_m2c is a 3x1 vector
#         if cam_t_m2c is not None:
#             cam_t_m2c = np.array(cam_t_m2c).reshape(3,)
#         else:
#             raise ValueError("cam_t_m2c is None, cannot generate field vectors.")
        
#         # 3D model vertices (N x 3)
#         points_model = model.vertices  # Should be N x 3
        
#         # Ensure points_model is in the right shape (N, 3)
#         if points_model.shape[1] != 3:
#             raise ValueError("points_model must have shape (N, 3).")
        
#         # Transform model vertices to camera coordinates (using rotation + translation)
#         points_camera = np.dot(points_model, cam_R_m2c.T) + cam_t_m2c  # Should give N x 3
        
#         # Project to image plane (2D)
#         points_image = np.dot(points_camera, K.T)  # This will give N x 3
        
#         # Normalize points to get (x, y) image coordinates in the range [0, width] and [0, height]
#         points_image[:, 0] /= points_image[:, 2]
#         points_image[:, 1] /= points_image[:, 2]

#         # Initialize flow field (2D array with flow vectors)
#         flow_field = np.zeros((height, width, 2), dtype=np.float32)

#         # Map 3D points onto the 2D image plane and generate flow vectors
#         for point in points_image:
#             x, y, z = point
#             if 0 <= x < width and 0 <= y < height:
#                 # Example: flow vector from the image center to the projected point
#                 flow_field[int(y), int(x)] = [x - width // 2, y - height // 2]

#         return flow_field
class BOPDataset(Dataset):
    def __init__(self, dataset_root, split='train', obj_id=1, transforms=None):
        self.dataset_root = dataset_root
        self.split = split
        self.obj_id = obj_id
        self.transforms = transforms
        self.data = []  # Store data in a list of dictionaries
        
        # Load 3D model
        model_path = os.path.join(self.dataset_root, 'models', f'obj_{self.obj_id:06d}.ply')
        self.model_path = model_path
        if os.path.exists(model_path):
            self.model = self.load_ply_model(model_path)
        else:
            print(f"Warning: No model found at {model_path}")
            self.model = None
        
        # Load global camera parameters if available
        camera_path = os.path.join(self.dataset_root, 'camera.json')
        if os.path.exists(camera_path):
            with open(camera_path, 'r') as f:
                self.global_camera_params = json.load(f)
        else:
            self.global_camera_params = None
        
        # Path to the split (train or test)
        split_path = os.path.join(self.dataset_root, self.split)
        if not os.path.exists(split_path):
            print(f"Warning: Split path {split_path} does not exist")
            return
        
        # Iterate over scene folders in the split
        for scene_folder in os.listdir(split_path):
            scene_path = os.path.join(split_path, scene_folder)
            if not os.path.isdir(scene_path):
                continue
                
            # Load scene ground truth
            scene_gt_path = os.path.join(scene_path, 'scene_gt.json')
            if not os.path.exists(scene_gt_path):
                print(f"Warning: No scene_gt.json found in {scene_path}")
                continue
                
            with open(scene_gt_path, 'r') as f:
                scene_gt = json.load(f)
                
            # Load scene camera parameters
            scene_camera_path = os.path.join(scene_path, 'scene_camera.json')
            if not os.path.exists(scene_camera_path):
                print(f"Warning: No scene_camera.json found in {scene_path}")
                continue
                
            with open(scene_camera_path, 'r') as f:
                scene_camera = json.load(f)

            # Load gt info if available
            gt_info = {}
            gt_info_path = os.path.join(scene_path, 'scene_gt_info.json') 
            if os.path.exists(gt_info_path):
                with open(gt_info_path, 'r') as f:
                    gt_info = json.load(f)

            # Iterate over images in this scene
            for img_id_str, objects in scene_gt.items():
                img_id = int(img_id_str)
                
                # Check if camera parameters exist for this image
                if img_id_str not in scene_camera:
                    print(f"Warning: No camera parameters for image {img_id} in scene {scene_folder}")
                    continue
                
                cam_params = scene_camera[img_id_str]
                
                # Path to RGB image
                rgb_path = os.path.join(scene_path, 'rgb', f'{img_id:06d}.png')
                if not os.path.exists(rgb_path):
                    print(f"Warning: RGB image not found at {rgb_path}")
                    continue
                
                # Find our object of interest in this image
                for obj_idx, obj_info in enumerate(objects):
                    if obj_info['obj_id'] == self.obj_id:
                        # Paths to mask files
                        mask_path = os.path.join(scene_path, 'mask', f'{img_id:06d}_{obj_idx:06d}.png')
                        mask_visib_path = os.path.join(scene_path, 'mask_visib', f'{img_id:06d}_{obj_idx:06d}.png')
                        
                        # Check if mask files exist
                        mask_exists = os.path.exists(mask_path)
                        mask_visib_exists = os.path.exists(mask_visib_path)
                        
                        # Initialize bbox values
                        gt_center_x, gt_center_y, visib_fract = 0, 0, 0
                        
                        # Get bbox info if available
                        if img_id_str in gt_info:
                            img_gt_info = gt_info[img_id_str]
                            # Check if gt_info contains a list of objects
                            if isinstance(img_gt_info, list) and len(img_gt_info) > obj_idx:
                                obj_gt_info = img_gt_info[obj_idx]
                                if "bbox_obj" in obj_gt_info and "visib_fract" in obj_gt_info:
                                    bbox_obj = obj_gt_info["bbox_obj"]
                                    visib_fract = obj_gt_info["visib_fract"]
                                    gt_center_x = bbox_obj[0] + bbox_obj[2] / 2
                                    gt_center_y = bbox_obj[1] + bbox_obj[3] / 2
                            # If gt_info is a dictionary (for single object)
                            elif isinstance(img_gt_info, dict):
                                if "bbox_obj" in img_gt_info and "visib_fract" in img_gt_info:
                                    bbox_obj = img_gt_info["bbox_obj"]
                                    visib_fract = img_gt_info["visib_fract"]
                                    gt_center_x = bbox_obj[0] + bbox_obj[2] / 2
                                    gt_center_y = bbox_obj[1] + bbox_obj[3] / 2
                        
                        # Add sample to dataset
                        self.data.append({
                            'scene_id': scene_folder,
                            'img_id': img_id,
                            'obj_idx': obj_idx,
                            'obj_id': self.obj_id,
                            'rgb_path': rgb_path,
                            'mask_path': mask_path if mask_exists else None,
                            'mask_visib_path': mask_visib_path if mask_visib_exists else None,
                            'cam_R_m2c': obj_info.get('cam_R_m2c', None),
                            'cam_t_m2c': obj_info.get('cam_t_m2c', None),
                            'cam_K': cam_params.get('cam_K', None),
                            'gt_center_x': gt_center_x,
                            'gt_center_y': gt_center_y,
                            'visib_fract': visib_fract,
                        })
        
        print(f"Loaded {len(self.data)} samples for object {obj_id} in {split} set")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load RGB image
        rgb = cv2.imread(sample['rgb_path'])
        if rgb is None:
            raise ValueError(f"Could not load RGB image from {sample['rgb_path']}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load mask if available
        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        if sample['mask_path'] and os.path.exists(sample['mask_path']):
            mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
            # Normalize mask (sometimes BOP masks are 255 for object pixels)
            mask = (mask > 0).astype(np.uint8)
        
        # Load visible mask if available
        mask_visib = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        if sample['mask_visib_path'] and os.path.exists(sample['mask_visib_path']):
            mask_visib = cv2.imread(sample['mask_visib_path'], cv2.IMREAD_GRAYSCALE)
            # Normalize mask (sometimes BOP masks are 255 for object pixels)
            mask_visib = (mask_visib > 0).astype(np.uint8)
        
        # Convert to torch tensors
        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0  # Normalize to [0, 1]
        mask_tensor = torch.from_numpy(mask).float()
        mask_visib_tensor = torch.from_numpy(mask_visib).float()

        gt_center_x = sample['gt_center_x']
        gt_center_y = sample['gt_center_y']
        visib_fract = sample['visib_fract']

        gt_center_x_tensor = torch.tensor(gt_center_x, dtype=torch.float32)
        gt_center_y_tensor = torch.tensor(gt_center_y, dtype=torch.float32)
        visib_fract_tensor = torch.tensor(visib_fract, dtype=torch.float32)

        # Get camera parameters
        cam_K = np.array(sample['cam_K']).reshape(3, 3)
        cam_K_tensor = torch.from_numpy(cam_K).float()
        
        # Get pose if available
        cam_R_m2c = sample.get('cam_R_m2c', None)
        cam_t_m2c = sample.get('cam_t_m2c', None)
        
        cam_R_m2c_tensor = torch.from_numpy(np.array(cam_R_m2c).reshape(3, 3)).float() if cam_R_m2c else None
        cam_t_m2c_tensor = torch.from_numpy(np.array(cam_t_m2c)).float() if cam_t_m2c else None
        
        # Create depth map if the model and pose are available
        depth_map_tensor = None
        if self.model is not None and cam_R_m2c is not None and cam_t_m2c is not None:
            depth_map = self.generate_depth_map(cam_R_m2c, cam_t_m2c, cam_K, rgb.shape[0], rgb.shape[1])
            depth_map_tensor = torch.from_numpy(depth_map).float()
        
        # Create vector field if the model and pose are available
        vector_field_tensor = None
        if self.model is not None and cam_R_m2c is not None and cam_t_m2c is not None:
            vector_field = self.generate_field_vectors(cam_R_m2c, cam_t_m2c, self.model, cam_K, rgb.shape[0], rgb.shape[1])
            vector_field_tensor = torch.from_numpy(vector_field).float()
        
        # Calculate model extent/diameter for metrics
        extens = None
        if self.model is not None:
            extens = self.compute_extent(self.model_path)
            extens = torch.tensor(extens, dtype=torch.float32)
        
        # Create return dictionary
        result = {
            'rgb': rgb_tensor,
            'mask': mask_tensor,
            'mask_visib': mask_visib_tensor,
            'cam_K': cam_K_tensor,
            'obj_id': sample['obj_id'],
            'img_id': sample['img_id'],
            'gt_center_x': gt_center_x_tensor,
            'gt_center_y': gt_center_y_tensor,
            'visib_fract': visib_fract_tensor
        }
        
        # Add optional fields if available
        if cam_R_m2c_tensor is not None:
            result['cam_R_m2c'] = cam_R_m2c_tensor
        if cam_t_m2c_tensor is not None:
            result['cam_t_m2c'] = cam_t_m2c_tensor
        if depth_map_tensor is not None:
            result['depth_map'] = depth_map_tensor
        if vector_field_tensor is not None:
            result['vector_field'] = vector_field_tensor
        if mask_tensor is not None:
            result['segmentation_mask'] = mask_tensor  # Using the same mask for segmentation
        if extens is not None:
            result['extens'] = extens
        
        # Apply transforms if specified
        if self.transforms:
            result = self.transforms(result)
        
        return result
        
    @staticmethod
    def load_ply_model(ply_path):
        """Loads a 3D model from a PLY file."""
        try:
            model = trimesh.load_mesh(ply_path)
            return model
        except Exception as e:
            print(f"Error loading PLY model from {ply_path}: {e}")
            return None
            
    def generate_depth_map(self, R, t, K, height, width):
        """Generates a depth map from the object's pose and 3D model."""
        if self.model is None:
            return np.zeros((height, width), dtype=np.float32)
            
        # Convert inputs to numpy arrays if they're not already
        R = np.array(R).reshape(3, 3)
        t = np.array(t)
        K = np.array(K).reshape(3, 3)
        
        # Get model vertices
        points_model = np.array(self.model.vertices)
        
        # Transform to camera coordinates
        points_camera = np.dot(points_model, R.T) + t
        
        # Project to image coordinates
        points_image = np.dot(points_camera, K.T)
        
        # Normalize
        z = points_image[:, 2]
        points_image[:, 0] /= z
        points_image[:, 1] /= z
        
        # Create depth map
        depth_map = np.zeros((height, width), dtype=np.float32)
        
        # Filter points that are in the image bounds
        valid_idx = (points_image[:, 0] >= 0) & (points_image[:, 0] < width) & \
                    (points_image[:, 1] >= 0) & (points_image[:, 1] < height)
        
        # Convert to integer pixel coordinates
        x = np.round(points_image[valid_idx, 0]).astype(int)
        y = np.round(points_image[valid_idx, 1]).astype(int)
        z_valid = z[valid_idx]
        
        # Assign depth values (handle z-fighting with minimum depth)
        for i in range(len(x)):
            if depth_map[y[i], x[i]] == 0 or z_valid[i] < depth_map[y[i], x[i]]:
                depth_map[y[i], x[i]] = z_valid[i]
        
        return depth_map
    
    def generate_field_vectors(self, R, t, model, K, height, width):
        """Generates a vector field pointing to the object center."""
        # Create empty vector field
        vector_field = np.zeros((height, width, 2), dtype=np.float32)
        
        # Convert to numpy arrays
        R = np.array(R).reshape(3, 3)
        t = np.array(t)
        K = np.array(K).reshape(3, 3)
        
        # Compute object center in 3D
        vertices = np.array(model.vertices)
        center_3d = vertices.mean(axis=0)
        
        # Transform center to camera coordinates
        center_camera = np.dot(center_3d, R.T) + t
        
        # Project center to image
        center_image = np.dot(center_camera, K.T)
        center_image /= center_image[2]
        center_x, center_y = center_image[0], center_image[1]
        
        # Generate vector field (vectors pointing to center)
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Compute vectors (from pixel to center)
        dx = center_x - x_coords
        dy = center_y - y_coords
        
        # Normalize vectors
        magnitude = np.sqrt(dx**2 + dy**2) + 1e-10
        dx /= magnitude
        dy /= magnitude
        
        # Store in vector field
        vector_field[:, :, 0] = dx
        vector_field[:, :, 1] = dy
        
        return vector_field
    
    @staticmethod
    def compute_extent(model_path):
        """Compute the extent (width, height, depth) of the 3D model."""
        try:
            mesh = trimesh.load_mesh(model_path)
            min_corner = mesh.bounds[0]  # Minimum XYZ coordinates
            max_corner = mesh.bounds[1]  # Maximum XYZ coordinates
            extent = max_corner - min_corner  # Compute width, height, depth
            return extent
        except Exception as e:
            print(f"Error computing extent from {model_path}: {e}")
            return np.array([0.1, 0.1, 0.1])  # Default extent

if __name__ == '__main__':
    print(torch.cuda.memory_summary())

