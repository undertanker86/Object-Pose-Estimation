import torch
import numpy as np
import matplotlib.pyplot as plt

class HoughVoterPyTorch:
    def __init__(self, image_height, image_width, vote_threshold=0.5, device="cuda"):
        """
        Initialize the Hough voting system with PyTorch CUDA support
        
        Args:
            image_height: Height of the input image
            image_width: Width of the input image
            vote_threshold: Threshold for considering a vote valid
            device: Computing device ('cuda' or 'cpu')
        """
        self.height = image_height
        self.width = image_width
        self.vote_threshold = vote_threshold
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"Using device: {self.device}")
        
        # Camera intrinsic parameters (can be replaced with actual values)
        self.fx = torch.tensor(500.0, device=self.device)  # focal length x
        self.fy = torch.tensor(500.0, device=self.device)  # focal length y 
        self.px = torch.tensor(self.width // 2, device=self.device, dtype=torch.float32)  # principal point x
        self.py = torch.tensor(self.height // 2, device=self.device, dtype=torch.float32)  # principal point y
        
        # Create coordinate grid for vectorized operations
        y_coords, x_coords = torch.meshgrid(
            torch.arange(image_height, device=self.device, dtype=torch.float32), 
            torch.arange(image_width, device=self.device, dtype=torch.float32),
            indexing="ij"
        )
        self.coords = torch.stack([x_coords, y_coords], dim=2)
        
    def cast_votes(self, segmentation_map, center_directions, depth_predictions, fx=500, fy=500, px=320, py=240):
        """
        Cast votes for object centers based on pixel predictions
        
        Args:
            segmentation_map: Binary mask of object class pixels (H x W)
            center_directions: Direction vectors to object center (H x W x 2)
            depth_predictions: Depth predictions for each pixel (H x W)
            
        Returns:
            vote_map: Accumulated voting scores (tensor)
            centers: Detected object centers (tensor)
            translations: 3D translations of detected objects (tensor)
        """
        # Ensure all inputs are on the correct device
        segmentation_map = self._ensure_tensor(segmentation_map)
        center_directions = self._ensure_tensor(center_directions)
        depth_predictions = self._ensure_tensor(depth_predictions)
        
        # Initialize voting map
        vote_map = torch.zeros((self.height, self.width), device=self.device, dtype=torch.float32)
        
        # Find pixels belonging to the object class
        object_mask = segmentation_map > 0
        object_indices = torch.nonzero(object_mask, as_tuple=True)
        
        if len(object_indices[0]) == 0:
            # Return empty tensors if no object pixels
            empty_centers = torch.zeros((0, 2), device=self.device, dtype=torch.float32)
            empty_translations = torch.zeros((0, 3), device=self.device, dtype=torch.float32)
            return vote_map, empty_centers, empty_translations
        
        # Get coordinates and directions of object pixels
        object_coords = self.coords[object_indices]
        object_directions = center_directions[object_indices]
        
        # Normalize direction vectors - fixed to handle dimensions properly
        dir_magnitudes = torch.norm(object_directions, dim=1, keepdim=True)
        valid_dirs = dir_magnitudes > 1e-5
        normalized_dirs = torch.zeros_like(object_directions)
        
        # Only normalize valid directions (avoid division by zero)
        valid_indices = valid_dirs.squeeze()
        normalized_dirs[valid_indices] = object_directions[valid_indices] / dir_magnitudes[valid_indices]
        
        # Vectorized vote casting
        max_dist = 100  # Maximum voting distance
        for t in range(1, max_dist):
            # Compute vote positions for all object pixels at once
            vote_positions = object_coords + normalized_dirs * t
            
            # Round to integers and convert to indices
            vote_x = vote_positions[:, 0].round().long()
            vote_y = vote_positions[:, 1].round().long()
            
            # Filter votes within image boundaries
            valid_votes = (vote_x >= 0) & (vote_x < self.width) & (vote_y >= 0) & (vote_y < self.height)
            
            if valid_votes.sum() > 0:
                valid_x = vote_x[valid_votes]
                valid_y = vote_y[valid_votes]
                
                # Accumulate votes using index_put_ for atomic operations
                vote_map.index_put_((valid_y, valid_x), torch.ones(len(valid_y), device=self.device), accumulate=True)
        
        # Find local maxima in vote map (object centers)
        centers, inlier_maps = self.find_centers(vote_map, segmentation_map, center_directions)
        
        # Calculate 3D translations for each center
        translations = self.calculate_translations(centers, inlier_maps, depth_predictions, fx=fx, fy=fy, px=px, py=py)
        
        return vote_map, centers, translations
    
    def find_centers(self, vote_map, segmentation_map, center_directions):
        """
        Find object centers by detecting local maxima in the voting map
        
        Args:
            vote_map: Accumulated voting scores (tensor)
            segmentation_map: Binary mask of object class pixels (tensor)
            center_directions: Direction vectors to object center (tensor)
            
        Returns:
            centers: Tensor of detected centers (N x 2)
            inlier_maps: Tensor of inlier maps for each center (N x H x W)
        """
        # Apply max pooling for non-maximum suppression
        kernel_size = 3
        padding = kernel_size // 2
        
        pooled = torch.nn.functional.max_pool2d(
            vote_map.unsqueeze(0), 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        ).squeeze(0)
        
        # Find local maxima
        is_max = (vote_map == pooled)
        # Apply threshold based on maximum value
        max_value = torch.max(vote_map).item()
        threshold = max_value * self.vote_threshold
        is_center = is_max & (vote_map > threshold)
        
        # Get center coordinates
        center_indices = torch.nonzero(is_center, as_tuple=False)  # Returns (N, 2) tensor
        
        if len(center_indices) == 0:
            # Return empty tensors if no centers found
            empty_centers = torch.zeros((0, 2), device=self.device, dtype=torch.float32)
            empty_inlier_maps = torch.zeros((0, self.height, self.width), device=self.device, dtype=torch.bool)
            return empty_centers, empty_inlier_maps
        
        # Flip coordinates to (x, y) format for centers
        centers = torch.flip(center_indices, [1]).float()  # Convert to (N, 2) tensor with (x, y) format
        
        # Create inlier maps for all centers
        num_centers = centers.shape[0]
        inlier_maps = torch.zeros((num_centers, self.height, self.width), device=self.device, dtype=torch.bool)
        
        # Find object pixels
        object_indices = torch.nonzero(segmentation_map > 0, as_tuple=False)  # (N, 2) tensor
        
        if len(object_indices) == 0:
            return centers, inlier_maps
        
        # Get coordinates and directions of object pixels
        object_pixels = self.coords[object_indices[:, 0], object_indices[:, 1]]  # (N, 2)
        directions = center_directions[object_indices[:, 0], object_indices[:, 1]]  # (N, 2)
        
        # Process each center
        for i in range(num_centers):
            center_tensor = centers[i].unsqueeze(0)  # (1, 2)
            
            # Compute vectors from pixels to center
            to_center_vectors = center_tensor - object_pixels  # (N, 2)
            
            # Normalize vectors
            to_center_dists = torch.norm(to_center_vectors, dim=1, keepdim=True)
            valid_dists = to_center_dists > 0
            normalized_to_center = torch.zeros_like(to_center_vectors)
            
            # Only normalize valid vectors
            valid_indices_dist = valid_dists.squeeze()
            if valid_indices_dist.sum() > 0:
                normalized_to_center[valid_indices_dist] = to_center_vectors[valid_indices_dist] / to_center_dists[valid_indices_dist]
            
            # Compute dot products
            dot_products = (normalized_to_center * directions).sum(dim=1)
            
            # Find inliers (dot product > 0.9)
            inliers = dot_products > 0.9
            
            # Set inlier map
            if inliers.sum() > 0:
                inlier_indices = object_indices[inliers]
                inlier_maps[i, inlier_indices[:, 0], inlier_indices[:, 1]] = True
        
        return centers, inlier_maps
    
    def calculate_translations(self, centers, inlier_maps, depth_predictions, fx=500, fy=500, px=320, py=240):
        """
        Calculate 3D translations for detected objects
        
        Args:
            centers: Tensor of detected centers (N x 2)
            inlier_maps: Tensor of inlier maps for each center (N x H x W)
            depth_predictions: Depth predictions for each pixel (H x W)
            
        Returns:
            translations: Tensor of 3D translations (N x 3)
        """
        num_centers = centers.shape[0]
        translations = torch.zeros((num_centers, 3), device=self.device, dtype=torch.float32)
        self.fx = torch.tensor(fx, device=self.device)  # focal length x
        self.fy = torch.tensor(fy, device=self.device)  # focal length y 
        self.px = torch.tensor(px, device=self.device)  # principal point x
        self.py = torch.tensor(py, device=self.device)  # principal point y
        for i in range(num_centers):
            cx, cy = centers[i]
            inlier_map = inlier_maps[i]
            
            # Get depths of inlier pixels
            inlier_depths = depth_predictions[inlier_map]
            
            if inlier_depths.numel() > 0:
                # Calculate average depth (Tz)
                Tz = inlier_depths.mean()
                
                # Calculate Tx and Ty using projection equation
                Tx = (cx - self.px) * Tz / self.fx
                Ty = (cy - self.py) * Tz / self.fy
                
                translations[i, 0] = Tx
                translations[i, 1] = Ty
                translations[i, 2] = Tz
        
        return translations
    
    def _ensure_tensor(self, data):
        """Helper function to ensure data is a tensor on the correct device"""
        if isinstance(data, np.ndarray):
            return torch.tensor(data, device=self.device, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device).float()
        else:
            raise TypeError(f"Expected numpy array or torch tensor, got {type(data)}")

    def visualize_results(self, vote_map, centers, translations, segmentation_map=None, center_directions=None):
        """
        Visualize the results of Hough voting
        
        Args:
            vote_map: Accumulated voting scores
            centers: Detected object centers
            translations: 3D translations of detected objects
            segmentation_map: Binary mask of object class pixels
            center_directions: Direction vectors to object center
        """
        # Convert tensors to numpy for visualization
        vote_map_np = vote_map.detach().cpu().numpy()
        centers_np = centers.detach().cpu().numpy()
        translations_np = translations.detach().cpu().numpy()
        
        if segmentation_map is not None:
            segmentation_np = segmentation_map.detach().cpu().numpy()
        
        if center_directions is not None:
            directions_np = center_directions.detach().cpu().numpy()
        
        # Visualize results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original segmentation
        if segmentation_map is not None:
            axes[0].imshow(segmentation_np, cmap='gray')
            axes[0].set_title('Object Segmentation')
        
        # Direction vectors (subsampled for clarity)
        if segmentation_map is not None and center_directions is not None:
            axes[1].imshow(segmentation_np, cmap='gray')
            stride = 20
            y_indices, x_indices = np.where(segmentation_np > 0)
            for i in range(0, len(y_indices), stride):
                y, x = y_indices[i], x_indices[i]
                nx, ny = directions_np[y, x]
                axes[1].arrow(x, y, nx*15, ny*15, head_width=3, head_length=3, fc='red', ec='red')
            axes[1].set_title('Center Direction Vectors')
        
        # Voting map and detected centers
        axes[2].imshow(vote_map_np, cmap='jet')
        axes[2].set_title('Vote Map and Detected Centers')
        for center in centers_np:
            cx, cy = center
            axes[2].plot(cx, cy, 'wo', markersize=10)
            axes[2].plot(cx, cy, 'ko', markersize=8)
        
        plt.tight_layout()
        plt.show()
        
        print("Detected centers:", centers_np)
        print("Estimated 3D translations:", translations_np)


# Example usage with PyTorch and gradient calculation capability
def demo_hough_voting_pytorch_differentiable():
    # Create synthetic data
    image_height, image_width = 480, 640
    
    # Create segmentation map (1 for object pixels, 0 for background)
    segmentation = np.zeros((image_height, image_width), dtype=np.uint8)
    segmentation[160:320, 200:400] = 1  # Object region
    
    # Create direction vectors toward center
    directions = np.zeros((image_height, image_width, 2), dtype=np.float32)
    center_x, center_y = 300, 240  # True object center
    
    # Create depth predictions
    depths = np.zeros((image_height, image_width), dtype=np.float32)
    true_depth = 2.5  # meters
    
    # Set direction vectors and depths for object pixels
    for y in range(image_height):
        for x in range(image_width):
            if segmentation[y, x] > 0:
                # Vector pointing to center
                dx = center_x - x
                dy = center_y - y
                mag = np.sqrt(dx*dx + dy*dy)
                if mag > 0:
                    directions[y, x, 0] = dx / mag  # nx
                    directions[y, x, 1] = dy / mag  # ny
                
                # Set depth with some noise
                depths[y, x] = true_depth + np.random.normal(0, 0.1)
    
    # Convert to PyTorch tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hough_voter = HoughVoterPyTorch(image_height, image_width, 0.5 ,device)
    import time
    start_time = time.time()
    vote_map, centers, translations = hough_voter.cast_votes(segmentation, directions, depths)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    vote_map = vote_map.detach().cpu()
    centers = centers.detach().cpu()
    translations = translations.detach().cpu()
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original segmentation
    axes[0].imshow(segmentation, cmap='gray')
    axes[0].set_title('Object Segmentation')
    
    # Direction vectors (subsampled for clarity)
    axes[1].imshow(segmentation, cmap='gray')
    stride = 20
    y_indices, x_indices = np.where(segmentation > 0)
    for i in range(0, len(y_indices), stride):
        y, x = y_indices[i], x_indices[i]
        nx, ny = directions[y, x]
        axes[1].arrow(x, y, nx*15, ny*15, head_width=3, head_length=3, fc='red', ec='red')
    axes[1].set_title('Center Direction Vectors')
    
    # Voting map and detected centers
    axes[2].imshow(vote_map, cmap='jet')
    axes[2].set_title('Vote Map and Detected Centers')
    for center in centers:
        cx, cy = center
        axes[2].plot(cx, cy, 'wo', markersize=10)
        axes[2].plot(cx, cy, 'ko', markersize=8)
    
    plt.tight_layout()
    plt.show()
    
    print("Detected centers:", centers)
    print("Estimated 3D translations:", translations)
    
    return vote_map, centers, translations

if __name__ == "__main__":
    demo_hough_voting_pytorch_differentiable()