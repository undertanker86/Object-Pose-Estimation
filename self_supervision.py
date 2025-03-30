import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from TeacherRenderer import TeacherRenderer

class GeometryGuidedFilter:
    def __init__(self, visual_threshold=0.3, geom_threshold=0.05):
        self.visual_threshold = visual_threshold
        self.geom_threshold = geom_threshold
        
    def compute_visual_alignment(self, original_image, rendered_image, mask):
        """Calculate visual alignment score"""
        # mask loss
        mask_diff = F.binary_cross_entropy_with_logits(
            rendered_image['mask'], mask, reduction='mean')
        
        # appearance loss (L1 on masked regions)
        masked_original = original_image * mask
        masked_rendered = rendered_image['rgb'] * mask
        appearance_diff = F.l1_loss(masked_rendered, masked_original, reduction='mean')
        
        # structural similarity loss (simplified)
        ssim_loss = 1.0 - self._compute_ssim(masked_rendered, masked_original)
        
        # Combined visual score
        alpha, beta = 0.3, 0.6  # weights
        visual_score = mask_diff + alpha * appearance_diff + beta * ssim_loss
        
        return visual_score
    
    def _compute_ssim(self, img1, img2, window_size=11, size_average=True):
        """Compute Structural Similarity Index (simplified implementation)"""
        # Simplified SSIM implementation - could be replaced with a more accurate version
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def compute_geometric_alignment(self, predicted_depth, rendered_depth, mask):
        """Calculate geometric alignment score using Chamfer distance"""
        # Only consider pixels where both depths are valid
        valid_mask = (predicted_depth > 0) & (rendered_depth > 0) & mask.bool()
        
        if not valid_mask.any():
            return float('inf')  # No valid pixels for comparison
            
        # Compute Chamfer distance
        pred_depth_valid = predicted_depth[valid_mask]
        render_depth_valid = rendered_depth[valid_mask]
        
        # L1 distance for simplicity
        depth_diff = torch.abs(pred_depth_valid - render_depth_valid).mean()
        
        return depth_diff
        
    def filter_poses(self, predictions, images, depths, renderer):
        """Filter pose predictions based on visual and geometric criteria"""
        filtered_preds = []
        
        for pred in predictions:
            # Render with predicted pose
            rendered = renderer.render(pred['R'], pred['t'], pred['K'])
            
            # Compute alignment scores
            visual_score = self.compute_visual_alignment(
                images[pred['idx']], rendered, pred['mask'])
            
            geom_score = self.compute_geometric_alignment(
                depths[pred['idx']], rendered['depth'], pred['mask'])
            
            # Filter based on thresholds
            if visual_score < self.visual_threshold and geom_score < self.geom_threshold:
                filtered_preds.append(pred)
                
        return filtered_preds


class TeacherStudentTrainer:
    def __init__(self, teacher_model, student_model, renderer, geometry_filter, ema_decay=0.999):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.renderer = renderer
        self.geometry_filter = geometry_filter
        self.ema_decay = ema_decay
        
        # Initialize teacher with student weights
        self._initialize_teacher()
        
    def _initialize_teacher(self):
        """Initialize teacher model with student weights"""
        for teacher_param, student_param in zip(self.teacher_model.parameters(), 
                                               self.student_model.parameters()):
            teacher_param.data.copy_(student_param.data)
            teacher_param.requires_grad = False
    
    def update_teacher(self):
        """Update teacher model using EMA"""
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_model.parameters(), 
                                                   self.student_model.parameters()):
                teacher_param.data.mul_(self.ema_decay).add_(
                    student_param.data * (1.0 - self.ema_decay))
    
    def generate_pseudo_labels(self, images, camera_intrinsics):
        """Generate pseudo-labels from teacher model predictions"""
        with torch.no_grad():
            # Run teacher model
            outputs = self.teacher_model(images, 
                                         fx=camera_intrinsics[:, 0, 0], 
                                         fy=camera_intrinsics[:, 1, 1], 
                                         px=camera_intrinsics[:, 0, 2], 
                                         py=camera_intrinsics[:, 1, 2])
            
            vote_map, centers, translations, depth, seg, vector_field, _ = outputs
            
            # Create predictions structure
            batch_size = images.size(0)
            predictions = []
            
            for i in range(batch_size):
                # For each detected center
                for j in range(centers.size(0)):
                    # Extract pose information
                    center = centers[j]
                    translation = translations[j]
                    
                    # Create rotation and translation matrices
                    # Note: In a real implementation, you would need to estimate rotation as well
                    # This is a simplified version that only uses translation
                    R = torch.eye(3, device=images.device)  # Identity rotation as placeholder
                    t = translation
                    
                    predictions.append({
                        'idx': i,
                        'R': R,
                        't': t,
                        'K': camera_intrinsics[i],
                        'mask': seg[i].argmax(dim=0) > 0,  # Binary segmentation mask
                        'center': center
                    })
            
            # Filter predictions using geometry-guided criteria
            filtered_predictions = self.geometry_filter.filter_poses(
                predictions, images, depth, self.renderer)
            
            # Create pseudo-labels from filtered predictions
            pseudo_labels = self._create_pseudo_labels(filtered_predictions, images.shape)
            
            return pseudo_labels if pseudo_labels else None
    
    def _create_pseudo_labels(self, filtered_predictions, image_shape):
        """Convert filtered predictions to pseudo-labels format"""
        if not filtered_predictions:
            return None
            
        batch_size, channels, height, width = image_shape
        device = filtered_predictions[0]['R'].device
        
        # Initialize pseudo-label tensors
        seg_labels = torch.zeros(batch_size, height, width, device=device)
        depth_labels = torch.zeros(batch_size, 1, height, width, device=device)
        vector_field_labels = torch.zeros(batch_size, 2, height, width, device=device)
        
        # Fill in pseudo-labels based on filtered predictions
        for pred in filtered_predictions:
            idx = pred['idx']
            
            # Render the prediction
            rendered = self.renderer.render(pred['R'], pred['t'], pred['K'])
            
            # Update segmentation mask
            seg_labels[idx] = rendered['mask'].squeeze()
            
            # Update depth map where mask is valid
            depth_labels[idx, 0] = rendered['depth'].squeeze() * rendered['mask'].squeeze()
            
            # Generate vector field pointing to the predicted center
            center = pred['center']
            y_coords = torch.arange(0, height, device=device).view(1, -1, 1).expand(1, -1, width)
            x_coords = torch.arange(0, width, device=device).view(1, 1, -1).expand(1, height, -1)
            
            # Calculate vectors pointing to center
            vectors_x = center[0] - x_coords
            vectors_y = center[1] - y_coords
            
            # Normalize vectors
                        # Normalize vectors
            magnitude = torch.sqrt(vectors_x**2 + vectors_y**2) + 1e-10
            unit_vectors_x = vectors_x / magnitude
            unit_vectors_y = vectors_y / magnitude
            
            # Apply to regions with valid mask
            mask = rendered['mask'].squeeze().bool()
            vector_field_labels[idx, 0][mask] = unit_vectors_x[0][mask]
            vector_field_labels[idx, 1][mask] = unit_vectors_y[0][mask]
        
        return {
            'segmentation': seg_labels,
            'depth': depth_labels,
            'vector_field': vector_field_labels,
            'pose': {
                'R': torch.stack([pred['R'] for pred in filtered_predictions]),
                't': torch.stack([pred['t'] for pred in filtered_predictions])
            }
            
        }

    def train_step(self, batch, optimizer, criterion):
        """Perform a single training step with teacher-student framework"""
        images = batch['rgb']
        camera_intrinsics = batch['cam_K']
        
        # Generate pseudo-labels from teacher model
        pseudo_labels = self.generate_pseudo_labels(images, camera_intrinsics)
        
        if pseudo_labels is None:
            # Skip this batch if no good pseudo-labels were found
            return None
        
        # Forward pass with student model
        outputs = self.student_model(
            images, 
            fx=camera_intrinsics[:, 0, 0], 
            fy=camera_intrinsics[:, 1, 1], 
            px=camera_intrinsics[:, 0, 2], 
            py=camera_intrinsics[:, 1, 2]
        )
        
        # Compute losses using pseudo-labels
        loss = criterion(outputs, pseudo_labels)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update teacher using EMA
        self.update_teacher()
        
        return loss