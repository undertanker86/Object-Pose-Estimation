import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import tomli
from tqdm import tqdm
import torch.nn as nn
from model import MTGOE
# from mmd import PoseEstimationModel
from train_ver2 import BOPDataset
from TeacherRenderer import TeacherRenderer
from self_supervision import TeacherStudentTrainer, GeometryGuidedFilter
class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_pos=1.0, lambda_depth=0.5, lambda_seg=0.5, lambda_vf=0.5):
        super(MultiTaskLoss, self).__init__()
        self.lambda_pos = lambda_pos
        self.lambda_depth = lambda_depth
        self.lambda_seg = lambda_seg
        self.lambda_vf = lambda_vf
        
    def forward(self, outputs, targets):
        """
        Compute multi-task loss
        
        Args:
            outputs: tuple of (vote_map, centers, translations, depth, seg, vector_field, fused_features)
            targets: dictionary with 'depth_map', 'segmentation_mask', 'vector_field', etc.
            
        Returns:
            total_loss: combined loss value
        """
        vote_map, centers, translations, depth, seg, vector_field, _ = outputs
        if 'pose' in targets and targets['pose'] is not None:
            # Ground truth poses and centers available
            gt_translations = targets['pose']['t']  # Ground truth translations [N_gt, 3]
            gt_centers = targets['pose']['centers']  # Ground truth centers [N_gt, 2]

            # Compute position error for each prediction
            if len(translations) > 0 and len(gt_translations) > 0:
                pos_errors = []
                center_errors = []  # To store center errors

                for pred_t, pred_c in zip(translations, centers):
                    # Translation error: Find closest ground truth translation
                    dists_t = torch.norm(pred_t.unsqueeze(0) - gt_translations, dim=1)
                    min_dist_t = torch.min(dists_t)
                    pos_errors.append(min_dist_t)

                    # Center error: Compute MSE with ground truth centers
                    dists_c = torch.norm(pred_c.unsqueeze(0) - gt_centers, dim=1)
                    min_dist_c = torch.min(dists_c)
                    center_errors.append(min_dist_c)

                # Aggregate position and center errors
                if pos_errors:
                    pos_loss = torch.stack(pos_errors).mean()
                else:
                    pos_loss = torch.tensor(0.0, device=depth.device)

                if center_errors:
                    center_loss = torch.stack(center_errors).mean()
                else:
                    center_loss = torch.tensor(0.0, device=depth.device)

            else:
                pos_loss = torch.tensor(0.0, device=depth.device)
                center_loss = torch.tensor(0.0, device=depth.device)
        else:
            pos_loss = torch.tensor(0.0, device=depth.device)
            center_loss = torch.tensor(0.0, device=depth.device)
        # Pose loss (position error)
        if 'pose' in targets and targets['pose'] is not None:
            # Ground truth poses available
            gt_translations = targets['pose']['t']
            
            # Compute position error for each prediction
            if len(translations) > 0 and len(gt_translations) > 0:
                pos_errors = []
                for pred_t in translations:
                    # Find closest ground truth
                    dists = torch.norm(pred_t.unsqueeze(0) - gt_translations, dim=1)
                    min_dist = torch.min(dists)
                    pos_errors.append(min_dist)
                
                if pos_errors:
                    pos_loss = torch.stack(pos_errors).mean()
                else:
                    pos_loss = torch.tensor(0.0, device=depth.device)
            else:
                pos_loss = torch.tensor(0.0, device=depth.device)
        else:
            pos_loss = torch.tensor(0.0, device=depth.device)
        
        # Depth loss
        if 'depth' in targets and targets['depth'] is not None:
            depth_loss = F.l1_loss(depth, targets['depth'])
        else:
            depth_loss = torch.tensor(0.0, device=depth.device)
        
        # Segmentation loss
        if 'segmentation' in targets and targets['segmentation'] is not None:
            # Cross-entropy loss for segmentation
            seg_loss = F.cross_entropy(seg, targets['segmentation'].long())
        else:
            seg_loss = torch.tensor(0.0, device=depth.device)
        
        if 'vector_field' in targets and targets['vector_field'] is not None:
        
            print(f"vector_field shape: {vector_field.shape}")
            print(f"gt_vector_field shape: {targets['vector_field'].shape}")
            
          
            if 'segmentation_mask' in targets and targets['segmentation_mask'] is not None:
                mask = targets['segmentation_mask'].float().unsqueeze(1)
                print(f"mask shape: {mask.shape}")
            else:
                mask = torch.ones_like(vector_field[:, :1])
                print(f"created mask shape: {mask.shape}")
            
            
            gt_vector_field = targets['vector_field']
            
           
            if gt_vector_field.dim() == 4 and gt_vector_field.shape[-1] == 2:
                gt_vector_field = gt_vector_field.permute(0, 3, 1, 2)
                print(f"permuted gt_vector_field shape: {gt_vector_field.shape}")
                
        
            if mask.shape[2:] != vector_field.shape[2:]:
                mask = F.interpolate(mask, size=vector_field.shape[2:], mode='nearest')
                print(f"resized mask shape: {mask.shape}")
                
            
            vf_loss = F.smooth_l1_loss(
                vector_field * mask, 
                gt_vector_field * mask, 
                reduction='sum'
            ) / (mask.sum() + 1e-10)
        else:
            vf_loss = torch.tensor(0.0, device=depth.device)
            # Combine losses
        total_loss = (
            self.lambda_pos * pos_loss +
            self.lambda_depth * depth_loss +
            self.lambda_seg * seg_loss +
            self.lambda_vf * vf_loss
            + center_loss * 0.5
        )
        
        return total_loss, {
            'pos': pos_loss.item(),
            'depth': depth_loss.item(),
            'seg': seg_loss.item(),
            'vf': vf_loss.item(),
            'total': total_loss.item()
        }
# Create argument parser
parser = argparse.ArgumentParser(description='Train MTGOE model for 6D pose estimation')
parser.add_argument('--config', type=str, default='config.toml', help='Path to config file')
parser.add_argument('--obj_id', type=int, default=1, help='Object ID to train on')
parser.add_argument('--mode', type=str, choices=['supervised', 'self_supervised', 'both'], default='both',
                   help='Training mode: supervised, self_supervised, or both')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for checkpoints')
args = parser.parse_args()

# Load config
with open(args.config, "rb") as f:
    config = tomli.load(f)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Initialize datasets
# Paths adjusted for your dataset structure
# Đường dẫn tới dataset
synthetic_dataset = BOPDataset(
    dataset_root='./data/lmo',
    split='train',
    obj_id=1
)

real_dataset = BOPDataset(
    dataset_root='./data/lmo',
    split='train',
    obj_id=1
)

test_dataset = BOPDataset(
    dataset_root='./data/lmo',
    split='test',
    obj_id=1
)

# Create data loaders
synthetic_loader = DataLoader(
    synthetic_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

real_loader = DataLoader(
    real_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2
)

# Initialize models
mtgoe_model = MTGOE(
    image_height=config['data']['image_height'],
    image_width=config['data']['image_width']
).to(device)

# pose_model = PoseEstimationModel(
#     depth_channels=1,
#     seg_channels=19,
#     vector_channels=2,
#     feature_dim=256,
#     num_keypoints=8
# ).to(device)

# Initialize loss function
multi_task_loss = MultiTaskLoss(
    lambda_pos=config['lambdas']['lambda_pos'],
    lambda_depth=config['lambdas']['lambda_depth'],
    lambda_seg=config['lambdas']['lambda_sem'],
    lambda_vf=config['lambdas']['lambda_vf']
)

# Initialize optimizer
optimizer = optim.Adam(
    list(mtgoe_model.parameters()),
    lr=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay']
)

# Initialize learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config['training']['scheduler_step_size'],
    gamma=config['training']['scheduler_gamma']
)

# Resume from checkpoint if specified
start_epoch = 0
if args.resume:
    if os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        mtgoe_model.load_state_dict(checkpoint['mtgoe_state_dict'])
        # pose_model.load_state_dict(checkpoint['pose_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    else:
        print(f"No checkpoint found at {args.resume}")

# Function to save checkpoint
def save_checkpoint(epoch, is_best=False):
    checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'mtgoe_state_dict': mtgoe_model.state_dict(),
        # 'pose_state_dict': pose_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(args.output_dir, 'best_model.pth')
        torch.save({
            'epoch': epoch,
            'mtgoe_state_dict': mtgoe_model.state_dict(),
            # 'pose_state_dict': pose_model.state_dict(),
        }, best_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")

# Function for supervised training
def train_supervised(num_epochs):
    """Train with synthetic data"""
    print(f"\nStarting supervised training for {num_epochs} epochs...")
    mtgoe_model.train()
    # pose_model.train()
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_losses = {'total': 0, 'pos': 0, 'depth': 0, 'seg': 0, 'vf': 0}
        
        progress_bar = tqdm(enumerate(synthetic_loader), total=len(synthetic_loader), desc=f"Epoch {epoch+1}/{start_epoch + num_epochs}")
        for batch_idx, batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = mtgoe_model(
                batch['rgb'],
                fx=batch['cam_K'][:, 0, 0],
                fy=batch['cam_K'][:, 1, 1],
                px=batch['cam_K'][:, 0, 2],
                py=batch['cam_K'][:, 1, 2]
            )
            
            # Create targets
            targets = {
                'depth': batch['depth_map'].unsqueeze(1) if 'depth_map' in batch else None,
                'segmentation': batch['segmentation_mask'] if 'segmentation_mask' in batch else None,
                'vector_field': batch['vector_field'] if 'vector_field' in batch else None,
                'pose': {
                    'R': batch['cam_R_m2c'] if 'cam_R_m2c' in batch else None,
                    't': batch['cam_t_m2c'] if 'cam_t_m2c' in batch else None
                }
            }
            
            # Compute loss
            loss, losses_dict = multi_task_loss(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(mtgoe_model.parameters()), 
                config['training']['gradient_clip']
            )
            
            optimizer.step()
            
            # Update progress bar
            for k, v in losses_dict.items():
                epoch_losses[k] += v
                
            avg_loss = epoch_losses['total'] / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f"{avg_loss:.4f}"})
        
        # Step the scheduler
        scheduler.step()
        
        # Log epoch results
        print(f"Epoch {epoch+1}/{start_epoch + num_epochs} Results:")
        for k, v in epoch_losses.items():
            avg_v = v / len(synthetic_loader)
            print(f"  {k}: {avg_v:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_frequency'] == 0:
            save_checkpoint(epoch)
    
    return start_epoch + num_epochs

# Function for self-supervised training
def train_self_supervised(num_epochs, start_from_epoch):
    """Self-supervised training with real data"""
    print(f"\nStarting self-supervised training for {num_epochs} epochs...")
    
    # Initialize teacher-student framework
    teacher_model = MTGOE(
        image_height=config['data']['image_height'],
        image_width=config['data']['image_width']
    ).to(device)
    teacher_model.load_state_dict(mtgoe_model.state_dict())
    
    renderer = TeacherRenderer(
        cad_model_path=real_dataset.model_path,
        image_height=config['data']['image_height'],
        image_width=config['data']['image_width'],
        device=device
    )
    
    geometry_filter = GeometryGuidedFilter(
        visual_threshold=0.3,
        geom_threshold=0.05
    )
    
    teacher_student = TeacherStudentTrainer(
        teacher_model=teacher_model,
        student_model=mtgoe_model,
        renderer=renderer,
        geometry_filter=geometry_filter,
        ema_decay=0.999
    )
    
    mtgoe_model.train()
    # pose_model.train()
    
    for epoch in range(start_from_epoch, start_from_epoch + num_epochs):
        epoch_losses = {'total': 0, 'pos': 0, 'depth': 0, 'seg': 0, 'vf': 0}
        valid_batches = 0
        
        progress_bar = tqdm(enumerate(real_loader), total=len(real_loader), desc=f"Epoch {epoch+1}/{start_from_epoch + num_epochs}")
        for batch_idx, batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Train step
            result = teacher_student.train_step(batch, optimizer, multi_task_loss)
            
            if result is not None:
                valid_batches += 1
                loss_val, losses_dict = result
                
                # Update epoch losses
                for k, v in losses_dict.items():
                    epoch_losses[k] += v
                
                # Update progress bar
                avg_loss = epoch_losses['total'] / valid_batches
                progress_bar.set_postfix({'loss': f"{avg_loss:.4f}", 'valid_batches': valid_batches})
        
        # Step the scheduler
        scheduler.step()
        
        # Log epoch results
        print(f"Epoch {epoch+1}/{start_from_epoch + num_epochs} Results:")
        for k, v in epoch_losses.items():
            avg_v = v / max(1, valid_batches)
            print(f"  {k}: {avg_v:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_frequency'] == 0:
            save_checkpoint(epoch)
    
    return start_from_epoch + num_epochs

# Function for evaluation
def evaluate():
    """Evaluate the model on test data"""
    print("\nEvaluating model...")
    mtgoe_model.eval()
    # pose_model.eval()
    
    # Metrics
    translation_errors = []
    rotation_errors = []
    add_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = mtgoe_model(
                batch['rgb'],
                fx=batch['cam_K'][:, 0, 0],
                fy=batch['cam_K'][:, 1, 1],
                px=batch['cam_K'][:, 0, 2],
                py=batch['cam_K'][:, 1, 2]
            )
            
            # Get predictions
            _, centers, translations, _, _, _, _ = outputs
            
            # Get ground truth
            gt_translation = batch['cam_t_m2c'][0] if 'cam_t_m2c' in batch else None
            gt_rotation = batch['cam_R_m2c'][0] if 'cam_R_m2c' in batch else None
            
            if gt_translation is not None and len(translations) > 0:
                # Find the closest prediction to ground truth
                dists = torch.norm(translations - gt_translation.unsqueeze(0), dim=1)
                min_idx = torch.argmin(dists)
                
                # Translation error
                trans_error = dists[min_idx].item()
                translation_errors.append(trans_error)
                
                # Compute ADD metric if we have model points and rotation
                if gt_rotation is not None and hasattr(test_dataset, 'model'):
                    # In real implementation, we would also need to estimate rotation
                    # This is a simplified version that assumes we have ground truth rotation
                    model_points = torch.tensor(test_dataset.model.vertices, 
                                               dtype=torch.float32, device=device)
                    
                    # Transform model points with predicted and ground truth poses
                    pred_points = torch.mm(gt_rotation, model_points.T) + translations[min_idx].view(3, 1)
                    gt_points = torch.mm(gt_rotation, model_points.T) + gt_translation.view(3, 1)
                    
                    # Compute ADD metric
                    model_dists = torch.norm(pred_points - gt_points, dim=0)
                    add = torch.mean(model_dists).item()
                    
                    # Get model diameter
                    if hasattr(batch, 'extens'):
                        diameter = torch.norm(batch['extens'][0]).item()
                    else:
                        # Use a default if not available
                        diameter = 0.1
                    
                    # ADD relative to diameter
                    add_rel = add / diameter
                    add_metrics.append(add_rel)
    
    # Compute average metrics
    avg_trans_error = np.mean(translation_errors) if translation_errors else float('nan')
    avg_add = np.mean(add_metrics) if add_metrics else float('nan')
    
    # Compute ADD accuracy (% of samples with ADD < 10% of diameter)
    add_accuracy = np.mean(np.array(add_metrics) < 0.1) if add_metrics else float('nan')
    
    print(f"Evaluation Results:")
    print(f"  Average Translation Error: {avg_trans_error:.4f} m")
    print(f"  Average ADD: {avg_add:.4f}")
    print(f"  ADD Accuracy (<10%): {add_accuracy:.4f}")
    
    return add_accuracy

# Main training loop
if __name__ == "__main__":
    current_epoch = start_epoch
    
    if args.mode in ['supervised', 'both']:
        current_epoch = train_supervised(config['training']['supervised_epochs'])
        
    if args.mode in ['self_supervised', 'both'] and config['training']['use_self_supervised']:
        current_epoch = train_self_supervised(config['training']['self_supervised_epochs'], current_epoch)
    
    # Final evaluation
    add_accuracy = evaluate()
    
    # Save final model
    save_checkpoint(current_epoch - 1, is_best=(add_accuracy > 0.5))
    
    print("Training and evaluation complete!")