"""
Hybrid Multitasking Model for Image Captioning, Visual Grounding, Segmentation, Classification, and Object Detection
"""

# --------------------- PROJECT STRUCTURE --------------------- #

# hybrid_multitask_model/
# ├── config/
# │   └── config.yaml
# ├── data/
# │   ├── __init__.py
# │   ├── coco_dataset.py
# │   ├── refcoco_dataset.py
# │   ├── imagenet_dataset.py
# │   └── data_utils.py
# ├── models/
# │   ├── __init__.py
# │   ├── feature_extractors/
# │   │   ├── __init__.py
# │   │   ├── cnn.py
# │   │   ├── resnet.py
# │   │   ├── swin.py
# │   │   └── vit.py
# │   ├── fusion/
# │   │   ├── __init__.py
# │   │   └── fusion_module.py
# │   ├── heads/
# │   │   ├── __init__.py
# │   │   ├── captioning_head.py
# │   │   ├── classification_head.py
# │   │   ├── detection_head.py
# │   │   ├── grounding_head.py
# │   │   └── segmentation_head.py
# │   └── multitask_model.py
# ├── losses/
# │   ├── __init__.py
# │   ├── captioning_loss.py
# │   ├── classification_loss.py
# │   ├── detection_loss.py
# │   ├── grounding_loss.py
# │   ├── segmentation_loss.py
# │   └── multi_task_loss.py
# ├── utils/
# │   ├── __init__.py
# │   ├── metrics.py
# │   └── visualization.py
# ├── train.py
# ├── evaluate.py
# └── requirements.txt



# --------------------- DATA LOADERS --------------------- #

# data/__init__.py




# data/coco_dataset.py


# data/refcoco_dataset.py



# data/imagenet_dataset.py



# data/data_utils.py

# --------------------- MODEL COMPONENTS --------------------- #

# models/feature_extractors/__init__.py




# models/feature_extractors/cnn.py




# models/feature_extractors/resnet.py



# models/feature_extractors/swin.py




# models/feature_extractors/vit.py



# models/fusion/__init__.py




# models/fusion/fusion_module.py




# models/heads/__init__.py




# models/heads/captioning_head.py




# models/heads/classification_head.py




# models/heads/detection_head.py



# models/heads/grounding_head.py


# models/heads/segmentation_head.py




# models/__init__.py



# models/multitask_model.py




# --------------------- LOSSES --------------------- #

# losses/__init__.py



# losses/captioning_loss.py


# losses/classification_loss.py




# losses/detection_loss.py


# losses/grounding_loss.py



# losses/segmentation_loss.py




# losses/multi_task_loss.py




# --------------------- UTILS --------------------- #

# utils/__init__.py

from .metrics import evaluate_captioning, evaluate_classification, evaluate_detection, evaluate_grounding, evaluate_segmentation
from .visualization import visualize_predictions


# utils/metrics.py



# utils/visualization.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torchvision.utils import make_grid

def visualize_predictions(images, predictions, targets=None, task='all'):
    """
    Visualize model predictions for different tasks
    
    Args:
        images: Batch of images [B, C, H, W]
        predictions: Dict of predictions for different tasks
        targets: Dict of targets for different tasks (optional)
        task: Specific task to visualize or 'all'
    """
    # Convert images from tensor to numpy and denormalize
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
        # Denormalize
        images = images * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        images = torch.clamp(images, 0, 1)
        
        # Convert to numpy
        images = images.permute(0, 2, 3, 1).numpy()
    
    batch_size = len(images)
    
    if task == 'all':
        fig, axes = plt.subplots(batch_size, 5, figsize=(20, 4 * batch_size))
        
        # If batch size is 1, wrap axes in a list
        if batch_size == 1:
            axes = [axes]
        
        for i in range(batch_size):
            # Original image
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Captioning
            if 'captioning' in predictions:
                # Placeholder for caption visualization
                axes[i, 1].imshow(images[i])
                axes[i, 1].set_title('Captioning')
                axes[i, 1].axis('off')
            else:
                axes[i, 1].axis('off')
            
            # Detection/Grounding
            if 'detection' in predictions or 'grounding' in predictions:
                axes[i, 2].imshow(images[i])
                
                # Draw bounding boxes
                if 'detection' in predictions:
                    boxes = predictions['detection'][0][i].detach().cpu().numpy()  # class_preds, box_preds
                    for box in boxes:
                        x, y, w, h = box
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                        axes[i, 2].add_patch(rect)
                
                if 'grounding' in predictions:
                    box = predictions['grounding'][i].detach().cpu().numpy()
                    x, y, w, h = box
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
                    axes[i, 2].add_patch(rect)
                
                axes[i, 2].set_title('Detection/Grounding')
                axes[i, 2].axis('off')
            else:
                axes[i, 2].axis('off')
            
            # Segmentation
            if 'segmentation' in predictions:
                mask = predictions['segmentation'][0][i, 0].detach().cpu().numpy()  # mask_pred, class_pred
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Convert to RGB
                
                # Create overlay
                alpha = 0.5
                overlay = images[i] * (1 - alpha) + mask * alpha
                
                axes[i, 3].imshow(overlay)
                axes[i, 3].set_title('Segmentation')
                axes[i, 3].axis('off')
            else:
                axes[i, 3].axis('off')
            
            # Classification
            if 'classification' in predictions:
                axes[i, 4].imshow(images[i])
                
                # Get top-5 predictions
                if isinstance(predictions['classification'], torch.Tensor):
                    probs = F.softmax(predictions['classification'][i], dim=0)
                    top5_prob, top5_idx = torch.topk(probs, 5)
                    
                    # Display class probabilities
                    class_text = '\n'.join([f'Class {idx.item()}: {prob.item():.3f}' for idx, prob in zip(top5_idx, top5_prob)])
                    axes[i, 4].text(0, 0, class_text, transform=axes[i, 4].transAxes)
                
                axes[i, 4].set_title('Classification')
                axes[i, 4].axis('off')
            else:
                axes[i, 4].axis('off')
    
    else:
        # Visualize specific task
        fig, axes = plt.subplots(batch_size, 2, figsize=(10, 4 * batch_size))
        
        # If batch size is 1, wrap axes in a list
        if batch_size == 1:
            axes = [axes]
        
        for i in range(batch_size):
            # Original image
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Task-specific visualization
            if task == 'captioning':
                # Placeholder for caption visualization
                axes[i, 1].imshow(images[i])
                axes[i, 1].set_title('Captioning')
                axes[i, 1].axis('off')
            
            elif task in ['detection', 'grounding']:
                axes[i, 1].imshow(images[i])
                
                # Draw bounding boxes
                if task == 'detection' and 'detection' in predictions:
                    boxes = predictions['detection'][0][i].detach().cpu().numpy()
                    for box in boxes:
                        x, y, w, h = box
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                        axes[i, 1].add_patch(rect)
                
                if task == 'grounding' and 'grounding' in predictions:
                    box = predictions['grounding'][i].detach().cpu().numpy()
                    x, y, w, h = box
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
                    axes[i, 1].add_patch(rect)
                
                axes[i, 1].set_title(task.capitalize())
                axes[i, 1].axis('off')
            
            elif task == 'segmentation':
                if 'segmentation' in predictions:
                    mask = predictions['segmentation'][0][i, 0].detach().cpu().numpy()
                    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                    
                    # Create overlay
                    alpha = 0.5
                    overlay = images[i] * (1 - alpha) + mask * alpha
                    
                    axes[i, 1].imshow(overlay)
                    axes[i, 1].set_title('Segmentation')
                    axes[i, 1].axis('off')
            
            elif task == 'classification':
                if 'classification' in predictions:
                    axes[i, 1].imshow(images[i])
                    
                    # Get top-5 predictions
                    if isinstance(predictions['classification'], torch.Tensor):
                        probs = F.softmax(predictions['classification'][i], dim=0)
                        top5_prob, top5_idx = torch.topk(probs, 5)
                        
                        # Display class probabilities
                        class_text = '\n'.join([f'Class {idx.item()}: {prob.item():.3f}' for idx, prob in zip(top5_idx, top5_prob)])
                        axes[i, 1].text(0, 0, class_text, transform=axes[i, 1].transAxes)
                    
                    axes[i, 1].set_title('Classification')
                    axes[i, 1].axis('off')
    
    plt.tight_layout()
    return fig


# train.py

import argparse
import os
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import MultitaskModel
from losses import MultiTaskLoss
from utils.metrics import evaluate_captioning, evaluate_classification, evaluate_detection, evaluate_grounding, evaluate_segmentation
from utils.visualization import visualize_predictions
from data.data_utils import get_dataloaders

def train(config):
    # Set device
    device = torch.device(config['general']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Set random seed
    torch.manual_seed(config['general']['seed'])
    np.random.seed(config['general']['seed'])
    
    # Create directories for checkpoints and logs
    os.makedirs(config['general']['save_dir'], exist_ok=True)
    os.makedirs(config['general']['log_dir'], exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(config['general']['log_dir'])
    
    # Load data
    dataloaders = get_dataloaders(config)
    
    # Initialize model
    model = MultitaskModel(config).to(device)
    
    # Initialize loss function
    criterion = MultiTaskLoss(config)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['lr'] / 10
    )
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        # Training
        model.train()
        train_losses = []
        
        for task, dataloader in dataloaders.items():
            if 'train' not in task:
                continue
            
            task_name = task.split('_')[0]  # Extract task name (e.g., 'coco_train' -> 'coco')
            print(f"Training on {task_name} dataset")
            
            for batch in tqdm(dataloader):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch, task=task_name)
                
                # Prepare targets based on task
                task_loss = None
                if 'caption' in batch:
                    targets = batch['caption'][:, 1:]  # Exclude <sos> token
                    task_loss = 'captioning'
                elif 'bbox' in batch:
                    targets = batch['bbox']
                    task_loss = 'grounding'
                elif 'label' in batch:
                    targets = batch['label']
                    task_loss = 'classification'
                else:
                    continue
                
                # Calculate loss
                _, loss = criterion(outputs, targets, task=task_loss)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Record loss
                train_losses.append(loss.item())
        
        # Log training loss
        avg_train_loss = sum(train_losses) / max(len(train_losses), 1)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        print(f"Epoch {epoch+1}: Average Training Loss: {avg_train_loss:.4f}")
    
        # Validation
        model.eval()
        val_losses = []
        all_metrics = {task: {} for task in dataloaders if 'val' in task}
    
        with torch.no_grad():
            for task, dataloader in dataloaders.items():
                if 'val' not in task:
                    continue
    
                task_name = task.split('_')[0]  # Extract task name
                print(f"Validating on {task_name} dataset")
    
                for batch in tqdm(dataloader):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    outputs = model(batch, task=task_name)
    
                    # Define validation targets
                    task_loss = None
                    if 'caption' in batch:
                        targets = batch['caption'][:, 1:]
                        task_loss = 'captioning'
                    elif 'bbox' in batch:
                        targets = batch['bbox']
                        task_loss = 'grounding'
                    elif 'label' in batch:
                        targets = batch['label']
                        task_loss = 'classification'
                    elif 'mask' in batch:
                        targets = batch['mask']
                        task_loss = 'segmentation'
                    elif 'bounding_boxes' in batch:
                        targets = batch['bounding_boxes']
                        task_loss = 'detection'
                    else:
                        continue
    
                    _, loss = criterion(outputs, targets, task=task_loss)
                    val_losses.append(loss.item())
    
                    # Evaluate predictions
                    if task_loss == 'captioning':
                        metric = evaluate_captioning(outputs, targets)
                    elif task_loss == 'grounding':
                        metric = evaluate_grounding(outputs, targets)
                    elif task_loss == 'classification':
                        metric = evaluate_classification(outputs, targets)
                    elif task_loss == 'segmentation':
                        metric = evaluate_segmentation(outputs, targets)
                    elif task_loss == 'detection':
                        metric = evaluate_detection(outputs, targets)
                    else:
                        metric = {}
    
                    # Store metrics
                    for key, value in metric.items():
                        if key not in all_metrics[task]:
                            all_metrics[task][key] = []
                        all_metrics[task][key].append(value)
    
        # Compute average validation loss
        avg_val_loss = sum(val_losses) / max(len(val_losses), 1)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        print(f"Epoch {epoch+1}: Average Validation Loss: {avg_val_loss:.4f}")
    
        # Log evaluation metrics
        for task, metrics in all_metrics.items():
            for metric_name, values in metrics.items():
                avg_value = sum(values) / len(values)
                writer.add_scalar(f"Metrics/{task}/{metric_name}", avg_value, epoch)
                print(f"{task} {metric_name}: {avg_value:.4f}")
    
        # Save model checkpoint
        checkpoint_path = os.path.join(config['general']['save_dir'], f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
        # Step the learning rate scheduler
        scheduler.step()
    
    # Final cleanup
    writer.close()
    print("Training complete!")

    ##evaluate.py
import torch
import yaml
import os
from tqdm import tqdm
from dataset import get_dataloaders
from model import HybridMultitaskModel
from metrics import evaluate_captioning, evaluate_grounding, evaluate_classification, evaluate_segmentation, evaluate_detection

# Load Configurations
config_path = "config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get test dataloaders
dataloaders = get_dataloaders(config, mode="test")

# Load model
model = HybridMultitaskModel(config).to(device)
checkpoint_path = config["general"]["checkpoint_path"]

if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint from {checkpoint_path}")
else:
    raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")

# Evaluation Mode
model.eval()
all_metrics = {task: {} for task in dataloaders}

# Start Evaluation
with torch.no_grad():
    for task, dataloader in dataloaders.items():
        print(f"\nEvaluating on {task} dataset")

        for batch in tqdm(dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            task_name = task.split("_")[0]  # Extract task name
            outputs = model(batch, task=task_name)

            # Define evaluation targets
            if task_name == "coco":
                targets = batch["caption"][:, 1:]  # Remove <sos> token
                metric_func = evaluate_captioning
            elif task_name == "refcoco":
                targets = batch["bbox"]
                metric_func = evaluate_grounding
            elif task_name == "imagenet":
                targets = batch["label"]
                metric_func = evaluate_classification
            elif task_name == "coco-stuff":
                targets = batch["mask"]
                metric_func = evaluate_segmentation
            elif task_name == "voc":
                targets = batch["bounding_boxes"]
                metric_func = evaluate_detection
            else:
                continue

            # Compute evaluation metrics
            metric_results = metric_func(outputs, targets)

            # Store metrics
            for key, value in metric_results.items():
                if key not in all_metrics[task]:
                    all_metrics[task][key] = []
                all_metrics[task][key].append(value)

# Compute and Print Final Metrics
print("\n📊 Final Evaluation Results:")
for task, metrics in all_metrics.items():
    print(f"\n📌 {task} Results:")
    for metric_name, values in metrics.items():
        avg_value = sum(values) / len(values)
        print(f"   {metric_name}: {avg_value:.4f}")

print("\n✅ Evaluation Complete!")
