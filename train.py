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
