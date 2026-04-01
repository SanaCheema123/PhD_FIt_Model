import torch
import torch.nn as nn
from .captioning_loss import CaptioningLoss
from .classification_loss import ClassificationLoss
from .detection_loss import DetectionLoss
from .grounding_loss import GroundingLoss
from .segmentation_loss import SegmentationLoss

class MultiTaskLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Task-specific losses
        self.losses = {
            'captioning': CaptioningLoss(),
            'classification': ClassificationLoss(),
            'detection': DetectionLoss(),
            'grounding': GroundingLoss(),
            'segmentation': SegmentationLoss()
        }
        
        # Task weights
        self.task_weights = {
            'captioning': config['training']['loss_weights']['captioning'],
            'classification': config['training']['loss_weights']['classification'],
            'detection': config['training']['loss_weights']['detection'],
            'grounding': config['training']['loss_weights']['grounding'],
            'segmentation': config['training']['loss_weights']['segmentation']
        }
    
    def forward(self, predictions, targets, task=None):
        # If task is specified, compute loss for that task only
        if task is not None:
            if task not in self.losses:
                raise ValueError(f"Unknown task: {task}")
            
            loss = self.losses[task](predictions, targets)
            weighted_loss = self.task_weights[task] * loss
            
            return {task: loss.item()}, weighted_loss
        
        # Otherwise, compute loss for all tasks
        total_loss = 0.0
        loss_dict = {}
        
        for task, loss_fn in self.losses.items():
            if task in predictions and task in targets:
                task_loss = loss_fn(predictions[task], targets[task])
                weighted_loss = self.task_weights[task] * task_loss
                
                loss_dict[task] = task_loss.item()
                total_loss += weighted_loss
        
        return loss_dict, total_loss