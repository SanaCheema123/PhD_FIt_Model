
import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        # Focal loss for classification
        self.cls_criterion = self._focal_loss
        
        # Smooth L1 loss for box regression
        self.box_criterion = nn.SmoothL1Loss(reduction='sum')
    
    def _focal_loss(self, predictions, targets):
        # Focal loss implementation
        probs = torch.sigmoid(predictions)
        pt = probs * targets + (1 - probs) * (1 - targets)
        weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        weight = weight * (1 - pt).pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(
            predictions, targets, weight=weight, reduction='sum'
        )
        return loss
    
    def forward(self, predictions, targets):
        # Predictions: [class_preds, box_preds]
        # Targets: [class_targets, box_targets]
        
        class_preds, box_preds = predictions
        class_targets, box_targets = targets
        
        # Compute classification loss
        class_loss = self.cls_criterion(class_preds, class_targets)
        
        # Compute box regression loss
        # First, get positive indices
        positive_idx = (class_targets > 0).nonzero(as_tuple=True)
        
        # Get positive box predictions and targets
        pos_box_preds = box_preds[positive_idx]
        pos_box_targets = box_targets[positive_idx]
        
        # Compute regression loss
        box_loss = self.box_criterion(pos_box_preds, pos_box_targets)
        
        # Normalize by number of positive samples
        num_positive = max(1, len(positive_idx[0]))
        class_loss /= num_positive
        box_loss /= num_positive
        
        # Combine losses
        total_loss = class_loss + box_loss
        
        return total_loss
