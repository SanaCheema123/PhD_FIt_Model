import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = self._dice_loss
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def _dice_loss(self, predictions, targets):
        # Sigmoid activation
        predictions = torch.sigmoid(predictions)
        
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection) / (union + 1e-8)
        
        # Return Dice loss
        return 1.0 - dice
    
    def forward(self, predictions, targets):
        # Predictions: [mask_pred, class_pred]
        # Targets: [mask_target, class_target]
        
        mask_pred, class_pred = predictions
        mask_target, class_target = targets
        
        # Mask loss (combination of BCE and Dice)
        mask_loss = self.bce_loss(mask_pred, mask_target) + self.dice_loss(mask_pred, mask_target)
        
        # Class loss (if applicable)
        class_loss = 0.0
        if class_pred is not None and class_target is not None:
            class_loss = F.cross_entropy(class_pred, class_target)
        
        # Total loss
        total_loss = mask_loss + class_loss
        
        return total_loss