import torch
import torch.nn as nn
import torch.nn.functional as F

class GroundingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.SmoothL1Loss(reduction='mean')
    
    def forward(self, predictions, targets):
        # Predictions: [batch_size, 4] (x, y, w, h)
        # Targets: [batch_size, 4] (x, y, w, h)
        
        loss = self.criterion(predictions, targets)
        
        return loss
