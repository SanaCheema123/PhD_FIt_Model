import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationLoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, predictions, targets):
        # Predictions: [batch_size, num_classes]
        # Targets: [batch_size]
        
        loss = self.criterion(predictions, targets)
        
        return loss