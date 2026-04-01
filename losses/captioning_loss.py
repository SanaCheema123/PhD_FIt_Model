
import torch
import torch.nn as nn
import torch.nn.functional as F

class CaptioningLoss(nn.Module):
    def __init__(self, pad_idx=0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    def forward(self, predictions, targets):
        # Predictions: [batch_size, seq_len, vocab_size]
        # Targets: [batch_size, seq_len]
        
        batch_size = predictions.shape[0]
        
        # Reshape predictions for cross-entropy
        predictions = predictions.view(-1, predictions.shape[-1])
        targets = targets.view(-1)
        
        # Calculate loss
        loss = self.criterion(predictions, targets)
        
        return loss
