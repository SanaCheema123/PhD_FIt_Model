import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.num_classes = config['model']['heads']['classification']['num_classes']
        self.hidden_dim = config['model']['heads']['classification']['hidden_dim']
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
    
    def forward(self, features):
        return self.classifier(features)