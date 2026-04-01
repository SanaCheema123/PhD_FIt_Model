
import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionHead(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.num_classes = config['model']['heads']['detection']['num_classes']
        self.anchor_sizes = config['model']['heads']['detection']['anchor_sizes']
        self.aspect_ratios = config['model']['heads']['detection']['aspect_ratios']
        
        # Number of anchors per location
        self.num_anchors = len(self.anchor_sizes) * len(self.aspect_ratios)
        
        # Feature pyramid network - simplified for example
        self.fpn = nn.ModuleDict({
            f'p{i}': nn.Conv2d(input_dim, 256, kernel_size=1)
            for i in range(3, 8)  # P3-P7
        })
        
        # Classification subnet
        self.cls_subnet = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, self.num_anchors * self.num_classes, kernel_size=3, padding=1)
        )
        
        # Box regression subnet
        self.box_subnet = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, self.num_anchors * 4, kernel_size=3, padding=1)
        )
    
    def forward(self, features):
        # For simplicity, assume features is already processed through FPN
        # and we're working with a single feature map
        features = features.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        features = F.interpolate(features, size=(14, 14))  # Resize to feature map size
        
        # Process through FPN (simplified)
        fpn_features = self.fpn['p4'](features)
        
        # Get class predictions and box predictions
        class_preds = self.cls_subnet(fpn_features)
        box_preds = self.box_subnet(fpn_features)
        
        # Reshape predictions
        batch_size = features.shape[0]
        class_preds = class_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        
        return class_preds, box_preds