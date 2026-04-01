import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationHead(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.decoder_dim = config['model']['heads']['segmentation']['decoder_dim']
        self.mask_classification = config['model']['heads']['segmentation']['mask_classification']
        
        # Feature projection
        self.input_projection = nn.Conv2d(input_dim, self.decoder_dim, kernel_size=1)
        
        # Simple decoder (upsampling + convolutions)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.decoder_dim, self.decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.decoder_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(self.decoder_dim, self.decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.decoder_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(self.decoder_dim, self.decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.decoder_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(self.decoder_dim, self.decoder_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.decoder_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.decoder_dim // 2, 1, kernel_size=1)  # Binary mask by default
        )
        
        # Optional classification head for panoptic segmentation
        if self.mask_classification:
            self.class_embed = nn.Linear(input_dim, 80)  # 80 COCO classes
    
    def forward(self, features):
        batch_size = features.shape[0]
        
        # Convert to spatial feature map
        if len(features.shape) == 2:
            # Convert 1D features to 2D feature map
            h = w = int(features.shape[1] ** 0.5)
            features = features.view(batch_size, -1, h, w)
        
        # Project features
        proj_features = self.input_projection(features)
        
        # Generate mask
        mask = self.decoder(proj_features)
        
        # Optional class prediction
        class_pred = None
        if self.mask_classification:
            # Global average pooling
            pooled = F.adaptive_avg_pool2d(features, (1, 1)).view(batch_size, -1)
            class_pred = self.class_embed(pooled)
        
        return mask, class_pred