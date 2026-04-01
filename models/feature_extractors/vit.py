
import torch
import torch.nn as nn
import timm
from transformers import ViTModel

class ViTFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_size = config['model']['backbone']['vit']['model_size']
        pretrained = config['model']['backbone']['vit']['pretrained']
        
        # Map model size to model name and output dimension
        model_map = {
            'tiny': ('vit_tiny_patch16_224', 192),
            'small': ('vit_small_patch16_224', 384),
            'base': ('vit_base_patch16_224', 768),
            'large': ('vit_large_patch16_224', 1024)
        }
        
        model_name, self.output_dim = model_map.get(model_size, ('vit_base_patch16_224', 768))
        
        # Load model
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False
        )
        
        # Remove classification head
        self.model.head = nn.Identity()
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Get features from ViT
        features = self.model(x)
        
        # For compatibility with other extractors, create a fake list of intermediate features
        # Each level with progressively smaller spatial dimensions
        h, w = x.shape[2] // 4, x.shape[3] // 4
        intermediate_features = [
            features[:, :, None, None].expand(-1, -1, h, w),
            features[:, :, None, None].expand(-1, -1, h//2, w//2),
            features[:, :, None, None].expand(-1, -1, h//4, w//4),
            features[:, :, None, None].expand(-1, -1, h//8, w//8)
        ]
        
        return features, intermediate_features