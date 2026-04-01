import torch
import torch.nn as nn
import timm

class SwinFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config['model']['backbone']['swin']['embed_dim']
        depths = config['model']['backbone']['swin']['depths']
        num_heads = config['model']['backbone']['swin']['num_heads']
        pretrained = config['model']['backbone']['swin']['pretrained']
        
        # Determine model size based on parameters
        if embed_dim == 96 and depths == [2, 2, 6, 2]:
            model_name = 'swin_tiny_patch4_window7_224'
            self.output_dim = 768
        elif embed_dim == 128 and depths == [2, 2, 18, 2]:
            model_name = 'swin_small_patch4_window7_224'
            self.output_dim = 1024
        elif embed_dim == 128 and depths == [2, 2, 18, 2]:
            model_name = 'swin_base_patch4_window7_224'
            self.output_dim = 1024
        elif embed_dim == 192 and depths == [2, 2, 18, 2]:
            model_name = 'swin_large_patch4_window7_224'
            self.output_dim = 1536
        else:
            raise ValueError(f"Unsupported Swin Transformer configuration")
        
        # Load model
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
    
    def forward(self, x):
        features = self.model(x)
        return features[-1], features