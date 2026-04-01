import torch
import torch.nn as nn
import yaml
from typing import Dict, List, Tuple, Any

from .feature_extractors import CNNFeatureExtractor, ResNetFeatureExtractor, SwinFeatureExtractor, ViTFeatureExtractor
from .fusion import FusionModule
from .heads import CaptioningHead, ClassificationHead, DetectionHead, GroundingHead, SegmentationHead

class MultitaskModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize feature extractors
        self.feature_extractors = nn.ModuleDict()
        feature_dims = {}
        
        if config['model']['backbone']['swin']['use']:
            self.feature_extractors['swin'] = SwinFeatureExtractor(config)
            # models/multitask_model.py (continuing from where it left off)
            feature_dims['swin'] = self.feature_extractors['swin'].output_dim
        
        if config['model']['backbone']['vit']['use']:
            self.feature_extractors['vit'] = ViTFeatureExtractor(config)
            feature_dims['vit'] = self.feature_extractors['vit'].output_dim
        
        if config['model']['backbone']['resnet']['use']:
            self.feature_extractors['resnet'] = ResNetFeatureExtractor(config)
            feature_dims['resnet'] = self.feature_extractors['resnet'].output_dim
        
        if config['model']['backbone']['cnn']['use']:
            self.feature_extractors['cnn'] = CNNFeatureExtractor(config)
            feature_dims['cnn'] = self.feature_extractors['cnn'].output_dim
        
        # Initialize fusion module
        self.fusion = FusionModule(config, feature_dims)
        fusion_dim = self.fusion.fusion_output_dim
        
        # Initialize task-specific heads
        self.heads = nn.ModuleDict()
        self.heads['captioning'] = CaptioningHead(config, fusion_dim)
        self.heads['classification'] = ClassificationHead(config, fusion_dim)
        self.heads['detection'] = DetectionHead(config, fusion_dim)
        self.heads['grounding'] = GroundingHead(config, fusion_dim)
        self.heads['segmentation'] = SegmentationHead(config, fusion_dim)
        
        # Task sampling probabilities
        self.task_probs = {
            'captioning': config['training']['task_sampling']['captioning'],
            'grounding': config['training']['task_sampling']['grounding'],
            'segmentation': config['training']['task_sampling']['segmentation'],
            'classification': config['training']['task_sampling']['classification'],
            'detection': config['training']['task_sampling']['detection']
        }
    
    def extract_features(self, images):
        # Extract features from all enabled backbones
        features = {}
        intermediate_features = {}
        
        for name, extractor in self.feature_extractors.items():
            feat, inter_feat = extractor(images)
            features[name] = feat
            intermediate_features[name] = inter_feat
        
        return features, intermediate_features
    
    def forward(self, batch, task=None):
        # Extract features
        features, intermediate_features = self.extract_features(batch['image'])
        
        # Fuse features
        fused_features = self.fusion(features)
        
        # If task is not specified, sample randomly based on task probabilities
        if task is None and self.training:
            tasks = list(self.task_probs.keys())
            probs = list(self.task_probs.values())
            task = torch.multinomial(torch.tensor(probs), 1).item()
            task = tasks[task]
        elif task is None:
            # During inference, run all tasks
            outputs = {}
            for task_name in self.heads.keys():
                outputs[task_name] = self.forward_task(task_name, batch, fused_features, intermediate_features)
            return outputs
        
        # Forward through the selected task head
        return self.forward_task(task, batch, fused_features, intermediate_features)
    
    def forward_task(self, task, batch, fused_features, intermediate_features=None):
        if task == 'captioning':
            if 'caption' in batch:
                return self.heads['captioning'](fused_features, batch['caption'])
            else:
                return self.heads['captioning'](fused_features)
        
        elif task == 'classification':
            return self.heads['classification'](fused_features)
        
        elif task == 'detection':
            return self.heads['detection'](fused_features)
        
        elif task == 'grounding':
            if 'query' in batch:
                # Placeholder for text features
                text_features = torch.zeros(batch['image'].shape[0], 10, 300, device=batch['image'].device)
                return self.heads['grounding'](fused_features, text_features)
            else:
                return None
        
        elif task == 'segmentation':
            return self.heads['segmentation'](fused_features)
        
        else:
            raise ValueError(f"Unknown task: {task}")