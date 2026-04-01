
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        depth = config['model']['backbone']['resnet']['depth']
        pretrained = config['model']['backbone']['resnet']['pretrained']
        
        # Load pretrained ResNet
        if depth == 18:
            self.model = models.resnet18(pretrained=pretrained)
            self.output_dim = 512
        elif depth == 34:
            self.model = models.resnet34(pretrained=pretrained)
            self.output_dim = 512
        elif depth == 50:
            self.model = models.resnet50(pretrained=pretrained)
            self.output_dim = 2048
        elif depth == 101:
            self.model = models.resnet101(pretrained=pretrained)
            self.output_dim = 2048
        elif depth == 152:
            self.model = models.resnet152(pretrained=pretrained)
            self.output_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
        
        # Remove classification head
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        
        # Hook to store intermediate features
        self.intermediate_features = []
        
        def hook_fn(module, input, output):
            self.intermediate_features.append(output)
        
        # Register hooks for intermediate features
        self.model[4].register_forward_hook(hook_fn)  # layer1
        self.model[5].register_forward_hook(hook_fn)  # layer2
        self.model[6].register_forward_hook(hook_fn)  # layer3
        self.model[7].register_forward_hook(hook_fn)  # layer4
    
    def forward(self, x):
        self.intermediate_features = []
        features = self.model(x)
        return features, self.intermediate_features