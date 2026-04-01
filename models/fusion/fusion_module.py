import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, config, feature_dims):
        super().__init__()
        self.fusion_type = config['model']['fusion']['type']
        self.hidden_dim = config['model']['fusion']['hidden_dim']
        self.num_heads = config['model']['fusion']['num_heads']
        self.dropout = config['model']['fusion']['dropout']
        
        # Feature dimensions from different extractors
        self.feature_dims = feature_dims
        
        # Projection layers for each feature extractor
        self.projections = nn.ModuleDict()
        for name, dim in feature_dims.items():
            self.projections[name] = nn.Linear(dim, self.hidden_dim)
        
        # Fusion modules
        if self.fusion_type == 'concat':
            self.fusion = self._concat_fusion
            self.fusion_output_dim = self.hidden_dim * len(feature_dims)
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
        elif self.fusion_type == 'attention':
            self.fusion = self._attention_fusion
            self.fusion_output_dim = self.hidden_dim
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(self.hidden_dim)
        elif self.fusion_type == 'hybrid':
            self.fusion = self._hybrid_fusion
            self.fusion_output_dim = self.hidden_dim
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(self.hidden_dim)
            self.norm2 = nn.LayerNorm(self.hidden_dim)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                nn.Dropout(self.dropout)
            )
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
    
    def _concat_fusion(self, features):
        # Project each feature to the same dimension
        projected_features = []
        for name, feature in features.items():
            projection = self.projections[name]
            projected = projection(feature)
            projected_features.append(projected)
        
        # Concatenate all features
        concat_features = torch.cat(projected_features, dim=1)
        
        # Process the concatenated features
        fused_features = self.fusion_layer(concat_features)
        
        return fused_features
    
    def _attention_fusion(self, features):
        # Project each feature to the same dimension
        projected_features = []
        for name, feature in features.items():
            projection = self.projections[name]
            projected = projection(feature)
            projected_features.append(projected)
        
        # Stack features for attention
        stacked_features = torch.stack(projected_features, dim=1)  # [B, N, D]
        
        # Self-attention across features
        attn_output, _ = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Add residual connection and normalize
        fused_features = self.norm(stacked_features + attn_output)
        
        # Average across the feature dimension
        fused_features = torch.mean(fused_features, dim=1)
        
        return fused_features
    
    # models/fusion/fusion_module.py (continuation)
    def _hybrid_fusion(self, features):
        # Project each feature to the same dimension
        projected_features = []
        for name, feature in features.items():
            projection = self.projections[name]
            projected = projection(feature)
            projected_features.append(projected)
        
        # Stack features for attention
        stacked_features = torch.stack(projected_features, dim=1)  # [B, N, D]
        
        # Self-attention across features
        attn_output, _ = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Add residual connection and normalize
        features_norm = self.norm1(stacked_features + attn_output)
        
        # Average across the feature dimension
        features_avg = torch.mean(features_norm, dim=1)
        
        # Apply MLP with residual connection
        mlp_output = self.fusion_mlp(features_avg)
        fused_features = self.norm2(features_avg + mlp_output)
        
        return fused_features
    
    def forward(self, feature_dict):
        return self.fusion(feature_dict)