
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroundingHead(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.hidden_dim = config['model']['heads']['grounding']['hidden_dim']
        self.dropout = config['model']['heads']['grounding']['dropout']
        
        # Text encoder (placeholder - in real implementation would use BERT/RoBERTa)
        self.text_encoder = nn.LSTM(
            input_size=300,  # Assuming 300-dim word embeddings
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.text_projection = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        
        # Visual feature projection
        self.visual_projection = nn.Linear(input_dim, self.hidden_dim)
        
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Bounding box regression
        self.bbox_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 4)  # [x, y, w, h]
        )
    
    def forward(self, features, text_features):
        batch_size = features.shape[0]
        
        # Process text features (placeholder)
        # In a real implementation, you would encode text properly
        text_output, _ = self.text_encoder(text_features)
        text_emb = self.text_projection(text_output[:, -1])  # Use final hidden state
        
        # Project visual features
        visual_emb = self.visual_projection(features)
        
        # Cross-modal attention
        # Reshape visual features for attention
        visual_emb_seq = visual_emb.unsqueeze(1)  # [B, 1, D]
        
        # Attention between text and visual features
        attn_output, _ = self.attention(
            query=text_emb.unsqueeze(1),
            key=visual_emb_seq,
            value=visual_emb_seq
        )
        
        # Predict bounding box
        bbox = self.bbox_predictor(attn_output.squeeze(1))
        
        return bbox
