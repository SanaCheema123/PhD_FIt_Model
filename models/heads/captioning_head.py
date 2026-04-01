import torch
import torch.nn as nn
import torch.nn.functional as F

class CaptioningHead(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.vocab_size = config['model']['heads']['captioning']['vocab_size']
        self.max_seq_len = config['model']['heads']['captioning']['max_seq_len']
        self.embedding_dim = config['model']['heads']['captioning']['embedding_dim']
        self.hidden_dim = config['model']['heads']['captioning']['hidden_dim']
        self.num_layers = config['model']['heads']['captioning']['num_layers']
        
        # Word embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # Image feature projection
        self.img_projection = nn.Linear(input_dim, self.hidden_dim)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim, self.vocab_size)
    
    def forward(self, features, captions=None, teacher_forcing_ratio=0.5):
        batch_size = features.shape[0]
        
        # Project image features
        img_features = self.img_projection(features)
        
        # Initialize hidden state with image features
        h0 = img_features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        # Training mode (with teacher forcing)
        if self.training and captions is not None:
            # Embed caption inputs
            embedded = self.embedding(captions[:, :-1])  # Exclude <eos>
            
            # LSTM decoding with image features as initial state
            outputs, _ = self.lstm(embedded, (h0, c0))
            
            # Project to vocabulary
            outputs = self.output_layer(outputs)
            return outputs
        
        # Inference mode
        else:
            # Start with <sos> token
            input_seq = torch.ones(batch_size, 1, dtype=torch.long, device=features.device)
            
            # Initialize output tensor
            outputs = torch.zeros(batch_size, self.max_seq_len, self.vocab_size, device=features.device)
            
            # Hidden states
            hidden = (h0, c0)
            
            # Generate sequence
            for t in range(self.max_seq_len):
                # Embed current input
                embedded = self.embedding(input_seq)
                
                # LSTM step
                output, hidden = self.lstm(embedded, hidden)
                
                # Project to vocabulary
                output = self.output_layer(output)
                outputs[:, t:t+1] = output
                
                # Get next input (greedy decoding)
                input_seq = output.argmax(2)
            
            return outputs