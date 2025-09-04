import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TinyTransformerModel(nn.Module):
    def __init__(self, input_features, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes, max_seq_len=60, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.input_projection = nn.Linear(input_features, d_model)

        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=max_seq_len)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.input_projection(src)  
        src = src * math.sqrt(self.d_model) 

        src = src.transpose(0, 1) 
        src = self.positional_encoding(src)
        src = src.transpose(0, 1)

        output = self.transformer_encoder(src)

        pooled_output = output.mean(dim=1) 

        logits = self.classifier(pooled_output)

        return logits