"""Model Architecture"""
import torch
import numpy as np


class NflBERT(torch.nn.Module):
    """Bert style model for encoding nfl player movement
    
    BERT uses a nearly identical transformer encoder as "Attention is all you need" and thus
    similar to the provided torch TransformerEncoder layer"""

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        hidden_size: int=512,
        num_layers: int=8,
        num_heads: int=8,
        ffn_size: int=2048,
        ffn_act: str="gelu",
        dropout: float=.3,
    ):
        super(NflBERT, self).__init__()


        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_size = ffn_size
        self.dropout = dropout


        self.norm_layer = torch.nn.BatchNorm1d(feature_dim)
        
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Dropout(dropout)
        )

        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ffn_size,
                dropout=dropout,
                activation=ffn_act,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 4),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(hidden_size // 4),
            torch.nn.Linear(hidden_size // 4, output_dim)
        )


    
    def forward(self, x: torch.Tensor):
        
        # x: [B: batch_size, P: # of players, F: feature_len]
        B, P, F = x.size()

        # Normalize features
        x = x.permute(0, 2, 1)

        x = self.norm_layer(x).permute(0, 2, 1)  # [B,P,F] -> [B,P,F]

        # Embed features
        x = self.embed(x)  # [B,P,F] -> [B,P,M: model_dim]

        # Apply transformer encoder
        x = self.transformer(x)  # [B,P,M] -> [B,P,M]

        # Decode to predict tackle location
        x = self.decoder(x)  # [B, P, M] -> [B,P,O]

        return x


def save_model(model: torch.nn.Module, file):
    torch.save(model.state_dict(), file)

