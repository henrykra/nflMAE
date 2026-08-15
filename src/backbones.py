import torch
import torch.nn as nn
import einops
import datasets
from functools import partial

# Notes
# implement the zoo architecture and sportstrackingtransformer, use the same test task
# compare the data preparation from the sportstrackigntrasnformer paper

# explore forcing hierarchy later
# - block attention
# - slot attention for transition down layers

# explore alternatives for embedding 
#   linear, mlp, kpconv

# compare versions of rpe (crpe (Stratified Transformer for 3D Point Cloud Segmentation)

# try path dropout



# B - batch dimension
# P - player dimension
# D - model feature dimension
# H - heads


class Layer(nn.Module):

    def __init__(self,
                 model_dimension: int,
                 heads: int,
                 attn_drop: float,
                 proj_drop: float,
                 add_rpb: bool=True
    ):
        super().__init__()
        
        self.d = model_dimension
        self.h = heads

        assert model_dimension % heads == 0
        
        self.qkv_proj = nn.Linear(self.d, self.d * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.scale = (self.d // self.h) ** -.5

        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(proj_drop)

        self.out_proj = nn.Linear(self.d, self.d)


        self.add_rpb = add_rpb
        self.RPB = RPB(model_dimension // 4, heads)

        # self.rpb_scale = nn.Parameter(torch.Tensor(1), requires_grad=True)

        

    def forward(self, coords, feats: torch.Tensor):

        # feats = B P D
        # coords = B P 2
        qkv = self.qkv_proj(feats) # B P 3D

        
        qkv: torch.Tensor = einops.rearrange(qkv, 'B P (x c h) -> x B h P c', x=3, h=self.h, c=self.d // self.h)
        q, k, v = qkv.unbind(dim=0)
        
        # attention
        attn = q @ k.transpose(-2, -1) * self.scale 

        # add rpb
        if self.add_rpb: 
            attn += self.RPB(self.RPB.get_relative_positions(coords))

        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)
        
        feats = attn @ v

        feats = einops.rearrange(feats, 'B H P C -> B P (H C)')
        feats = self.proj_dropout(self.out_proj(feats))

        return feats


# relative position bias
# swintransformer v2 - continuous position bias

# relative position bias (swin transformer v1) is less applicable to point tasks because
# it benefits from the regular gridwise location of image patches
class RPB(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
    ):
        super().__init__()
        self.rpb_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads, bias=False)
        )
        

    def forward(self, rel_pos: torch.Tensor):
        # log scale relative positions
        log_rel_pos = torch.sign(rel_pos) * torch.log(torch.abs(rel_pos) + 1)

        bias: torch.Tensor = self.rpb_mlp(log_rel_pos) # B P P H
        return bias.permute(0, 3, 1, 2) # B H P P


    @torch.no_grad()
    def get_relative_positions(self, coords: torch.Tensor) -> torch.Tensor: 
        # TODO: cache this, only needs to be computed once

        # B 1 P 2   -   B P 1 2  = B P P 2
        relative_coords = coords.unsqueeze(1) - coords.unsqueeze(2)

        return relative_coords
        




class MyBlock(nn.Module):

    def __init__(self,
                 model_dimension: int,
                 hidden_dimension: int,
                 heads: int,
                 attn_drop: float,
                 proj_drop: float,
                 ffn_drop: float,
                 ffn_act: nn.Module=nn.GELU,
                 add_rpb: bool=True,
                 norm_first: bool=True,
                 ):
        
        super().__init__()
        self.d = model_dimension
        self.hd = hidden_dimension

        self.ffn_act = ffn_act


        self.ln1 = nn.LayerNorm(self.d)
        self.ln2 = nn.LayerNorm(self.d)
        self.attn_layer = Layer(self.d, heads, attn_drop, proj_drop, add_rpb=add_rpb)


        self.mlp = nn.Sequential(
            nn.Linear(self.d, self.hd),
            self.ffn_act(),
            nn.Dropout(ffn_drop),
            nn.Linear(self.hd, self.d),
            nn.Dropout(ffn_drop)
        )
        self.norm_first = norm_first


    def forward(self, coords: torch.Tensor, feats: torch.Tensor):
        
        skip = feats

        if self.norm_first: 
            feats = self.ln1(feats)
    
        feats = self.attn_layer(coords, feats)
        feats = skip + feats
        if not self.norm_first: 
            feats = self.ln1(feats)

        skip = feats
        if self.norm_first: 
            feats = self.ln2(feats)

        feats = self.mlp(feats)

        feats = skip + feats
        if not self.norm_first: 
            feats = self.ln2(feats)

        return feats


class AttentionPooling(nn.Module):

    def __init__(self): 
        super().__init__()


    def forward(self, feats: torch.Tensor): 
        ...



   


class MyModel(nn.Module):

    def __init__(
            self,
            feature_channels: int,
            model_dimension: int,
            hidden_dimension: int,
            heads: int,
            attn_drop: float,
            proj_drop: float,
            ffn_drop: float,
            depth: int,
            out_channels: int,
            ffn_act: nn.Module=nn.GELU,
            add_rpb: bool=True,
            norm_first: bool=True
    ): 
        super().__init__()

        self.add_rpb = add_rpb
        self.bn = nn.BatchNorm1d(feature_channels)

        self.embed_layer = nn.Sequential(
            nn.Linear(feature_channels, model_dimension),
            nn.LayerNorm(model_dimension)
        )
        
        self.blocks = nn.ModuleList([
            MyBlock(model_dimension, hidden_dimension, heads, attn_drop, proj_drop, ffn_drop, ffn_act=ffn_act, add_rpb=add_rpb, norm_first = norm_first)
            for _ in range(depth)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.decoder = nn.Sequential(
            nn.Linear(model_dimension, model_dimension // 2),
            nn.ReLU(),
            nn.Dropout(proj_drop),
            nn.Linear(model_dimension // 2, model_dimension // 4),
            nn.ReLU(),
            nn.LayerNorm(model_dimension // 4),
            nn.Dropout(proj_drop),
            nn.Linear(model_dimension // 4, out_channels)
        )

        self.final_norm = nn.LayerNorm(model_dimension)
        self.norm_first = norm_first


    def forward(self, frame: datasets.Frame):

        coords, feats = frame.coords, frame.feats

        feats = self.bn(feats.permute(0, 2, 1)).permute(0, 2, 1)
        feats = self.embed_layer(feats)

        for block in self.blocks:
            feats = block(coords, feats)
        feats = self.final_norm(feats)
        

        # average pooling
        feats = einops.rearrange(feats, 'B P D -> B D P')
        feats = self.pool(feats).squeeze(-1)
        

        return self.decoder(feats)
        







from typing import Any

import torch
from torch import Tensor, nn, squeeze
from torch.optim import AdamW


class SportsTransformer(nn.Module):
    """
    Transformer model that treats all 22 players as a sequence for tackle prediction.
    """

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize the SportsTransformer.

        Args:
            feature_len (int): Number of input features per player.
            model_dim (int): Dimension of the model's internal representations.
            num_layers (int): Number of transformer encoder layers.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        dim_feedforward = model_dim * 4
        num_heads = min(16, max(2, 2 * round(model_dim / 64)))  # attention is better optimized for even number of heads

        self.hyperparams = {
            "model_dim": model_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dim_feedforward": dim_feedforward,
        }

        # Normalize input features
        self.feature_norm_layer = nn.BatchNorm1d(feature_len)

        # Embed input features to model dimension
        self.feature_embedding_layer = nn.Sequential(
            nn.Linear(feature_len, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )


        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.player_pooling_layer = nn.AdaptiveAvgPool1d(1)


        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(model_dim // 4),
            nn.Linear(model_dim // 4, 2),
        )

    def forward(self, frame: datasets.Frame) -> Tensor:
        """
        Forward pass of the SportsTransformer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_players, feature_len].

        Returns:
            Tensor: Predicted tackle location of shape [batch_size, 2].
        """
        # x: [B: batch_size, P: # of players, F: feature_len]
        x = frame.feats
        B, P, F = x.size()

        # Normalize features
        x = self.feature_norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B,P,F] -> [B,P,F]

        # Embed features
        x = self.feature_embedding_layer(x)  # [B,P,F] -> [B,P,M: model_dim]

        # Apply transformer encoder
        x = self.transformer_encoder(x)  # [B,P,M] -> [B,P,M]

        # Pool over player dimension
        x = squeeze(self.player_pooling_layer(x.permute(0, 2, 1)), -1)  # [B,M,P] -> [B,M]

        # Decode to predict tackle location
        x = self.decoder(x)  # [B,M] -> [B,2]

        return x
