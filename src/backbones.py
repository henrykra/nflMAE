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
                 add_rpb: bool=True,
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

        self.rel_cls_token = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.sid_value = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.causal = True

        # self.rpb_scale = nn.Parameter(torch.Tensor(1), requires_grad=True)

        

    def forward(self, feat: torch.Tensor, coord, fids, pids):
        B, P, D = feat.shape
        if coord.shape[1] != feat.shape[1]: 

            coord = torch.cat([self.rel_cls_token[None, None, :].repeat(B, 1, 2), coord], dim=1)
        # feat = B P D
        # coords = B P 2
        qkv = self.qkv_proj(feat) # B P 3D

        
        qkv: torch.Tensor = einops.rearrange(qkv, 'B P (x c h) -> x B h P c', x=3, h=self.h, c=self.d // self.h)
        q, k, v = qkv.unbind(dim=0)

        if fids is not None: 
            if fids.shape[-1] != feat.shape[1]: 
                frameid = fids[:, -1:]
                nfids = torch.cat((frameid, fids), dim=-1)

            q, k = self.add_rope(q, k, nfids) # TODO: precompute sins/cosines


        # attention
        attn = q @ k.transpose(-2, -1) * self.scale 

        # add rpb
        if self.add_rpb: 
            attn += self.RPB(self.RPB.get_relative_positions(coord))


        if pids is not None:
            if pids.shape[1] != feat.shape[1]:
                pids = torch.cat( (torch.zeros(B, 1, device='mps'), pids), dim=-1)
            sid_map = self.get_sid_mask(pids)
            attn += sid_map.unsqueeze(1) * self.sid_value

        if self.causal and fids is not None: 
            if fids.shape[-1] != feat.shape[1]:     
                frameid = fids[:, :1]
                nfids = torch.cat((frameid, fids), dim=-1)
            causal_mask = self.get_causal_mask(nfids)
            attn += causal_mask.unsqueeze(1)


        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)
        
        feat = attn @ v

        feat = einops.rearrange(feat, 'B H P C -> B P (H C)')
        feat = self.proj_dropout(self.out_proj(feat))

        return feat


    @staticmethod
    def add_rope(q, k, fids: torch.Tensor):
        b, h, p, hd = q.shape

        assert fids.shape == (b,p), f"{fids.shape}, ({b}, {p})"
        frequencies = 10000 ** -(torch.arange(0, hd, 2, dtype=torch.float, device='mps') / hd)

        position_frequency = fids.unsqueeze(-1) * frequencies.unsqueeze(0)
        position_encoding = torch.repeat_interleave(position_frequency, torch.ones(hd//2, dtype=torch.int, device="mps")*2, dim=-1)
        # b, p, hd

        q_rot = q * position_encoding.unsqueeze(1).cos() + rotate_half(q) * position_encoding.unsqueeze(1).sin()
        k_rot = k * position_encoding.unsqueeze(1).cos() + rotate_half(k) * position_encoding.unsqueeze(1).sin()

        return q_rot, k_rot


    @staticmethod
    def get_sid_mask(pids): 
        # b, 1, p - b, p, 1 = b, p, p
        sid_mask = (pids.unsqueeze(1) - pids.unsqueeze(2)) == 0
        return sid_mask


    @staticmethod
    def get_causal_mask(fids): 
        causal_mask = ((fids.unsqueeze(1) - fids.unsqueeze(2)) > 0)

        causal_mask = torch.where(causal_mask, -torch.inf, 0)

        return causal_mask






def rotate_half(u: torch.Tensor): 
    u = einops.rearrange(u, '... (d r) -> ... d r', r=2)
    u1, u2 = u.unbind(dim=-1)
    u = torch.stack((u2, -u1), dim=-1)

    return einops.rearrange(u, '... d r -> ... (d r)', r=2)


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


    def forward(self, feat: torch.Tensor, coord: torch.Tensor, fids: torch.Tensor, pids: torch.Tensor):
        
        skip = feat

        if self.norm_first: 
            feat = self.ln1(feat)
    
        feat = self.attn_layer(feat, coord, fids, pids)
        feat = skip + feat
        if not self.norm_first: 
            feat = self.ln1(feat)

        skip = feat
        if self.norm_first: 
            feat = self.ln2(feat)

        feat = self.mlp(feat)

        feat = skip + feat
        if not self.norm_first: 
            feat = self.ln2(feat)

        return feat

   


class MyModel(nn.Module):

    def __init__(
            self,
            model_dimension: int,
            hidden_dimension: int,
            heads: int,
            attn_drop: float,
            proj_drop: float,
            ffn_drop: float,
            depth: int,
            add_rpb: bool=True,
            norm_first: bool=True, 
            add_cls_token: bool=False,
            add_rope: bool=False,
            add_sids: bool=False,
    ): 
        super().__init__()

        self.add_rpb = add_rpb
        
        self.model_dimension = model_dimension
        
        self.blocks = nn.ModuleList([
            MyBlock(model_dimension, hidden_dimension, heads, attn_drop, proj_drop, ffn_drop, norm_first=norm_first, add_rpb=add_rpb)
            for _ in range(depth)
        ])


        self.final_norm = nn.LayerNorm(model_dimension)
        self.norm_first = norm_first
        self.add_cls_token = add_cls_token
        if self.add_cls_token: 
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dimension), requires_grad=True)

        self.add_rope = add_rope
        self.add_sids = add_sids


    def forward(self, feat, coord=None, fids=None, pids=None):

        # feat = self.bn(feat.permute(0, 2, 1)).permute(0, 2, 1)
        # feat = self.embed_layer(feat) - assming already embedded
        if self.add_cls_token: 
            feat = torch.cat([self.cls_token.expand(feat.shape[0], -1, -1), feat], dim=1)

        for block in self.blocks:
            feat = block(feat, coord, fids, pids)
        feat = self.final_norm(feat)
        

        return feat







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

    def forward(self, frame) -> Tensor:
        """
        Forward pass of the SportsTransformer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_players, feature_len].

        Returns:
            Tensor: Predicted tackle location of shape [batch_size, 2].
        """
        # x: [B: batch_size, P: # of players, F: feature_len]
        x = frame.feat
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
