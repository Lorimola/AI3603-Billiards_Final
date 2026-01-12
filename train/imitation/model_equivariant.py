#!/usr/bin/env python3
"""
model_equivariant.py - Equivariant Set Attention Network for Billiards

Implements a permutation-equivariant architecture that explicitly encodes
the symmetry that ball ordering should not affect the output.

Key components:
1. BallEncoder: Shared MLP that independently processes each ball's features
2. SetAttentionBlocks: Self-attention over ball set (permutation-equivariant)
3. CrossAttention: Cue ball queries object balls
4. Target-weighted pooling: Aggregate using target mask as attention weights
5. Separate prediction heads: V0, discrete φ (72 bins), θ, spin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for permutation-equivariant processing"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, C] where N is the number of balls
        Returns:
            out: [B, N, C] (permutation-equivariant)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.dropout(x)


class SetAttentionBlock(nn.Module):
    """Set Attention block - maintains permutation equivariance"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        """x: [B, N, C] - maintains permutation equivariance"""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    """Cross-attention: cue ball queries object balls"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim)  # query from cue ball
        self.kv = nn.Linear(dim, 2 * dim)  # key-value from object balls
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, cue_embed, object_embeds):
        """
        Args:
            cue_embed: [B, C] - cue ball embedding
            object_embeds: [B, N, C] - object ball embeddings
        Returns:
            out: [B, C] - attended cue ball representation
        """
        B, N, C = object_embeds.shape
        
        # Query from cue ball
        q = self.q(cue_embed).unsqueeze(1).reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Key-Value from object balls
        kv = self.kv(object_embeds).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum
        x = (attn @ v).transpose(1, 2).reshape(B, C)
        x = self.proj(x)
        return self.dropout(x)


class EquivariantBilliardsNetwork(nn.Module):
    """
    Equivariant Set Attention Network for Billiards
    
    Architecture:
    1. BallEncoder: Shared MLP processes each ball independently
    2. SetAttentionBlocks: 8 layers of self-attention (permutation-equivariant)
    3. CrossAttention: Cue ball queries object balls
    4. Target-weighted pooling: Aggregate using target mask
    5. Prediction heads: V0, discrete φ (72 bins), θ, spin
    """
    
    def __init__(
        self,
        state_dim=80,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        num_phi_bins=72,  # 5 degrees per bin
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_phi_bins = num_phi_bins
        
        # Ball feature extraction
        # Each ball: 3D position (x, y, pocketed) + 1D target mask = 4D
        self.ball_dim = 4
        self.num_balls = 16  # cue ball + 15 object balls
        
        # BallEncoder: Shared MLP that processes each ball independently
        self.ball_encoder = nn.Sequential(
            nn.Linear(self.ball_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # SetAttentionBlocks: Permutation-equivariant self-attention
        self.set_attention_blocks = nn.ModuleList([
            SetAttentionBlock(hidden_dim, num_heads, mlp_ratio=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # CrossAttention: Cue ball queries object balls
        self.cross_attention = CrossAttention(hidden_dim, num_heads, dropout)
        self.cross_attention_norm = nn.LayerNorm(hidden_dim)
        
        # Shared feature processing
        self.shared_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Prediction heads
        # Velocity head (V0)
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Phi head (discrete classification, 72 bins)
        self.phi_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_phi_bins),
        )
        
        # Theta head
        self.theta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Spin head (a, b)
        self.spin_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh(),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def _extract_ball_features(self, x):
        """
        Extract ball features from state representation
        
        Args:
            x: [B, 80] state representation
        Returns:
            cue_ball_feat: [B, 4] - cue ball features
            object_ball_feat: [B, 15, 4] - object ball features
            target_mask: [B, 15] - target ball mask
        """
        batch_size = x.shape[0]
        
        # Cue ball: [0:3] = (x, y, pocketed)
        cue_ball_pos = x[:, 0:3]
        cue_ball_feat = torch.cat([cue_ball_pos, torch.zeros(batch_size, 1, device=x.device)], dim=1)
        
        # Object balls: [3:48] = 15 balls * (x, y, pocketed)
        # Target mask: [48:63] = 15 binary flags
        object_ball_feat_list = []
        target_mask = x[:, 48:63]  # [B, 15]
        
        for i in range(15):
            ball_pos = x[:, 3 + i*3 : 3 + (i+1)*3]  # [B, 3]
            ball_target = target_mask[:, i:i+1]  # [B, 1]
            ball_feat = torch.cat([ball_pos, ball_target], dim=1)  # [B, 4]
            object_ball_feat_list.append(ball_feat.unsqueeze(1))
        
        object_ball_feat = torch.cat(object_ball_feat_list, dim=1)  # [B, 15, 4]
        
        return cue_ball_feat, object_ball_feat, target_mask
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, 80] state representation
        Returns:
            dict with:
                - 'v0': [B, 1]
                - 'phi_logits': [B, num_phi_bins]
                - 'theta': [B, 1]
                - 'spin': [B, 2]
        """
        # Extract ball features
        cue_ball_feat, object_ball_feat, target_mask = self._extract_ball_features(x)
        
        # Encode all balls (cue + objects) independently
        # Cue ball
        cue_embed = self.ball_encoder(cue_ball_feat)  # [B, hidden_dim]
        
        # Object balls
        B, N, _ = object_ball_feat.shape
        object_ball_feat_flat = object_ball_feat.view(B * N, -1)  # [B*15, 4]
        object_embeds_flat = self.ball_encoder(object_ball_feat_flat)  # [B*15, hidden_dim]
        object_embeds = object_embeds_flat.view(B, N, self.hidden_dim)  # [B, 15, hidden_dim]
        
        # Stack all balls: [cue, ball1, ball2, ..., ball15]
        all_ball_embeds = torch.cat([cue_embed.unsqueeze(1), object_embeds], dim=1)  # [B, 16, hidden_dim]
        
        # SetAttentionBlocks: Permutation-equivariant processing
        set_out = all_ball_embeds
        for block in self.set_attention_blocks:
            set_out = block(set_out)  # [B, 16, hidden_dim]
        
        # Separate cue and object representations
        cue_attended = set_out[:, 0, :]  # [B, hidden_dim]
        object_attended = set_out[:, 1:, :]  # [B, 15, hidden_dim]
        
        # CrossAttention: Cue ball queries object balls
        cue_cross = self.cross_attention(cue_attended, object_attended)  # [B, hidden_dim]
        cue_cross = self.cross_attention_norm(cue_attended + cue_cross)  # Residual connection
        
        # Target-weighted pooling: Aggregate object balls using target mask
        # target_mask: [B, 15] -> [B, 15, 1]
        target_weights = target_mask.unsqueeze(2)  # [B, 15, 1]
        target_weights = target_weights / (target_weights.sum(dim=1, keepdim=True) + 1e-8)  # Normalize
        
        object_pooled = (object_attended * target_weights).sum(dim=1)  # [B, hidden_dim]
        
        # Combine cue and object information
        combined = cue_cross + object_pooled  # [B, hidden_dim]
        combined = self.shared_layers(combined)  # [B, hidden_dim]
        
        # Prediction heads
        v0 = self.velocity_head(combined)  # [B, 1]
        phi_logits = self.phi_head(combined)  # [B, num_phi_bins]
        theta = self.theta_head(combined)  # [B, 1]
        spin = self.spin_head(combined)  # [B, 2]
        
        return {
            'v0': v0,
            'phi_logits': phi_logits,
            'theta': theta,
            'spin': spin,
        }


def create_equivariant_model(num_phi_bins=72, hidden_dim=512, num_layers=8, num_heads=8, dropout=0.1):
    """Factory function to create Equivariant model"""
    return EquivariantBilliardsNetwork(
        state_dim=80,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        num_phi_bins=num_phi_bins,
    )


# Test
if __name__ == '__main__':
    model = create_equivariant_model()
    x = torch.randn(4, 80)  # Batch size 4
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"V0 shape: {out['v0'].shape}")
    print(f"Phi logits shape: {out['phi_logits'].shape}")
    print(f"Theta shape: {out['theta'].shape}")
    print(f"Spin shape: {out['spin'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
