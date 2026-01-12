#!/usr/bin/env python3
"""
model_discrete.py - 离散化 phi 的台球策略网络

将 phi (0-360°) 离散化为 N 个 bin，使用分类损失替代 MSE
解决多模态动作的"平均化"问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力"""
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
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Transformer块"""
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
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BilliardsPolicyNetworkDiscrete(nn.Module):
    """
    离散化 phi 的策略网络
    
    输出:
    - V0: [0, 1] (连续)
    - phi: 36个bin的logits (离散分类)
    - theta: [0, 1] (连续)
    - a, b: [-1, 1] (连续)
    """
    
    def __init__(
        self,
        state_dim=80,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        num_phi_bins=36,  # 每10度一个bin
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_phi_bins = num_phi_bins
        self.action_dim = 1 + num_phi_bins + 1 + 2  # V0 + phi_bins + theta + spin
        
        # 球特征维度
        self.ball_dim = 4
        self.num_balls = 16
        
        # 嵌入层
        self.ball_embed = nn.Linear(self.ball_dim, hidden_dim)
        self.extra_dim = 12 + 5
        self.extra_embed = nn.Linear(self.extra_dim, hidden_dim)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_balls + 1, hidden_dim) * 0.02)
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 共享特征处理
        self.shared_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # 分离的输出头
        # 速度头 (V0) - 连续
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # 角度头 (phi) - 离散分类
        self.phi_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加 LayerNorm 稳定训练
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_phi_bins),  # 输出 logits
        )
        
        # 俯仰角头 (theta) - 连续
        self.theta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # 旋杆头 (a, b) - 连续
        self.spin_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh(),
        )
        
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
        """提取球特征"""
        batch_size = x.shape[0]
        
        cue_features = torch.cat([x[:, 0:3], torch.zeros(batch_size, 1, device=x.device)], dim=1)
        ball_features_list = [cue_features.unsqueeze(1)]
        
        for i in range(15):
            ball_pos = x[:, 3 + i*3 : 3 + (i+1)*3]
            ball_target = x[:, 48 + i : 48 + i + 1]
            ball_feat = torch.cat([ball_pos, ball_target], dim=1)
            ball_features_list.append(ball_feat.unsqueeze(1))
        
        ball_features = torch.cat(ball_features_list, dim=1)
        extra_features = x[:, 63:80]
        
        return ball_features, extra_features
    
    def forward(self, x):
        """
        前向传播
        
        返回: dict with
            - 'v0': [B, 1]
            - 'phi_logits': [B, num_phi_bins]
            - 'theta': [B, 1]
            - 'spin': [B, 2]
        """
        ball_features, extra_features = self._extract_ball_features(x)
        
        ball_embed = self.ball_embed(ball_features)
        extra_embed = self.extra_embed(extra_features).unsqueeze(1)
        
        tokens = torch.cat([ball_embed, extra_embed], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        
        for block in self.transformer_blocks:
            tokens = block(tokens)
        
        features = tokens.mean(dim=1)
        features = self.shared_layers(features)
        
        v0 = self.velocity_head(features)
        phi_logits = self.phi_head(features)
        # Clamp logits 防止数值溢出
        phi_logits = torch.clamp(phi_logits, min=-20.0, max=20.0)
        theta = self.theta_head(features)
        spin = self.spin_head(features)
        
        return {
            'v0': v0,
            'phi_logits': phi_logits,
            'theta': theta,
            'spin': spin,
        }
    
    def predict_action(self, state_features, temperature=1.0, sample=False):
        """
        推理时使用
        
        参数:
            state_features: [B, 80] 或 [80]
            temperature: 采样温度
            sample: 是否采样（False则用argmax）
        """
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)
        
        # 判断是否单输入
        single_input = (state_features.shape[0] == 1)
        
        with torch.no_grad():
            output = self.forward(state_features)
        
        actions = []
        for i in range(state_features.shape[0]):
            v0_norm = output['v0'][i, 0].item()
            phi_logits = output['phi_logits'][i]
            theta_norm = output['theta'][i, 0].item()
            a_norm = output['spin'][i, 0].item()
            b_norm = output['spin'][i, 1].item()
            
            # 从 phi logits 获取角度
            if sample:
                probs = F.softmax(phi_logits / temperature, dim=0)
                phi_bin = torch.multinomial(probs, 1).item()
            else:
                phi_bin = phi_logits.argmax().item()
            
            # bin 中心角度
            bin_size = 360.0 / self.num_phi_bins
            phi = (phi_bin + 0.5) * bin_size  # bin 中心
            
            # 反归一化
            V0 = v0_norm * 7.5 + 0.5
            theta = theta_norm * 90.0
            a = a_norm * 0.5
            b = b_norm * 0.5
            
            actions.append({
                'V0': float(V0),
                'phi': float(phi),
                'theta': float(theta),
                'a': float(a),
                'b': float(b),
                'phi_probs': F.softmax(phi_logits, dim=0).cpu().numpy(),
            })
        
        if single_input:
            return actions[0]
        return actions
    
    def get_top_k_actions(self, state_features, k=5):
        """
        获取 top-k 个候选动作
        
        用于后续 MCTS 重排
        """
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)
        
        single_input = (state_features.shape[0] == 1)
        
        with torch.no_grad():
            output = self.forward(state_features)
        
        results = []
        for i in range(state_features.shape[0]):
            v0_norm = output['v0'][i, 0].item()
            phi_logits = output['phi_logits'][i]
            theta_norm = output['theta'][i, 0].item()
            a_norm = output['spin'][i, 0].item()
            b_norm = output['spin'][i, 1].item()
            
            # Top-k phi bins
            probs = F.softmax(phi_logits, dim=0)
            top_probs, top_bins = torch.topk(probs, k)
            
            bin_size = 360.0 / self.num_phi_bins
            V0 = v0_norm * 7.5 + 0.5
            theta = theta_norm * 90.0
            a = a_norm * 0.5
            b = b_norm * 0.5
            
            candidates = []
            for j in range(k):
                phi = (top_bins[j].item() + 0.5) * bin_size
                candidates.append({
                    'V0': float(V0),
                    'phi': float(phi),
                    'theta': float(theta),
                    'a': float(a),
                    'b': float(b),
                    'prob': float(top_probs[j].item()),
                })
            
            results.append(candidates)
        
        if single_input:
            return results[0]
        return results


class BilliardsPolicyNetworkDiscreteSmall(nn.Module):
    """轻量级离散 phi 网络"""
    
    def __init__(
        self,
        state_dim=80,
        hidden_dim=256,
        num_layers=4,
        dropout=0.1,
        num_phi_bins=36,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_phi_bins = num_phi_bins
        self.action_dim = 1 + num_phi_bins + 1 + 2
        
        # 主干网络
        layers = [
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        
        self.backbone = nn.Sequential(*layers)
        
        # 输出头
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        self.phi_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 稳定训练
            nn.GELU(),
            nn.Linear(hidden_dim, num_phi_bins),
        )
        
        self.theta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        self.spin_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh(),
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        phi_logits = self.phi_head(features)
        # Clamp logits 防止数值溢出
        phi_logits = torch.clamp(phi_logits, min=-20.0, max=20.0)
        
        return {
            'v0': self.velocity_head(features),
            'phi_logits': phi_logits,
            'theta': self.theta_head(features),
            'spin': self.spin_head(features),
        }
    
    def predict_action(self, state_features, temperature=1.0, sample=False):
        """与大模型相同接口"""
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)
        
        single_input = (state_features.shape[0] == 1)
        
        with torch.no_grad():
            output = self.forward(state_features)
        
        actions = []
        for i in range(state_features.shape[0]):
            v0_norm = output['v0'][i, 0].item()
            phi_logits = output['phi_logits'][i]
            theta_norm = output['theta'][i, 0].item()
            a_norm = output['spin'][i, 0].item()
            b_norm = output['spin'][i, 1].item()
            
            if sample:
                probs = F.softmax(phi_logits / temperature, dim=0)
                phi_bin = torch.multinomial(probs, 1).item()
            else:
                phi_bin = phi_logits.argmax().item()
            
            bin_size = 360.0 / self.num_phi_bins
            phi = (phi_bin + 0.5) * bin_size
            
            V0 = v0_norm * 7.5 + 0.5
            theta = theta_norm * 90.0
            a = a_norm * 0.5
            b = b_norm * 0.5
            
            actions.append({
                'V0': float(V0),
                'phi': float(phi),
                'theta': float(theta),
                'a': float(a),
                'b': float(b),
            })
        
        if single_input:
            return actions[0]
        return actions


def create_discrete_model(model_type='large', **kwargs):
    """创建离散 phi 模型"""
    if model_type == 'large':
        return BilliardsPolicyNetworkDiscrete(**kwargs)
    elif model_type == 'small':
        return BilliardsPolicyNetworkDiscreteSmall(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 工具函数
def angle_to_bin(angle_deg, num_bins=36):
    """将角度转换为 bin index"""
    angle_deg = angle_deg % 360
    bin_size = 360.0 / num_bins
    return int(angle_deg / bin_size)


def bin_to_angle(bin_idx, num_bins=36):
    """将 bin index 转换为角度（bin 中心）"""
    bin_size = 360.0 / num_bins
    return (bin_idx + 0.5) * bin_size


def sincos_to_bin(phi_sin, phi_cos, num_bins=36):
    """将 sin/cos 编码转换为 bin index"""
    angle_rad = math.atan2(phi_sin, phi_cos)
    angle_deg = math.degrees(angle_rad) % 360
    return angle_to_bin(angle_deg, num_bins)


if __name__ == '__main__':
    # 测试
    model = BilliardsPolicyNetworkDiscrete(
        state_dim=80,
        hidden_dim=512,
        num_layers=8,
        num_phi_bins=36,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(4, 80)
    output = model(x)
    print(f"Output shapes:")
    print(f"  v0: {output['v0'].shape}")
    print(f"  phi_logits: {output['phi_logits'].shape}")
    print(f"  theta: {output['theta'].shape}")
    print(f"  spin: {output['spin'].shape}")
    
    action = model.predict_action(x[0])
    print(f"\nPredicted action: {action}")
    
    top_k = model.get_top_k_actions(x[0], k=5)
    print(f"\nTop-5 candidates:")
    for i, c in enumerate(top_k):
        print(f"  {i+1}. phi={c['phi']:.1f}°, prob={c['prob']:.3f}")

