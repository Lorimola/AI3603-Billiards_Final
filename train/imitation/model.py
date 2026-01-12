#!/usr/bin/env python3
"""
model.py - 台球策略网络模型

为RTX 6000 Ada设计的大型网络架构，支持:
- 深度残差网络
- 注意力机制
- 多头输出（速度、角度、旋杆）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return x + self.dropout(self.net(x))


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


class BilliardsPolicyNetwork(nn.Module):
    """
    台球策略网络 - 大型版本
    
    设计原则:
    1. 使用Transformer处理球的位置信息（具有排列不变性的潜力）
    2. 分离不同类型的输出头（速度、角度、旋杆）
    3. 支持RTX 6000 Ada的48GB显存，充分利用GPU能力
    
    输入: 80维状态特征
    输出: 6维动作 [V0, phi_sin, phi_cos, theta, a, b]
    """
    
    def __init__(
        self,
        state_dim=80,
        action_dim=6,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        use_transformer=True,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_transformer = use_transformer
        
        # 输入嵌入
        self.input_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        if use_transformer:
            # 将状态分解为球的token（适合自注意力）
            # 白球(3) + 15球(45) + 目标mask(15) + 袋口(12) + 统计(5) = 80
            # 我们把每个球作为一个token: 3(位置+状态) + 1(是否目标) = 4维
            self.ball_dim = 4
            self.num_balls = 16  # 白球 + 15个彩球
            
            # 球特征投影
            self.ball_embed = nn.Linear(self.ball_dim, hidden_dim)
            
            # 额外特征（袋口+统计）投影
            self.extra_dim = 12 + 5  # 袋口(12) + 统计(5)
            self.extra_embed = nn.Linear(self.extra_dim, hidden_dim)
            
            # 位置编码
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_balls + 1, hidden_dim) * 0.02)
            
            # Transformer层
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(hidden_dim, num_heads, mlp_ratio=4, dropout=dropout)
                for _ in range(num_layers)
            ])
            
            # 全局聚合
            self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        else:
            # 纯MLP版本（作为备选）
            self.mlp_blocks = nn.ModuleList([
                ResidualBlock(hidden_dim, dropout)
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
        # 速度头 (V0)
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # [0, 1]
        )
        
        # 角度头 (phi_sin, phi_cos, theta)
        self.angle_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh(),  # phi: [-1, 1], theta需要后处理
        )
        
        # 旋杆头 (a, b)
        self.spin_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh(),  # [-1, 1]
        )
        
        # 初始化
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
        从80维状态中提取球特征
        
        输入格式:
        - [0:3] 白球 (pos_x, pos_y, pocketed)
        - [3:48] 15球 (每球3维: pos_x, pos_y, pocketed)
        - [48:63] 目标mask (15维)
        - [63:75] 袋口位置 (6*2=12维)
        - [75:80] 统计特征 (5维)
        """
        batch_size = x.shape[0]
        
        # 白球特征: 位置(2) + 进袋(1) + 是否目标(始终为0)
        cue_features = torch.cat([x[:, 0:3], torch.zeros(batch_size, 1, device=x.device)], dim=1)
        
        # 彩球特征
        ball_features_list = [cue_features.unsqueeze(1)]
        
        for i in range(15):
            ball_pos = x[:, 3 + i*3 : 3 + (i+1)*3]  # 位置+进袋
            ball_target = x[:, 48 + i : 48 + i + 1]  # 是否目标
            ball_feat = torch.cat([ball_pos, ball_target], dim=1)
            ball_features_list.append(ball_feat.unsqueeze(1))
        
        ball_features = torch.cat(ball_features_list, dim=1)  # [B, 16, 4]
        
        # 额外特征
        extra_features = x[:, 63:80]  # 袋口 + 统计
        
        return ball_features, extra_features
    
    def forward(self, x):
        """
        前向传播
        
        输入: x [batch_size, 80]
        输出: [batch_size, 6] - [V0, phi_sin, phi_cos, theta, a, b]
        """
        batch_size = x.shape[0]
        
        if self.use_transformer:
            # 提取球特征
            ball_features, extra_features = self._extract_ball_features(x)
            
            # 嵌入
            ball_embed = self.ball_embed(ball_features)  # [B, 16, hidden]
            extra_embed = self.extra_embed(extra_features).unsqueeze(1)  # [B, 1, hidden]
            
            # 拼接所有token
            tokens = torch.cat([ball_embed, extra_embed], dim=1)  # [B, 17, hidden]
            
            # 添加位置编码
            tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
            
            # Transformer处理
            for block in self.transformer_blocks:
                tokens = block(tokens)
            
            # 全局平均池化
            features = tokens.mean(dim=1)  # [B, hidden]
        
        else:
            # MLP处理
            features = self.input_embed(x)
            for block in self.mlp_blocks:
                features = block(features)
        
        # 共享层
        features = self.shared_layers(features)
        
        # 分离输出头
        v0 = self.velocity_head(features)  # [B, 1]
        angles = self.angle_head(features)  # [B, 3]
        spin = self.spin_head(features)  # [B, 2]
        
        # 对theta进行后处理（确保非负）
        phi_sin = angles[:, 0:1]
        phi_cos = angles[:, 1:2]
        theta = torch.sigmoid(angles[:, 2:3])  # [0, 1]
        
        # 拼接输出
        output = torch.cat([v0, phi_sin, phi_cos, theta, spin], dim=1)
        
        return output
    
    def predict_action(self, state_features):
        """
        推理时使用，将网络输出转换为游戏动作
        
        输入: state_features [batch_size, 80] 或 [80]
        输出: dict with V0, phi, theta, a, b
        """
        single_input = False
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)
            single_input = True
        
        with torch.no_grad():
            output = self.forward(state_features)
        
        # 解码输出
        actions = []
        for i in range(output.shape[0]):
            v0_norm = output[i, 0].item()
            phi_sin = output[i, 1].item()
            phi_cos = output[i, 2].item()
            theta_norm = output[i, 3].item()
            a_norm = output[i, 4].item()
            b_norm = output[i, 5].item()
            
            # 反归一化
            V0 = v0_norm * 7.5 + 0.5  # [0,1] -> [0.5, 8.0]
            phi = math.degrees(math.atan2(phi_sin, phi_cos)) % 360  # rad -> deg
            theta = theta_norm * 90.0  # [0,1] -> [0, 90]
            a = a_norm * 0.5  # [-1,1] -> [-0.5, 0.5]
            b = b_norm * 0.5  # [-1,1] -> [-0.5, 0.5]
            
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


class BilliardsPolicyNetworkSmall(nn.Module):
    """
    轻量级策略网络 - 用于快速推理
    """
    
    def __init__(
        self,
        state_dim=80,
        action_dim=6,
        hidden_dim=256,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
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
        
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.net = nn.Sequential(*layers)
        
        # 输出激活
        self.v0_activation = nn.Sigmoid()
        self.angle_activation = nn.Tanh()
        self.spin_activation = nn.Tanh()
    
    def forward(self, x):
        out = self.net(x)
        
        # 分别处理各输出
        v0 = self.v0_activation(out[:, 0:1])
        phi_sin = self.angle_activation(out[:, 1:2])
        phi_cos = self.angle_activation(out[:, 2:3])
        theta = torch.sigmoid(out[:, 3:4])
        spin = self.spin_activation(out[:, 4:6])
        
        return torch.cat([v0, phi_sin, phi_cos, theta, spin], dim=1)
    
    def predict_action(self, state_features):
        """与大模型相同的接口"""
        single_input = False
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)
            single_input = True
        
        with torch.no_grad():
            output = self.forward(state_features)
        
        actions = []
        for i in range(output.shape[0]):
            v0_norm = output[i, 0].item()
            phi_sin = output[i, 1].item()
            phi_cos = output[i, 2].item()
            theta_norm = output[i, 3].item()
            a_norm = output[i, 4].item()
            b_norm = output[i, 5].item()
            
            V0 = v0_norm * 7.5 + 0.5
            phi = math.degrees(math.atan2(phi_sin, phi_cos)) % 360
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


def create_model(model_type='large', **kwargs):
    """
    工厂函数创建模型
    
    model_type: 'xlarge' | 'large' | 'small'
    """
    if model_type == 'xlarge':
        # 超大模型 ~100M+ 参数
        return BilliardsPolicyNetwork(
            hidden_dim=1024,
            num_layers=12,
            num_heads=16,
            dropout=0.1,
            use_transformer=True,
            **kwargs
        )
    elif model_type == 'large':
        return BilliardsPolicyNetwork(**kwargs)
    elif model_type == 'small':
        return BilliardsPolicyNetworkSmall(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 测试
if __name__ == '__main__':
    # 测试大模型
    model = BilliardsPolicyNetwork(
        state_dim=80,
        hidden_dim=512,
        num_layers=8,
        use_transformer=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    x = torch.randn(4, 80)
    y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    
    # 测试动作预测
    action = model.predict_action(x[0])
    print(f"Predicted action: {action}")




