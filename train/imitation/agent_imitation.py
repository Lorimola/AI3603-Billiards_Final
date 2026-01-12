#!/usr/bin/env python3
"""
agent_imitation.py - 基于模仿学习的Agent

用于推理阶段，加载训练好的神经网络模型进行决策。
可以直接复制到eval目录使用。
"""

import os
import sys
import math
import random
import numpy as np
import torch

# 添加模型路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import BilliardsPolicyNetwork, BilliardsPolicyNetworkSmall


class ImitationAgent:
    """
    基于模仿学习的台球Agent
    
    使用训练好的神经网络进行快速决策。
    """
    
    def __init__(
        self,
        checkpoint_path=None,
        model_type='large',
        device='cuda',
        use_ensemble=False,
    ):
        """
        初始化Agent
        
        参数:
            checkpoint_path: 模型检查点路径
            model_type: 'large' 或 'small'
            device: 推理设备
            use_ensemble: 是否使用多模型集成
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # 加载模型
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model = self._load_model(checkpoint_path)
            print(f"[ImitationAgent] Loaded model from {checkpoint_path}")
        else:
            print(f"[ImitationAgent] No checkpoint found, using random actions")
            self.model = None
        
        # 动作边界（用于裁剪）
        self.action_bounds = {
            'V0': (0.5, 8.0),
            'phi': (0.0, 360.0),
            'theta': (0.0, 90.0),
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5),
        }
    
    def _load_model(self, checkpoint_path):
        """加载模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 获取模型配置
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
        else:
            # 默认配置
            config = {
                'state_dim': 80,
                'action_dim': 6,
                'hidden_dim': 512,
            }
        
        # 创建模型
        if self.model_type == 'xlarge':
            model = BilliardsPolicyNetwork(
                state_dim=config.get('state_dim', 80),
                action_dim=config.get('action_dim', 6),
                hidden_dim=1024,
                num_layers=12,
                num_heads=16,
                use_transformer=True,
            )
        elif self.model_type == 'large':
            model = BilliardsPolicyNetwork(
                state_dim=config.get('state_dim', 80),
                action_dim=config.get('action_dim', 6),
                hidden_dim=config.get('hidden_dim', 512),
                num_layers=8,
                use_transformer=True,
            )
        else:
            # small model - 从配置中读取 hidden_dim
            hidden_dim = config.get('hidden_dim', 256)
            # 根据 hidden_dim 推断层数
            num_layers = 8 if hidden_dim >= 512 else 4
            model = BilliardsPolicyNetworkSmall(
                state_dim=config.get('state_dim', 80),
                action_dim=config.get('action_dim', 6),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 假设checkpoint直接是state_dict
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _extract_state_features(self, balls, my_targets, table):
        """
        从游戏状态中提取特征向量
        
        与collect_data.py中的函数保持一致
        """
        features = []
        
        # 球桌尺寸
        table_w = table.w if hasattr(table, 'w') else 1.0
        table_l = table.l if hasattr(table, 'l') else 2.0
        
        # 1. 白球位置
        cue_ball = balls.get('cue')
        if cue_ball and cue_ball.state.s != 4:
            cue_pos = cue_ball.state.rvw[0]
            features.extend([
                cue_pos[0] / table_l,
                cue_pos[1] / table_w,
                0.0
            ])
        else:
            features.extend([0.5, 0.5, 1.0])
        
        # 2. 其他球的位置和状态
        for ball_id in [str(i) for i in range(1, 16)]:
            ball = balls.get(ball_id)
            if ball and ball.state.s != 4:
                pos = ball.state.rvw[0]
                features.extend([
                    pos[0] / table_l,
                    pos[1] / table_w,
                    0.0
                ])
            else:
                features.extend([0.0, 0.0, 1.0])
        
        # 3. 目标球mask
        target_set = set(my_targets)
        for ball_id in [str(i) for i in range(1, 16)]:
            if ball_id in target_set:
                features.append(1.0)
            else:
                features.append(0.0)
        
        # 4. 袋口位置
        pocket_ids = ['lb', 'lc', 'lt', 'rb', 'rc', 'rt']
        for pid in pocket_ids:
            pocket = table.pockets.get(pid)
            if pocket:
                center = pocket.center
                features.extend([
                    center[0] / table_l,
                    center[1] / table_w
                ])
            else:
                features.extend([0.0, 0.0])
        
        # 5. 统计特征
        remaining_own = sum(1 for bid in my_targets 
                          if bid in balls and balls[bid].state.s != 4 and bid != '8')
        all_balls = set(str(i) for i in range(1, 16))
        enemy_balls = all_balls - target_set - {'8'}
        remaining_enemy = sum(1 for bid in enemy_balls 
                             if bid in balls and balls[bid].state.s != 4)
        targeting_eight = 1.0 if my_targets == ['8'] else 0.0
        eight_ball = balls.get('8')
        eight_pocketed = 1.0 if (eight_ball is None or eight_ball.state.s == 4) else 0.0
        
        features.extend([
            remaining_own / 7.0,
            remaining_enemy / 7.0,
            targeting_eight,
            eight_pocketed,
            len(my_targets) / 8.0
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _decode_action(self, output):
        """
        将网络输出解码为游戏动作
        
        输入: [V0_norm, phi_sin, phi_cos, theta_norm, a_norm, b_norm]
        输出: dict with V0, phi, theta, a, b
        """
        v0_norm = output[0]
        phi_sin = output[1]
        phi_cos = output[2]
        theta_norm = output[3]
        a_norm = output[4]
        b_norm = output[5]
        
        # 反归一化
        V0 = v0_norm * 7.5 + 0.5  # [0,1] -> [0.5, 8.0]
        phi = math.degrees(math.atan2(phi_sin, phi_cos)) % 360  # rad -> deg
        theta = theta_norm * 90.0  # [0,1] -> [0, 90]
        a = a_norm * 0.5  # [-1,1] -> [-0.5, 0.5]
        b = b_norm * 0.5  # [-1,1] -> [-0.5, 0.5]
        
        # 裁剪到合法范围
        V0 = np.clip(V0, *self.action_bounds['V0'])
        theta = np.clip(theta, *self.action_bounds['theta'])
        a = np.clip(a, *self.action_bounds['a'])
        b = np.clip(b, *self.action_bounds['b'])
        
        return {
            'V0': float(V0),
            'phi': float(phi),
            'theta': float(theta),
            'a': float(a),
            'b': float(b),
        }
    
    def _random_action(self):
        """生成随机动作"""
        return {
            'V0': round(random.uniform(0.5, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3),
        }
    
    def decision(self, balls=None, my_targets=None, table=None):
        """
        决策接口
        
        参数:
            balls: 球状态字典
            my_targets: 目标球ID列表
            table: 球桌对象
        
        返回:
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        if balls is None or my_targets is None or table is None:
            print("[ImitationAgent] Missing input, using random action")
            return self._random_action()
        
        if self.model is None:
            return self._random_action()
        
        try:
            # 检查并更新目标（如果己方球已清空）
            remaining_own = [
                bid for bid in my_targets
                if bid in balls and balls[bid].state.s != 4 and bid != '8'
            ]
            if len(remaining_own) == 0 and '8' not in my_targets:
                my_targets = ['8']
                print("[ImitationAgent] Targets cleared, switching to 8-ball")
            
            # 提取特征
            state_features = self._extract_state_features(balls, my_targets, table)
            
            # 转换为tensor
            state_tensor = torch.from_numpy(state_features).float().unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                output = self.model(state_tensor)
            
            # 解码动作
            output_np = output.cpu().numpy()[0]
            action = self._decode_action(output_np)
            
            print(f"[ImitationAgent] Action: V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"theta={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            
            return action
            
        except Exception as e:
            print(f"[ImitationAgent] Error: {e}, using random action")
            import traceback
            traceback.print_exc()
            return self._random_action()


# 用于eval目录的NewAgent别名
class NewAgent(ImitationAgent):
    """
    NewAgent - 用于evaluate.py的接口
    
    自动从默认路径加载模型。
    """
    
    def __init__(self):
        # 默认检查点路径（相对于eval目录）
        default_paths = [
            './checkpoints/model_best.pt',
            './checkpoints/checkpoint_best.pt',
            '../train/imitation/checkpoints/model_best.pt',
            '../train/imitation/checkpoints/checkpoint_best.pt',
        ]
        
        checkpoint_path = None
        for path in default_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        # 尝试使用GPU，否则用CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        super().__init__(
            checkpoint_path=checkpoint_path,
            model_type='large',
            device=device,
        )


# 测试代码
if __name__ == '__main__':
    # 简单测试
    agent = ImitationAgent(
        checkpoint_path='./checkpoints/model_best.pt',
        model_type='large',
        device='cuda',
    )
    
    print(f"Agent initialized with model: {agent.model is not None}")
    
    # 测试随机动作
    action = agent._random_action()
    print(f"Random action: {action}")











