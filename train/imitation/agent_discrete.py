#!/usr/bin/env python3
"""
agent_discrete.py - 离散 phi 的推理 Agent

使用训练好的离散 phi 模型进行推理
支持 Top-K 候选 + 简单重排
"""

import os
import sys
import math
import numpy as np
import torch

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_discrete import create_discrete_model, BilliardsPolicyNetworkDiscrete


class DiscreteImitationAgent:
    """
    离散 phi 的模仿学习 Agent
    
    特点:
    - phi 使用离散分类，避免多模态平均化
    - 支持输出 Top-K 候选动作
    - 可选择 argmax 或采样
    """
    
    def __init__(
        self,
        model_path,
        model_type='large',
        device='cuda',
        use_top_k=False,
        top_k=5,
        temperature=1.0,
    ):
        """
        参数:
            model_path: 模型权重路径
            model_type: 'large' 或 'small'
            device: 'cuda' 或 'cpu'
            use_top_k: 是否使用 Top-K 候选
            top_k: 候选数量
            temperature: 采样温度
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_top_k = use_top_k
        self.top_k = top_k
        self.temperature = temperature
        
        # 加载模型
        self.model = self._load_model(model_path, model_type)
        self.model.eval()
        
        # 目标球追踪
        self.my_targets = None
        
        print(f"[DiscreteAgent] Loaded model from {model_path}")
        print(f"[DiscreteAgent] Device: {self.device}, Top-K: {use_top_k}")
    
    def _load_model(self, model_path, model_type):
        """加载模型"""
        # 尝试加载检查点
        if model_path.endswith('.pt'):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                # 完整检查点
                config = checkpoint.get('model_config', {})
                state_dim = config.get('state_dim', 80)
                hidden_dim = config.get('hidden_dim', 512)
                num_phi_bins = config.get('num_phi_bins', 36)
                
                model = create_discrete_model(
                    model_type,
                    state_dim=state_dim,
                    hidden_dim=hidden_dim,
                    num_phi_bins=num_phi_bins,
                )
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 纯权重
                model = create_discrete_model(model_type, state_dim=80)
                model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unsupported model file: {model_path}")
        
        model = model.to(self.device)
        return model
    
    def _extract_state_features(self, balls, my_targets, table):
        """提取状态特征 (与 collect_data.py 一致)"""
        features = []
        
        table_w = table.w if hasattr(table, 'w') else 1.0
        table_l = table.l if hasattr(table, 'l') else 2.0
        
        # 白球
        cue_ball = balls.get('cue')
        if cue_ball and cue_ball.state.s != 4:
            cue_pos = cue_ball.state.rvw[0]
            features.extend([cue_pos[0] / table_l, cue_pos[1] / table_w, 0.0])
        else:
            features.extend([0.5, 0.5, 1.0])
        
        # 15个球
        for ball_id in [str(i) for i in range(1, 16)]:
            ball = balls.get(ball_id)
            if ball and ball.state.s != 4:
                pos = ball.state.rvw[0]
                features.extend([pos[0] / table_l, pos[1] / table_w, 0.0])
            else:
                features.extend([0.0, 0.0, 1.0])
        
        # 目标球 mask
        target_set = set(my_targets)
        for ball_id in [str(i) for i in range(1, 16)]:
            features.append(1.0 if ball_id in target_set else 0.0)
        
        # 袋口位置
        pocket_ids = ['lb', 'lc', 'lt', 'rb', 'rc', 'rt']
        for pid in pocket_ids:
            pocket = table.pockets.get(pid)
            if pocket:
                center = pocket.center
                features.extend([center[0] / table_l, center[1] / table_w])
            else:
                features.extend([0.0, 0.0])
        
        # 统计特征
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
    
    def make_decision(self, balls, my_targets, table):
        """
        做出决策
        
        返回: dict with V0, phi, theta, a, b
        """
        # 更新目标
        if self.my_targets is None:
            self.my_targets = my_targets.copy()
        else:
            remaining_own = [bid for bid in self.my_targets 
                           if bid in balls and balls[bid].state.s != 4]
            if len(remaining_own) == 0 and '8' not in self.my_targets:
                self.my_targets = ['8']
        
        # 提取特征
        state_features = self._extract_state_features(balls, self.my_targets, table)
        state_tensor = torch.from_numpy(state_features).float().unsqueeze(0).to(self.device)
        
        # 推理
        if self.use_top_k:
            candidates = self.model.get_top_k_actions(state_tensor, k=self.top_k)
            # 简单策略：选择概率最高的
            action = candidates[0]
        else:
            action = self.model.predict_action(state_tensor, temperature=self.temperature)
        
        return {
            'V0': action['V0'],
            'phi': action['phi'],
            'theta': action['theta'],
            'a': action['a'],
            'b': action['b'],
        }
    
    def get_top_k_candidates(self, balls, my_targets, table):
        """获取 Top-K 候选动作"""
        if self.my_targets is None:
            self.my_targets = my_targets.copy()
        
        state_features = self._extract_state_features(balls, self.my_targets, table)
        state_tensor = torch.from_numpy(state_features).float().unsqueeze(0).to(self.device)
        
        return self.model.get_top_k_actions(state_tensor, k=self.top_k)


class DiscreteImitationAgentWithSimulation(DiscreteImitationAgent):
    """
    带模拟验证的离散 phi Agent
    
    使用 Top-K 候选 + 物理模拟验证选择最佳动作
    """
    
    def __init__(
        self,
        model_path,
        model_type='large',
        device='cuda',
        top_k=5,
        simulation_per_candidate=3,
    ):
        super().__init__(
            model_path=model_path,
            model_type=model_type,
            device=device,
            use_top_k=True,
            top_k=top_k,
        )
        self.simulation_per_candidate = simulation_per_candidate
        
        # 延迟导入
        self._pt = None
        self._analyze_shot_for_reward = None
        
        print(f"[SimAgent] Top-K: {top_k}, Simulations per candidate: {simulation_per_candidate}")
    
    def _lazy_import(self):
        """延迟导入物理引擎"""
        if self._pt is None:
            import pooltool as pt
            from agent import analyze_shot_for_reward
            self._pt = pt
            self._analyze_shot_for_reward = analyze_shot_for_reward
    
    def _save_balls_state(self, balls):
        """保存球状态"""
        import copy
        return {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
    
    def _simulate_action(self, balls, table, action, my_targets):
        """
        模拟一个动作并返回奖励
        """
        self._lazy_import()
        pt = self._pt
        
        try:
            # 创建 shot
            cue = pt.Cue(
                V0=action['V0'],
                phi=action['phi'],
                theta=action['theta'],
                a=action['a'],
                b=action['b'],
            )
            
            # 创建系统
            system = pt.System(cue=cue, table=table, balls=balls)
            
            # 保存状态
            last_state = self._save_balls_state(balls)
            
            # 模拟
            pt.simulate(system, inplace=True)
            
            # 计算奖励
            reward = self._analyze_shot_for_reward(system, last_state, my_targets)
            
            return reward
            
        except Exception as e:
            return -100  # 模拟失败
    
    def make_decision(self, balls, my_targets, table):
        """
        使用 Top-K + 模拟验证做决策
        """
        # 更新目标
        if self.my_targets is None:
            self.my_targets = my_targets.copy()
        else:
            remaining_own = [bid for bid in self.my_targets 
                           if bid in balls and balls[bid].state.s != 4]
            if len(remaining_own) == 0 and '8' not in self.my_targets:
                self.my_targets = ['8']
        
        # 获取 Top-K 候选
        candidates = self.get_top_k_candidates(balls, self.my_targets, table)
        
        if len(candidates) == 1:
            # 只有一个候选，直接返回
            action = candidates[0]
            return {
                'V0': action['V0'],
                'phi': action['phi'],
                'theta': action['theta'],
                'a': action['a'],
                'b': action['b'],
            }
        
        # 模拟验证每个候选
        best_action = None
        best_score = float('-inf')
        
        for candidate in candidates:
            total_reward = 0
            
            for _ in range(self.simulation_per_candidate):
                # 添加微小噪声
                noisy_action = {
                    'V0': candidate['V0'] + np.random.normal(0, 0.05),
                    'phi': candidate['phi'] + np.random.normal(0, 0.5),
                    'theta': candidate['theta'] + np.random.normal(0, 0.5),
                    'a': candidate['a'] + np.random.normal(0, 0.01),
                    'b': candidate['b'] + np.random.normal(0, 0.01),
                }
                
                # 裁剪到合法范围
                noisy_action['V0'] = np.clip(noisy_action['V0'], 0.5, 8.0)
                noisy_action['phi'] = noisy_action['phi'] % 360
                noisy_action['theta'] = np.clip(noisy_action['theta'], 0, 90)
                noisy_action['a'] = np.clip(noisy_action['a'], -0.5, 0.5)
                noisy_action['b'] = np.clip(noisy_action['b'], -0.5, 0.5)
                
                reward = self._simulate_action(balls, table, noisy_action, self.my_targets)
                total_reward += reward
            
            avg_reward = total_reward / self.simulation_per_candidate
            
            # 加上概率权重
            score = avg_reward + 10 * candidate.get('prob', 0)
            
            if score > best_score:
                best_score = score
                best_action = candidate
        
        return {
            'V0': best_action['V0'],
            'phi': best_action['phi'],
            'theta': best_action['theta'],
            'a': best_action['a'],
            'b': best_action['b'],
        }


def test_agent():
    """测试 agent"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--model_type', type=str, default='large')
    parser.add_argument('--num_games', type=int, default=10)
    args = parser.parse_args()
    
    # 尝试导入环境
    try:
        from poolenv import PoolEnv
        from agent import BasicAgent
    except ImportError:
        print("[ERROR] Cannot import PoolEnv or BasicAgent")
        print("[INFO] Please run from project root or set PYTHONPATH")
        return
    
    agent = DiscreteImitationAgent(
        model_path=args.model,
        model_type=args.model_type,
        use_top_k=True,
        top_k=5,
    )
    
    opponent = BasicAgent()
    
    wins = 0
    losses = 0
    draws = 0
    
    for game_id in range(args.num_games):
        env = PoolEnv()
        obs = env.reset()
        done = False
        
        agent.my_targets = None
        
        while not done:
            current_player = env.current_player
            balls = obs['balls']
            table = obs['table']
            
            if current_player == 0:
                my_targets = obs['target_balls']
                action = agent.make_decision(balls, my_targets, table)
            else:
                action = opponent.make_decision(obs)
            
            obs, reward, done, info = env.step(action)
        
        if reward > 0:
            wins += 1
            result = "WIN"
        elif reward < 0:
            losses += 1
            result = "LOSS"
        else:
            draws += 1
            result = "DRAW"
        
        print(f"Game {game_id + 1}: {result}")
    
    total = wins + losses + draws
    print(f"\nResults: {wins}W / {losses}L / {draws}D")
    print(f"Win rate: {wins/total:.1%}")


if __name__ == '__main__':
    test_agent()


