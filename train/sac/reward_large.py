"""
reward_large.py - 大奖励范围版本

复现早期版本的奖励设计：
- 终局奖励：±1000
- 进球奖励：100×比例
- 平局惩罚：-1000（与失败相同）
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class RewardCalculatorLarge:
    """
    SAC 奖励计算器 - 大奖励范围版本（复现早期实验）
    
    奖励范围：[-1000, +1000]
    
    组成：
    - R_terminal: 终局奖励 [±1000]
    - R_pocket: 进球奖励 [0, +100]
    - R_foul: 犯规惩罚 [-25, 0]
    """
    
    def __init__(self,
                 C1: float = 100.0,  # 己方进球基础价值
                 C2: float = 100.0,  # 对方进球基础价值
                 C3: float = 10.0,   # 球权维持价值
                 win_reward: float = 1000.0,
                 loss_reward: float = -1000.0,
                 draw_reward: float = -1000.0,  # 平局 = 失败
                 table_w: float = 0.99,
                 table_l: float = 1.98):
        
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.draw_reward = draw_reward
        
        self.table_w = table_w
        self.table_l = table_l
        
        # 犯规惩罚（来自原始配置）
        self.foul_penalties = {
            'WHITE_BALL_INTO_POCKET': -20.0,
            'FOUL_FIRST_HIT': -15.0,
            'NO_POCKET_NO_RAIL': -10.0,
            'NO_HIT': -25.0,
        }
        
        # 失去球权惩罚
        self.lose_turn_penalty = -5.0
    
    def compute_reward(self,
                       shot_result: Dict,
                       balls_before: Dict,
                       balls_after: Dict,
                       my_targets: List[str],
                       enemy_targets: List[str],
                       table,
                       game_done: bool,
                       winner: Optional[str],
                       my_player: str,
                       is_my_shot: bool = True) -> Tuple[float, Dict]:
        """
        计算单步奖励（大奖励范围版本）
        """
        reward_details = {}
        total_reward = 0.0
        
        # 1. 终局奖励（最高优先级）
        if game_done:
            terminal_r = self._compute_terminal_reward(winner, my_player)
            reward_details['terminal'] = terminal_r
            return terminal_r, reward_details
        
        # 2. 进球奖励（比例制）
        my_pocketed = shot_result.get('ME_INTO_POCKET', [])
        enemy_pocketed = shot_result.get('ENEMY_INTO_POCKET', [])
        my_balls_before = len([b for b in balls_before.values() 
                              if any(t in str(b) for t in my_targets)])
        enemy_balls_before = len([b for b in balls_before.values() 
                                 if any(t in str(b) for t in enemy_targets)])
        
        # 己方进球价值
        if my_pocketed and my_balls_before > 0:
            my_value = (len(my_pocketed) / my_balls_before) * self.C1
            total_reward += my_value
            reward_details['my_pocket'] = my_value
        
        # 对方进球惩罚
        if enemy_pocketed and enemy_balls_before > 0:
            enemy_value = (len(enemy_pocketed) / enemy_balls_before) * self.C2
            total_reward -= enemy_value
            reward_details['enemy_pocket'] = -enemy_value
        
        # 3. 球权价值
        if my_pocketed:
            total_reward += self.C3
            reward_details['keep_turn'] = self.C3
        else:
            total_reward += self.lose_turn_penalty
            reward_details['lose_turn'] = self.lose_turn_penalty
        
        # 4. 犯规惩罚
        foul_penalty = 0.0
        for foul_type, penalty in self.foul_penalties.items():
            if shot_result.get(foul_type, False):
                foul_penalty += penalty
        if foul_penalty != 0:
            total_reward += foul_penalty
            reward_details['foul'] = foul_penalty
        
        reward_details['total'] = total_reward
        return total_reward, reward_details
    
    def _compute_terminal_reward(self, winner: Optional[str], my_player: str) -> float:
        """终局奖励"""
        if winner == 'SAME':  # 平局
            return self.draw_reward  # -1000，与失败相同
        elif winner == my_player:
            return self.win_reward  # +1000
        elif winner is not None:
            return self.loss_reward  # -1000
        else:
            return 0.0





