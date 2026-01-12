#!/usr/bin/env python3
"""
evaluate_imitation.py - 评估模仿学习Agent

测试训练好的神经网络Agent与BasicAgent的对战表现。
"""

import os
import sys
import argparse
import random
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

from agent_imitation import ImitationAgent


def set_random_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description='Evaluate imitation learning agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='large',
                        choices=['xlarge', 'large', 'small'],
                        help='Model type')
    parser.add_argument('--num_games', type=int, default=120,
                        help='Number of games to play')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 导入环境和BasicAgent
    try:
        sys.path.insert(0, '/home/chjin/AI/AI3603-Billiards/long_mcts')
        from poolenv import PoolEnv
        from agent import BasicAgent
    except ImportError:
        from poolenv import PoolEnv
        from agent import BasicAgent
    
    # 创建环境
    env = PoolEnv()
    
    # 创建Agents
    agent_a = BasicAgent()  # 基准Agent
    agent_b = ImitationAgent(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
    )
    
    # 统计结果
    results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
    
    players = [agent_a, agent_b]
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']
    
    print(f"\n{'='*60}")
    print(f"Evaluating ImitationAgent vs BasicAgent")
    print(f"Games: {args.num_games}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")
    
    for i in range(args.num_games):
        print(f"\n------- Game {i+1}/{args.num_games} -------")
        
        env.reset(target_ball=target_ball_choice[i % 4])
        
        player_class = players[i % 2].__class__.__name__
        ball_type = target_ball_choice[i % 4]
        print(f"Player A: {player_class}, Ball type: {ball_type}")
        
        step_count = 0
        max_steps = 100
        
        while step_count < max_steps:
            player = env.get_curr_player()
            obs = env.get_observation(player)
            
            if player == 'A':
                action = players[i % 2].decision(*obs)
            else:
                action = players[(i + 1) % 2].decision(*obs)
            
            try:
                env.take_shot(action)
            except Exception as e:
                print(f"Error in take_shot: {e}")
                break
            
            done, info = env.get_done()
            if done:
                # 统计结果
                if info['winner'] == 'SAME':
                    results['SAME'] += 1
                elif info['winner'] == 'A':
                    results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
                else:
                    results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]] += 1
                break
            
            step_count += 1
        
        # 中间结果
        total = results['AGENT_A_WIN'] + results['AGENT_B_WIN'] + results['SAME']
        if total > 0:
            agent_b_score = results['AGENT_B_WIN'] + 0.5 * results['SAME']
            win_rate = agent_b_score / total * 100
            print(f"Current: A={results['AGENT_A_WIN']}, B={results['AGENT_B_WIN']}, "
                  f"Same={results['SAME']} | B win rate: {win_rate:.1f}%")
    
    # 最终结果
    results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] + 0.5 * results['SAME']
    results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] + 0.5 * results['SAME']
    
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"{'='*60}")
    print(f"Agent A (BasicAgent): {results['AGENT_A_WIN']} wins, {results['AGENT_A_SCORE']:.1f} score")
    print(f"Agent B (Imitation):  {results['AGENT_B_WIN']} wins, {results['AGENT_B_SCORE']:.1f} score")
    print(f"Draws: {results['SAME']}")
    print(f"\nAgent B Win Rate: {results['AGENT_B_SCORE'] / args.num_games * 100:.2f}%")
    print(f"{'='*60}")
    
    return results


if __name__ == '__main__':
    main()




