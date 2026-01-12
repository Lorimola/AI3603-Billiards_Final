#!/usr/bin/env python3
"""
evaluate_discrete.py - 评估离散 phi 模仿学习 Agent
"""

import os
import sys
import argparse
import random
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))
sys.path.insert(0, '/home/chjin/AI/AI3603-Billiards/long_mcts')

from agent_discrete import DiscreteImitationAgent, DiscreteImitationAgentWithSimulation


def set_random_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description='Evaluate discrete phi imitation agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='large',
                        choices=['large', 'small'],
                        help='Model type')
    parser.add_argument('--num_games', type=int, default=40,
                        help='Number of games to play')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_top_k', action='store_true',
                        help='Use top-k candidates')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top-k candidates')
    parser.add_argument('--use_simulation', action='store_true',
                        help='Use simulation verification for top-k candidates')
    parser.add_argument('--sim_per_candidate', type=int, default=3,
                        help='Number of simulations per candidate')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 导入环境和BasicAgent
    from poolenv import PoolEnv
    from agent import BasicAgent
    
    # 创建Agents
    agent_a = BasicAgent()  # 基准Agent
    
    if args.use_simulation:
        # 使用模拟验证的 Agent
        agent_b = DiscreteImitationAgentWithSimulation(
            model_path=args.checkpoint,
            model_type=args.model_type,
            device=args.device,
            top_k=args.top_k,
            simulation_per_candidate=args.sim_per_candidate,
        )
        agent_name = "DiscreteSimAgent"
    else:
        agent_b = DiscreteImitationAgent(
            model_path=args.checkpoint,
            model_type=args.model_type,
            device=args.device,
            use_top_k=args.use_top_k,
            top_k=args.top_k,
        )
        agent_name = "DiscreteImitationAgent"
    
    # 统计结果
    results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
    
    players = [agent_a, agent_b]
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']
    
    print(f"\n{'='*60}")
    print(f"Evaluating {agent_name} vs BasicAgent")
    print(f"Games: {args.num_games}")
    print(f"Checkpoint: {args.checkpoint}")
    if args.use_simulation:
        print(f"Top-K: {args.top_k}, Simulations: {args.sim_per_candidate}")
    print(f"{'='*60}\n")
    
    for i in range(args.num_games):
        print(f"\n------- Game {i+1}/{args.num_games} -------")
        
        env = PoolEnv()
        env.reset(target_ball=target_ball_choice[i % 4])
        
        # 重置 Agent 状态
        agent_b.my_targets = None
        
        player_class = players[i % 2].__class__.__name__
        ball_type = target_ball_choice[i % 4]
        print(f"First Player: {player_class}, Ball type: {ball_type}")
        
        step_count = 0
        max_steps = 100
        
        while step_count < max_steps:
            player = env.get_curr_player()
            obs = env.get_observation(player)
            
            balls, my_targets, table = obs
            
            if player == 'A':
                if i % 2 == 0:  # BasicAgent goes first
                    action = agent_a.decision(*obs)
                else:  # DiscreteImitationAgent goes first
                    action = agent_b.make_decision(balls, my_targets, table)
            else:
                if i % 2 == 0:  # DiscreteImitationAgent is player B
                    action = agent_b.make_decision(balls, my_targets, table)
                else:  # BasicAgent is player B
                    action = agent_a.decision(*obs)
            
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
                    # Player A wins
                    if i % 2 == 0:
                        results['AGENT_A_WIN'] += 1  # BasicAgent wins
                    else:
                        results['AGENT_B_WIN'] += 1  # DiscreteImitationAgent wins
                else:
                    # Player B wins
                    if i % 2 == 0:
                        results['AGENT_B_WIN'] += 1  # DiscreteImitationAgent wins
                    else:
                        results['AGENT_A_WIN'] += 1  # BasicAgent wins
                break
            
            step_count += 1
        
        # 中间结果
        total = results['AGENT_A_WIN'] + results['AGENT_B_WIN'] + results['SAME']
        if total > 0:
            agent_b_score = results['AGENT_B_WIN'] + 0.5 * results['SAME']
            win_rate = agent_b_score / total * 100
            print(f"Current: Basic={results['AGENT_A_WIN']}, Discrete={results['AGENT_B_WIN']}, "
                  f"Draw={results['SAME']} | Discrete win rate: {win_rate:.1f}%")
    
    # 最终结果
    results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] + 0.5 * results['SAME']
    results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] + 0.5 * results['SAME']
    
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"{'='*60}")
    print(f"BasicAgent:            {results['AGENT_A_WIN']} wins, {results['AGENT_A_SCORE']:.1f} score")
    print(f"DiscreteImitationAgent: {results['AGENT_B_WIN']} wins, {results['AGENT_B_SCORE']:.1f} score")
    print(f"Draws: {results['SAME']}")
    print(f"\nDiscreteImitationAgent Win Rate: {results['AGENT_B_SCORE'] / args.num_games * 100:.2f}%")
    print(f"{'='*60}")
    
    return results


if __name__ == '__main__':
    main()

