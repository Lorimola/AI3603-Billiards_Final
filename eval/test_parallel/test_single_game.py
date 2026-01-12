"""
test_single_game.py - 单局测试脚本（带完整输出，用于验证逻辑）
"""

import os
import sys

# 添加路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'time_limit_mcts'))

from poolenv import PoolEnv
from agent import BasicAgent, NewAgent

def main():
    print("=" * 60)
    print("  单局测试：NewAgent (merge_basic) vs BasicAgent")
    print("=" * 60)
    
    env = PoolEnv(enable_render=False)
    
    # 创建 Agent
    agent_new = NewAgent()  # merge_basic 的 MCTS agent
    agent_basic = BasicAgent()
    
    print(f"\nAgent A: NewAgent (merge_basic)")
    print(f"Agent B: BasicAgent")
    
    results = {'NEW_WIN': 0, 'BASIC_WIN': 0, 'SAME': 0}
    
    # 运行 4 局（覆盖所有先后手和球型组合）
    for game_id in range(4):
        players = [agent_new, agent_basic]  # [NewAgent, BasicAgent]
        target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']
        
        print(f"\n{'='*60}")
        print(f"第 {game_id} 局")
        print(f"  球型: {target_ball_choice[game_id % 4]}")
        
        # game_id % 2 决定谁先手
        if game_id % 2 == 0:
            print(f"  Player A (先手): NewAgent")
            print(f"  Player B: BasicAgent")
        else:
            print(f"  Player A (先手): BasicAgent")
            print(f"  Player B: NewAgent")
        
        env.reset(target_ball=target_ball_choice[game_id % 4])
        
        hit_count = 0
        while True:
            player = env.get_curr_player()
            obs = env.get_observation(player)
            
            # 决定使用哪个 agent
            if player == 'A':
                agent_idx = game_id % 2
            else:
                agent_idx = (game_id + 1) % 2
            
            agent_name = "NewAgent" if agent_idx == 0 else "BasicAgent"
            print(f"\n  [击球 {hit_count}] Player {player} ({agent_name})")
            
            action = players[agent_idx].decision(*obs)
            env.take_shot(action)
            
            hit_count += 1
            done, info = env.get_done()
            if done:
                print(f"\n  游戏结束: winner = {info['winner']}")
                
                # 转换 winner 到实际 agent
                if info['winner'] == 'SAME':
                    actual_winner = 'SAME'
                elif info['winner'] == 'A':
                    actual_winner = 'NewAgent' if game_id % 2 == 0 else 'BasicAgent'
                else:
                    actual_winner = 'BasicAgent' if game_id % 2 == 0 else 'NewAgent'
                
                print(f"  实际胜者: {actual_winner}")
                
                if actual_winner == 'NewAgent':
                    results['NEW_WIN'] += 1
                elif actual_winner == 'BasicAgent':
                    results['BASIC_WIN'] += 1
                else:
                    results['SAME'] += 1
                break
    
    print("\n" + "=" * 60)
    print("  4 局测试结果")
    print("=" * 60)
    print(f"  NewAgent 胜: {results['NEW_WIN']}")
    print(f"  BasicAgent 胜: {results['BASIC_WIN']}")
    print(f"  平局: {results['SAME']}")
    
    total = results['NEW_WIN'] + results['BASIC_WIN'] + results['SAME']
    if total > 0:
        new_rate = results['NEW_WIN'] / total
        print(f"  NewAgent 胜率: {new_rate:.1%}")

if __name__ == '__main__':
    main()

