"""
parallel_evaluate.py - 并行 Agent 评估脚本

功能：
- 使用多进程并行进行 Agent 对战
- 统计胜率和得分
- 支持灵活配置对战双方

使用方式：
    python parallel_evaluate.py --num_games 1000 --num_workers 80 --agent_a merge_basic --agent_b basic
    python parallel_evaluate.py --num_games 1000 --num_workers 80 --agent_a merge_basic --agent_b pro
"""

import os
import sys
import time
import argparse
import random
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import numpy as np

# 项目路径会在各函数内按需添加


class SuppressOutput:
    """抑制输出的上下文管理器"""
    def __init__(self):
        self._stdout = None
        self._stderr = None
        self._devnull = None
    
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._devnull = open(os.devnull, 'w')
        sys.stdout = self._devnull
        sys.stderr = self._devnull
        return self
    
    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._devnull.close()


def set_cpu_affinity(core_id, num_cores):
    """设置 CPU 亲和性"""
    try:
        os.sched_setaffinity(0, {core_id % num_cores})
    except Exception:
        pass


def load_module_from_path(module_name, file_path):
    """从指定路径加载模块，避免缓存冲突"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    
    # 临时添加模块目录到 sys.path（用于模块内部的相对导入）
    module_dir = os.path.dirname(file_path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    
    spec.loader.exec_module(module)
    return module


def create_agent(agent_type, mcts_simulations=None, mcts_candidates=None):
    """根据类型创建 Agent
    
    Args:
        agent_type: agent 类型
        mcts_simulations: MCTS 模拟次数 (用于 future/merge_basic/long)
        mcts_candidates: MCTS 候选数量 (用于 future/merge_basic/long)
    """
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if agent_type == 'merge_basic':
        # 正确的 NewAgent 在 time_limit_mcts/agent.py 中
        agent_path = os.path.join(project_root, 'time_limit_mcts', 'agent.py')
        agent_module = load_module_from_path('agent_time_limit', agent_path)
        return agent_module.NewAgent()
    elif agent_type == 'future':
        # Future 分支的 NewAgent (最新最强版本)
        agent_path = os.path.join(project_root, 'future_agent', 'agent.py')
        agent_module = load_module_from_path('agent_future', agent_path)
        return agent_module.NewAgent()
    elif agent_type == 'long':
        # Long MCTS: 独立的 long_mcts 模型 (默认 64 候选, 400 模拟)
        agent_path = os.path.join(project_root, 'long_mcts', 'agent.py')
        agent_module = load_module_from_path('agent_long', agent_path)
        return agent_module.NewAgent()
    elif agent_type == 'physics':
        # Physics-based Agent (物理模拟 agent)
        agent_path = os.path.join(project_root, 'physics_agent', 'agent.py')
        agent_module = load_module_from_path('agent_physics', agent_path)
        return agent_module.NewAgent()
    elif agent_type == 'basic':
        # BasicAgent 在 time_limit_mcts/agent.py 中
        agent_path = os.path.join(project_root, 'time_limit_mcts', 'agent.py')
        agent_module = load_module_from_path('agent_basic', agent_path)
        return agent_module.BasicAgent()
    elif agent_type == 'pro':
        # BasicAgentPro 在 test_parallel/basic_agent_pro.py 中
        sys.path.insert(0, os.path.join(project_root, 'test_parallel'))
        from basic_agent_pro import BasicAgentPro
        return BasicAgentPro(n_simulations=50)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def play_single_game(args):
    """执行单局对战"""
    game_id, agent_a_type, agent_b_type, num_cores, seed, mcts_simulations, mcts_candidates = args
    
    # 设置 CPU 亲和性
    set_cpu_affinity(game_id, num_cores)
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        with SuppressOutput():
            # 获取项目根目录并添加到路径
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, os.path.join(project_root, 'time_limit_mcts'))
            
            # 导入必要模块
            from poolenv import PoolEnv
            
            # 创建环境
            env = PoolEnv()
            
            # 创建 Agent
            agent_a = create_agent(agent_a_type, mcts_simulations, mcts_candidates)
            agent_b = create_agent(agent_b_type, mcts_simulations, mcts_candidates)
            
            players = [agent_a, agent_b]
            target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']
            
            # 根据 game_id 决定先后手和球型
            env.reset(target_ball=target_ball_choice[game_id % 4])
            
            while True:
                player = env.get_curr_player()
                obs = env.get_observation(player)
                
                if player == 'A':
                    action = players[game_id % 2].decision(*obs)
                else:
                    action = players[(game_id + 1) % 2].decision(*obs)
                
                env.take_shot(action)
                
                done, info = env.get_done()
                if done:
                    # 转换 winner
                    if info['winner'] == 'SAME':
                        winner = 'SAME'
                    elif info['winner'] == 'A':
                        winner = 'AGENT_A' if game_id % 2 == 0 else 'AGENT_B'
                    else:
                        winner = 'AGENT_B' if game_id % 2 == 0 else 'AGENT_A'
                    
                    return {
                        'game_id': game_id,
                        'winner': winner,
                        'hit_count': info.get('hit_count', 0),
                        'error': None
                    }
        
    except Exception as e:
        import traceback
        return {
            'game_id': game_id,
            'winner': None,
            'hit_count': 0,
            'error': f"{str(e)}\n{traceback.format_exc()}"
        }


def main():
    parser = argparse.ArgumentParser(description='并行 Agent 评估')
    parser.add_argument('--num_games', type=int, default=1000, help='对战局数')
    parser.add_argument('--num_workers', type=int, default=80, help='并行进程数')
    parser.add_argument('--cpu_cores', type=int, default=100, help='可用 CPU 核心数')
    parser.add_argument('--agent_a', type=str, default='merge_basic', 
                        choices=['merge_basic', 'future', 'long', 'basic', 'pro', 'physics'], help='Agent A 类型')
    parser.add_argument('--agent_b', type=str, default='basic',
                        choices=['merge_basic', 'future', 'long', 'basic', 'pro', 'physics'], help='Agent B 类型')
    parser.add_argument('--timeout', type=int, default=600, help='单局超时时间（秒）')
    parser.add_argument('--mcts_simulations', type=int, default=None, help='MCTS 模拟次数 (long agent)')
    parser.add_argument('--mcts_candidates', type=int, default=None, help='MCTS 候选数量 (long agent)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  并行 Agent 评估")
    print("=" * 60)
    print(f"  局数: {args.num_games}")
    print(f"  并行进程: {args.num_workers}")
    print(f"  CPU 核心: {args.cpu_cores}")
    print(f"  Agent A: {args.agent_a}")
    print(f"  Agent B: {args.agent_b}")
    print(f"  超时: {args.timeout}s")
    if args.mcts_simulations:
        print(f"  MCTS 模拟次数: {args.mcts_simulations}")
    if args.mcts_candidates:
        print(f"  MCTS 候选数: {args.mcts_candidates}")
    print("=" * 60)
    
    # 准备任务
    tasks = []
    for i in range(args.num_games):
        seed = random.randint(0, 2**31 - 1)
        tasks.append((i, args.agent_a, args.agent_b, args.cpu_cores, seed, 
                      args.mcts_simulations, args.mcts_candidates))
    
    # 统计结果
    results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0, 'ERROR': 0}
    total_hits = 0
    completed = 0
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(play_single_game, task): task for task in tasks}
        
        for future in as_completed(futures):
            try:
                result = future.result(timeout=args.timeout)
                
                if result['error']:
                    results['ERROR'] += 1
                    # 打印前 3 个错误的详细信息
                    if results['ERROR'] <= 3:
                        print(f"\n[ERROR {results['ERROR']}] Game {result['game_id']}: {result['error']}\n")
                elif result['winner'] == 'AGENT_A':
                    results['AGENT_A_WIN'] += 1
                elif result['winner'] == 'AGENT_B':
                    results['AGENT_B_WIN'] += 1
                else:
                    results['SAME'] += 1
                
                total_hits += result.get('hit_count', 0)
                completed += 1
                
                # 每 10 局输出进度
                if completed % 10 == 0 or completed == args.num_games:
                    elapsed = time.time() - start_time
                    speed = completed / elapsed if elapsed > 0 else 0
                    eta = (args.num_games - completed) / speed if speed > 0 else 0
                    
                    a_wins = results['AGENT_A_WIN']
                    b_wins = results['AGENT_B_WIN']
                    valid = a_wins + b_wins + results['SAME']
                    a_rate = a_wins / valid if valid > 0 else 0
                    b_rate = b_wins / valid if valid > 0 else 0
                    
                    print(f"\r[{completed}/{args.num_games}] "
                          f"A({args.agent_a}): {a_wins}({a_rate:.1%}) | "
                          f"B({args.agent_b}): {b_wins}({b_rate:.1%}) | "
                          f"Draw: {results['SAME']} | "
                          f"Err: {results['ERROR']} | "
                          f"Speed: {speed:.1f} g/s | "
                          f"ETA: {eta/60:.1f}min", end='', flush=True)
                    
            except Exception as e:
                results['ERROR'] += 1
                completed += 1
    
    print("\n")
    
    # 最终统计
    elapsed = time.time() - start_time
    valid_games = results['AGENT_A_WIN'] + results['AGENT_B_WIN'] + results['SAME']
    
    print("=" * 60)
    print("  最终结果")
    print("=" * 60)
    print(f"  总局数: {completed}")
    print(f"  有效局数: {valid_games}")
    print(f"  错误/超时: {results['ERROR']}")
    print(f"  总耗时: {elapsed/60:.1f} 分钟")
    print(f"  平均速度: {completed/elapsed:.2f} 局/秒")
    print()
    
    if valid_games > 0:
        a_rate = results['AGENT_A_WIN'] / valid_games
        b_rate = results['AGENT_B_WIN'] / valid_games
        draw_rate = results['SAME'] / valid_games
        
        print(f"  Agent A ({args.agent_a}):")
        print(f"    胜场: {results['AGENT_A_WIN']}")
        print(f"    胜率: {a_rate:.2%}")
        print()
        print(f"  Agent B ({args.agent_b}):")
        print(f"    胜场: {results['AGENT_B_WIN']}")
        print(f"    胜率: {b_rate:.2%}")
        print()
        print(f"  平局: {results['SAME']} ({draw_rate:.2%})")
        print()
        
        # 计算得分（胜1分，平0.5分）
        a_score = results['AGENT_A_WIN'] + results['SAME'] * 0.5
        b_score = results['AGENT_B_WIN'] + results['SAME'] * 0.5
        print(f"  Agent A 得分: {a_score:.1f}")
        print(f"  Agent B 得分: {b_score:.1f}")
        
        if valid_games > 0:
            avg_hits = total_hits / valid_games
            print(f"  平均击球数: {avg_hits:.1f}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()

