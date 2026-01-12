#!/usr/bin/env python3
"""
collect_data.py - 增量数据收集脚本

从 MCTS Agent 收集状态-动作对用于模仿学习。
支持在已有数据基础上继续收集。
"""

import os
import sys
import argparse
import numpy as np
import signal
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MCTS_PATH = os.path.join(PROJECT_ROOT, 'long_mcts')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MCTS_PATH)

# 全局中断标志
_interrupted = False


def signal_handler(signum, frame):
    """处理 Ctrl+C"""
    global _interrupted
    _interrupted = True
    print("\n[INFO] 收到中断信号，正在保存数据...")


def set_cpu_affinity(core_id, num_cores):
    """设置 CPU 亲和性"""
    try:
        os.sched_setaffinity(0, {core_id % num_cores})
    except:
        pass


def extract_state_features(balls, my_targets, table):
    """
    提取状态特征 (80维)
    
    结构：
    - 白球位置 (3): x, y, pocketed
    - 15个球位置 (45): x, y, pocketed × 15
    - 目标球 mask (15): 是否为己方目标
    - 袋口位置 (12): 6个袋口 × 2
    - 统计特征 (5): remaining_own, remaining_enemy, targeting_eight, eight_pocketed, target_count
    """
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


def extract_action(action):
    """
    提取动作特征 (6维)
    
    结构: [V0_norm, phi_sin, phi_cos, theta_norm, a_norm, b_norm]
    """
    V0 = action['V0']
    phi = action['phi']
    theta = action['theta']
    a = action['a']
    b = action['b']
    
    # 归一化
    V0_norm = (V0 - 0.5) / 7.5  # [0.5, 8.0] -> [0, 1]
    phi_rad = np.deg2rad(phi)
    phi_sin = np.sin(phi_rad)
    phi_cos = np.cos(phi_rad)
    theta_norm = theta / 90.0  # [0, 90] -> [0, 1]
    a_norm = a / 0.5  # [-0.5, 0.5] -> [-1, 1]
    b_norm = b / 0.5
    
    return np.array([V0_norm, phi_sin, phi_cos, theta_norm, a_norm, b_norm], dtype=np.float32)


class SuppressOutput:
    """抑制标准输出"""
    def __enter__(self):
        self._devnull = open(os.devnull, 'w')
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self._devnull
        sys.stderr = self._devnull
        return self
    
    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._devnull.close()


def collect_single_game(args):
    """
    收集单局游戏数据
    
    返回: (states, actions, game_info)
    """
    game_id, mcts_simulations, mcts_candidates, cpu_core, num_cores = args
    
    # 设置 CPU 亲和性
    set_cpu_affinity(cpu_core, num_cores)
    
    # 延迟导入（抑制导入时的输出）
    with SuppressOutput():
        from poolenv import PoolEnv
        from agent import BasicAgent, NewAgent
    
    try:
        # 抑制所有游戏输出
        with SuppressOutput():
            # 创建环境和 agent
            env = PoolEnv()
            mcts_agent = NewAgent(
                num_candidates=mcts_candidates,
                num_simulations=mcts_simulations,
                num_workers=1,
            )
            basic_agent = BasicAgent()
            
            # 随机选择先手和球型
            mcts_first = game_id % 2 == 0
            ball_type = ['solid', 'stripe'][game_id % 2]
            
            env.reset(target_ball=ball_type)
            
            states = []
            actions = []
            
            step = 0
            max_steps = 100
            
            while step < max_steps:
                player = env.get_curr_player()
                obs = env.get_observation(player)
                balls, my_targets, table = obs
                
                is_mcts_turn = (player == 'A' and mcts_first) or (player == 'B' and not mcts_first)
                
                if is_mcts_turn:
                    # MCTS agent 的回合，收集数据
                    action = mcts_agent.decision(*obs)
                    
                    # 提取特征
                    state_feat = extract_state_features(balls, my_targets, table)
                    action_feat = extract_action(action)
                    
                    states.append(state_feat)
                    actions.append(action_feat)
                else:
                    # BasicAgent 的回合
                    action = basic_agent.decision(*obs)
                
                try:
                    env.take_shot(action)
                except Exception as e:
                    break
                
                done, info = env.get_done()
                if done:
                    winner = info.get('winner', 'SAME')
                    mcts_win = (winner == 'A' and mcts_first) or (winner == 'B' and not mcts_first)
                    return (
                        np.array(states, dtype=np.float32) if states else np.zeros((0, 80), dtype=np.float32),
                        np.array(actions, dtype=np.float32) if actions else np.zeros((0, 6), dtype=np.float32),
                        {'game_id': game_id, 'mcts_win': mcts_win, 'samples': len(states)}
                    )
                
                step += 1
            
            # 超过最大步数
            return (
                np.array(states, dtype=np.float32) if states else np.zeros((0, 80), dtype=np.float32),
                np.array(actions, dtype=np.float32) if actions else np.zeros((0, 6), dtype=np.float32),
                {'game_id': game_id, 'mcts_win': False, 'samples': len(states), 'timeout': True}
            )
        
    except Exception as e:
        return (
            np.zeros((0, 80), dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32),
            {'game_id': game_id, 'error': str(e)}
        )


def load_existing_data(data_dir):
    """加载已有数据"""
    states = None
    actions = None
    
    # 查找最新的数据文件
    for f in sorted(os.listdir(data_dir), reverse=True):
        if f.startswith('states') and f.endswith('.npy') and 'checkpoint' not in f:
            states_path = os.path.join(data_dir, f)
            actions_path = os.path.join(data_dir, f.replace('states', 'actions'))
            if os.path.exists(actions_path):
                states = np.load(states_path)
                actions = np.load(actions_path)
                print(f"[INFO] 加载已有数据: {states.shape[0]} 样本")
                break
    
    return states, actions


def save_data(states, actions, output_dir, is_checkpoint=False):
    """保存数据"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    prefix = 'checkpoint_' if is_checkpoint else ''
    states_path = os.path.join(output_dir, f'{prefix}states_{timestamp}.npy')
    actions_path = os.path.join(output_dir, f'{prefix}actions_{timestamp}.npy')
    
    np.save(states_path, states)
    np.save(actions_path, actions)
    
    print(f"[INFO] 保存数据: {states.shape[0]} 样本 -> {states_path}")


def main():
    parser = argparse.ArgumentParser(description='增量数据收集')
    parser.add_argument('--num_games', type=int, default=10000,
                        help='要收集的游戏数量')
    parser.add_argument('--num_workers', type=int, default=64,
                        help='并行进程数')
    parser.add_argument('--cpu_cores', type=int, default=64,
                        help='使用的 CPU 核心数')
    parser.add_argument('--mcts_simulations', type=int, default=200,
                        help='MCTS 模拟次数')
    parser.add_argument('--mcts_candidates', type=int, default=32,
                        help='MCTS 候选动作数')
    parser.add_argument('--output_dir', type=str, default='./data_200k',
                        help='输出目录')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='检查点保存间隔')
    parser.add_argument('--task_timeout', type=int, default=600,
                        help='单任务超时时间')
    parser.add_argument('--append', action='store_true',
                        help='追加到已有数据')
    
    args = parser.parse_args()
    
    # 设置中断处理
    signal.signal(signal.SIGINT, signal_handler)
    
    # 设置主进程 CPU 亲和性
    try:
        os.sched_setaffinity(0, set(range(args.cpu_cores)))
        print(f"[INFO] CPU affinity set to cores 0-{args.cpu_cores-1}")
    except:
        pass
    
    # 加载已有数据
    all_states = []
    all_actions = []
    
    if args.append and os.path.exists(args.output_dir):
        existing_states, existing_actions = load_existing_data(args.output_dir)
        if existing_states is not None:
            all_states.append(existing_states)
            all_actions.append(existing_actions)
    
    print(f"\n{'='*60}")
    print(f"  增量数据收集")
    print(f"{'='*60}")
    print(f"  目标游戏数: {args.num_games}")
    print(f"  并行进程: {args.num_workers}")
    print(f"  CPU 核心: {args.cpu_cores}")
    print(f"  MCTS 模拟: {args.mcts_simulations}")
    print(f"  MCTS 候选: {args.mcts_candidates}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  已有样本: {sum(s.shape[0] for s in all_states)}")
    print(f"{'='*60}\n")
    
    # 准备任务
    tasks = [
        (i, args.mcts_simulations, args.mcts_candidates, i % args.cpu_cores, args.cpu_cores)
        for i in range(args.num_games)
    ]
    
    completed = 0
    mcts_wins = 0
    total_samples = sum(s.shape[0] for s in all_states)
    start_time = time.time()
    
    # 并行收集
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(collect_single_game, task): task[0] for task in tasks}
        
        for future in as_completed(futures):
            if _interrupted:
                break
            
            try:
                states, actions, info = future.result(timeout=args.task_timeout)
                
                if states.shape[0] > 0:
                    all_states.append(states)
                    all_actions.append(actions)
                    total_samples += states.shape[0]
                
                if info.get('mcts_win', False):
                    mcts_wins += 1
                
                completed += 1
                
                # 进度显示（每 100 局输出一次）
                if completed % 100 == 0 or completed == args.num_games:
                    elapsed = time.time() - start_time
                    speed = completed / elapsed if elapsed > 0 else 0
                    eta = (args.num_games - completed) / speed if speed > 0 else 0
                    win_rate = mcts_wins / completed if completed > 0 else 0
                    
                    print(f"[{completed:6d}/{args.num_games}] "
                          f"Samples: {total_samples:8d} | "
                          f"MCTS WR: {win_rate:.1%} | "
                          f"Speed: {speed:.2f} g/s | "
                          f"ETA: {eta/60:.1f}min")
                
                # 检查点
                if completed % args.checkpoint_interval == 0:
                    merged_states = np.concatenate(all_states, axis=0)
                    merged_actions = np.concatenate(all_actions, axis=0)
                    save_data(merged_states, merged_actions, args.output_dir, is_checkpoint=True)
                    
            except Exception as e:
                completed += 1
    
    # 最终保存
    if all_states:
        merged_states = np.concatenate(all_states, axis=0)
        merged_actions = np.concatenate(all_actions, axis=0)
        save_data(merged_states, merged_actions, args.output_dir, is_checkpoint=False)
    
    print(f"\n\n[INFO] 收集完成!")
    print(f"[INFO] 总游戏: {completed}")
    print(f"[INFO] 总样本: {total_samples}")
    print(f"[INFO] MCTS 胜率: {mcts_wins/completed:.1%}" if completed > 0 else "")


if __name__ == '__main__':
    main()
