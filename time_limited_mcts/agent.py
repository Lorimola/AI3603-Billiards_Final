"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from mcts import MCTSSolver
import time

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 150  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 150  # 清台前误打黑8，判负
            
    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


def analyze_shot_for_reward_mcts(
    shot: pt.System,
    last_state: dict,
    player_targets: list,
    candidate_type: str = None,
):

    new_pocketed = [
        bid for bid, b in shot.balls.items()
        if b.state.s == 4 and last_state[bid].state.s != 4
    ]
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [
        bid for bid in new_pocketed
        if bid not in player_targets and bid not in ["cue", "8"]
    ]
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    first_contact_ball_id = None
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, "ids") else []
        if ("cushion" not in et) and ("pocket" not in et) and ("cue" in ids):
            other_ids = [i for i in ids if i != "cue"]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break

    foul_first_hit = False
    if first_contact_ball_id is None:
        if len(last_state) > 2:
            foul_first_hit = True
    else:
        remaining_own_before = [
            bid for bid in player_targets if last_state[bid].state.s != 4
        ]
        opponent_plus_eight = [
            bid for bid in last_state.keys()
            if bid not in player_targets and bid not in ["cue"]
        ]
        if "8" not in opponent_plus_eight:
            opponent_plus_eight.append("8")
        if len(remaining_own_before) > 0 and first_contact_ball_id in opponent_plus_eight:
            foul_first_hit = True

    cue_hit_cushion = False
    target_hit_cushion = False
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, "ids") else []
        if "cushion" in et:
            if "cue" in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    foul_no_rail = False
    if (
        len(new_pocketed) == 0
        and first_contact_ball_id is not None
        and (not cue_hit_cushion)
        and (not target_hit_cushion)
    ):
        foul_no_rail = True

    contact_bonus = 0
    if first_contact_ball_id is not None:
        contact_bonus += 5
        if first_contact_ball_id in player_targets:
            contact_bonus += 8

    score = float(contact_bonus)

    if cue_pocketed and eight_pocketed:
        score -= 1000.0
    elif cue_pocketed:
        score -= 100.0
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 100.0 if is_targeting_eight_ball_legally else -1000.0

    if foul_no_rail:
        score -= 30.0

    if foul_first_hit:
        score -= 150.0

    if len(own_pocketed) > 0 and not foul_first_hit:
        score += 70.0

    score += len(own_pocketed) * 50.0
    score -= len(enemy_pocketed) * 20.0

    if (
        score == 0.0
        and not cue_pocketed
        and not eight_pocketed
        and not foul_first_hit
        and not foul_no_rail
    ):
        score = 10.0

    return float(score)


class NewAgent(Agent):
    """Bandit-style Monte Carlo agent (单杆多次蒙特卡洛评估 + 几何候选 + 绕球候选)."""

    def __init__(
        self,
        num_candidates: int = 30,
        num_simulations: int = 150,
        time_limit_s: float = 3.0,
        max_depth: int = 2,           # 保留参数，占位
        exploration_c: float = 1.4,     
        rollout_per_leaf: int = 2,    # 保留参数，占位
        risk_aversion: float = 0.4,   # 均值-λ*方差 中的 λ
        num_workers=6,                # CPU 并行线程数，None 表示自动
    ):
        super().__init__()
        self.time_limit_s = float(time_limit_s)
        self._decision_safety_margin_s = 0.05
        self.total_time_s = 85.0
        self.remaining_time_s = self.total_time_s
        self.shot_count = 0
        self.per_shot_time_s = 6.0
        self.fast_time_s = 2.0
        self._rack_positions = None
        self._rack_broken = False
        self.num_candidates = num_candidates
        self.pbounds = {
            "V0": (0.5, 8.0),
            "phi": (0.0, 360.0),
            "theta": (0.0, 90.0),
            "a": (-0.5, 0.5),
            "b": (-0.5, 0.5),
        }
        self.noise_std = {
            "V0": 0.1,
            "phi": 0.1,
            "theta": 0.1,
            "a": 0.003,
            "b": 0.003,
        }
        self.mcts = MCTSSolver(
            pbounds=self.pbounds,
            reward_fn=analyze_shot_for_reward_mcts,
            num_simulations=num_simulations,
            max_depth=max_depth,
            exploration_c=exploration_c,
            rollout_per_leaf=rollout_per_leaf,
            enable_noise=True,
            noise_std=self.noise_std,
            risk_aversion=risk_aversion,
            num_workers=num_workers,

        )
    def reset_budget(self):
        self.remaining_time_s = self.total_time_s
        self.shot_count = 0
        self._rack_positions = None
        self._rack_broken = False

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or my_targets is None or table is None:
            return self._random_action()
        start_ts = time.perf_counter()
        consume_time = True
        try:
            # If all object balls are cleared, switch target to 8.
            remaining_own = [
                bid for bid in my_targets
                if bid in balls and getattr(balls[bid].state, "s", 0) != 4 and bid != "8"
            ]
            if len(remaining_own) == 0 and "8" not in my_targets:
                my_targets = ["8"]
                print("[NewAgent] targets cleared, switch to 8")
            remaining_report = [
                bid for bid in my_targets
                if bid in balls and getattr(balls[bid].state, "s", 0) != 4
            ]
            print(f"[NewAgent] remaining targets: {remaining_report}")

            if self._is_unbroken_rack(balls, table):
                action = self._opening_safe_action(balls, my_targets)
                consume_time = False
                self._log_action(action, 0, start_ts)
                return action

            remaining_total = max(0.0, self.remaining_time_s)
            if remaining_total <= 0.0:
                action = self._safe_fallback_action(balls, my_targets, table)
                self._log_action(action, 0, start_ts)
                return action

            if self.shot_count < 10:
                time_budget = min(self.per_shot_time_s, remaining_total)
            elif remaining_total >= 16.0:
                time_budget = min(self.per_shot_time_s, remaining_total)
            elif remaining_total > 10.0:
                time_budget = max(3.0, remaining_total - 10.0)
            else:
                time_budget = min(self.fast_time_s, remaining_total)

            fast_mode = remaining_total <= 10.0 or (
                remaining_total > 10.0 and (remaining_total - 10.0) < 3.0
            )
            if fast_mode:
                time_budget = min(self.fast_time_s, remaining_total)
            num_candidates = self.num_candidates
            if fast_mode:
                num_candidates = max(1, int(self.num_candidates * 0.5))

            candidates = self._generate_candidate_actions(
                balls, my_targets, table, num_candidates, detour_penalty=fast_mode
            )
            if not candidates:
                action = self._random_action()
                self._log_action(action, 0, start_ts)
                return action

            fallback_action = candidates[0]
            deadline = time.perf_counter() + time_budget
            remaining = deadline - time.perf_counter() - self._decision_safety_margin_s
            if remaining <= 0.05:
                action = self._safe_fallback_action(balls, my_targets, table)
                self._log_action(action, 0, start_ts)
                return action

            action = self.mcts.search(
                balls=balls,
                my_targets=my_targets,
                table=table,
                candidate_actions=candidates,
                time_limit_s=remaining,
            )
            if action is None:
                self._log_action(fallback_action, 0, start_ts)
                return fallback_action
            sims_done = getattr(self.mcts, "last_simulations_done", None)
            self._log_action(action, sims_done, start_ts)
            return action
        except Exception as e:
            print(f"[NewAgent] error, fallback to random. Reason: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()
        finally:
            elapsed = time.perf_counter() - start_ts
            if consume_time:
                self.remaining_time_s = max(0.0, self.remaining_time_s - elapsed)
            self.shot_count += 1

    def _log_action(self, action: dict, sims_done, start_ts: float):
        remaining_preview = max(
            0.0, self.remaining_time_s - (time.perf_counter() - start_ts)
        )
        ctype = action.get("candidate_type", "random")
        print(
            f"[NewAgent] action({ctype}, sims={sims_done}, remain={remaining_preview:.2f}s): "
            f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
            f"theta={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}"

        )
    def _is_unbroken_rack(self, balls, table, tol: float = 5e-3) -> bool:
        if self._rack_broken:
            return False
        if balls is None or table is None:
            return False
        if self._rack_positions is None:
            if not self._looks_like_rack(balls, table):
                self._rack_broken = True
                return False
            self._rack_positions = {
                str(bid): np.array(ball.state.rvw[0], dtype=float)[:2]
                for bid, ball in balls.items()
                if bid != "cue"
            }
        for bid, ball in balls.items():
            if bid == "cue":
                continue
            if getattr(ball.state, "s", 0) == 4:
                self._rack_broken = True
                return False
            ref_pos = self._rack_positions.get(str(bid))
            if ref_pos is None:
                continue
            pos = np.array(ball.state.rvw[0], dtype=float)[:2]
            if np.linalg.norm(pos - ref_pos) > tol:
                self._rack_broken = True
                return False
        return True

    def _looks_like_rack(self, balls, table) -> bool:
        obj_positions = []
        for bid, ball in balls.items():
            if bid == "cue":
                continue
            if getattr(ball.state, "s", 0) == 4:
                return False
            obj_positions.append(np.array(ball.state.rvw[0], dtype=float)[:2])
        if len(obj_positions) < 15:
            return False
        coords = np.stack(obj_positions, axis=0)
        min_xy = coords.min(axis=0)
        max_xy = coords.max(axis=0)
        span = max_xy - min_xy
        ball_radius = 0.028575
        try:
            sample_ball = next(iter(balls.values()))
            ball_radius = float(getattr(sample_ball.params, "R", ball_radius))
        except Exception:
            pass
        span_limit = max(12.0 * ball_radius, 0.25 * min(table.w, table.l))
        return bool(span[0] <= span_limit and span[1] <= span_limit)

    def _opening_safe_action(self, balls, my_targets):
        cue_ball = balls.get("cue", None)
        if cue_ball is None:
            return self._random_action()
        live_targets = [
            bid for bid in my_targets
            if bid in balls and getattr(balls[bid].state, "s", 0) != 4
        ]
        if not live_targets:
            return self._random_action()
        target_id = random.choice(live_targets)
        cue_xy = np.array(cue_ball.state.rvw[0], dtype=float)[:2]
        target_xy = np.array(balls[target_id].state.rvw[0], dtype=float)[:2]
        vec = target_xy - cue_xy
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            phi = float(random.uniform(0.0, 360.0))
        else:
            phi = float(math.degrees(math.atan2(vec[1], vec[0])) % 360.0)
        v0 = float(np.clip(max(0.8, dist * 1.2), 0.5, 2.0))
        return {
            "V0": v0,
            "phi": phi,
            "theta": 2.0,
            "a": 0.0,
            "b": 0.0,
            "candidate_type": "opening_safe",
        }

    def _safe_fallback_action(self, balls, my_targets, table):
        # final_mode: geom-only + cheap safety score (cue_risk + co_pocket_risk)
        if balls is None or my_targets is None or table is None:
            return self._random_action()
        num_candidates = max(1, int(self.num_candidates * 0.5))
        candidates = self._generate_candidate_actions(
            balls,
            my_targets,
            table,
            num_candidates,
            geom_only=True,
            detour_penalty=False,
        )
        if not candidates:
            return self._random_action()
        geom_candidates = [
            act for act in candidates
            if str(act.get("candidate_type", "")).startswith("geom_")
        ]
        if not geom_candidates:
            return self._random_action()
        best_action = None
        best_score = -1e9
        for action in geom_candidates:
            cue_risk = float(action.get("prior_cue_risk", 0.0))
            co_pocket_risk = float(action.get("prior_co_pocket_risk", 0.0))
            combined_risk = 1.0 - (1.0 - cue_risk) * (1.0 - co_pocket_risk)
            score = 1.0 - combined_risk
            if score > best_score:
                best_score = score
                best_action = action
        return best_action or self._random_action()

    def _generate_candidate_actions(
        self,
        balls,
        my_targets,
        table,
        num_candidates: int,
        geom_only: bool = False,
        detour_penalty: bool = True,
    ):
        """
        生成候选动作：
        - geom: 几何直线进球
        - detour: 绕球/曲线球（使用单弯点优化）
        """
        candidates = []  # (cost, action)
        detour_candidates = []  # (cost, action)
        cue_ball = balls.get("cue", None)
        if cue_ball is None:
            return []

        cue_pos = np.array(cue_ball.state.rvw[0], dtype=float)
        cue_xy = cue_pos[:2]
        ball_radius = 0.028575
        clearance = ball_radius * 2.1   # 安全间隔（中心线与球心的最小距离）
        block_clearance = clearance * 1.6
        near_target_r = ball_radius * 3.2
        near_aim_r = ball_radius * 3.0
        near_line_band = max(block_clearance, ball_radius * 3.6)
        near_t_min = 0.6
        angle_offsets = [-4.0, -2.0, 0.0, 2.0, 4.0]
        priority_offsets = [-0.7, 0.0, 0.7]
        own_set = set(my_targets)

        # ------- 基础几何工具 -------

        def segment_distance(a_xy, b_xy, p_xy):
            """
            返回 (距离, t, side)
            t ∈ [0,1] 为 p 在 AB 上的投影参数；
            side 为 p 在 AB 左/右侧的符号（用于确定绕行方向）。
            """
            ab = b_xy - a_xy
            ab_len2 = float(np.dot(ab, ab))
            if ab_len2 < 1e-9:
                return float(np.linalg.norm(p_xy - a_xy)), 0.0, 0.0
            t = float(np.clip(np.dot(p_xy - a_xy, ab) / ab_len2, 0.0, 1.0))
            proj = a_xy + t * ab
            dist = float(np.linalg.norm(p_xy - proj))
            cross_z = ab[0] * (p_xy[1] - a_xy[1]) - ab[1] * (p_xy[0] - a_xy[0])
            side = 0.0
            if abs(cross_z) > 1e-9:
                side = math.copysign(1.0, cross_z)
            return dist, t, side

        def find_blockers(target_xy):
            """
            找出 C->target_xy 线段上的阻挡球（非己方目标球），按沿线距离排序。
            """
            blockers = []
            line_vec = target_xy - cue_xy
            for oid, ob in balls.items():
                if oid == "cue" or getattr(ob.state, "s", 0) == 4:
                    continue
                if oid in own_set:
                    continue
                pos_xy = np.array(ob.state.rvw[0], dtype=float)[:2]
                dist, t, side = segment_distance(cue_xy, target_xy, pos_xy)
                if 0.0 < t < 1.0 and dist < clearance:
                    along = float(np.linalg.norm(line_vec) * t)
                    blockers.append({
                        "id": oid,
                        "along": along,
                        "dist_to_line": dist,
                        "side": side
                    })
            return sorted(blockers, key=lambda x: x["along"])

        def find_near_target_blockers(bid, ball_xy, aim_xy, dist_ca):
            """
            找找目标球附近的阻挡球，用于触发 detour
            """
            blockers = []
            if dist_ca < 1e-6:
                return blockers
            for oid, ob in balls.items():
                if oid in ("cue", bid):
                    continue
                if getattr(ob.state, "s", 0) == 4:
                    continue
                pos_xy = np.array(ob.state.rvw[0], dtype=float)[:2]
                d_target = float(np.linalg.norm(pos_xy - ball_xy))
                d_aim = float(np.linalg.norm(pos_xy - aim_xy))
                if d_target > near_target_r and d_aim > near_aim_r:
                    continue
                dist, t, side = segment_distance(cue_xy, aim_xy, pos_xy)
                if t < near_t_min:
                    continue
                if dist > near_line_band:
                    continue
                if abs(side) < 1e-6:
                    continue
                blockers.append({
                    "id": oid,
                    "along": float(dist_ca * t),
                    "dist_to_line": dist,
                    "dist_to_target": d_target,
                    "dist_to_aim": d_aim,
                    "side": side,
                })
            return blockers

        def has_blocker_on_segment(p_start: np.ndarray, p_end: np.ndarray, allow_ids=None, threshold: float = None) -> bool:
            """
            segment_distance 判定线段上是否有阻挡。
            allow_ids: 允许被碰到的球（如目标球），其他存活球视为阻挡。
            """
            allow_ids = allow_ids or set()
            thresh = clearance if threshold is None else float(threshold)
            for oid, ob in balls.items():
                if oid == "cue" or getattr(ob.state, "s", 0) == 4:
                    continue
                if oid in allow_ids:
                    continue
                pos_xy = np.array(ob.state.rvw[0], dtype=float)[:2]
                dist, t, _ = segment_distance(p_start, p_end, pos_xy)
                if 0.0 < t < 1.0 and dist < thresh:
                    return True
            return False


        def compute_prefer_side(line_blockers, near_blockers):
            score = 0.0
            for blk in line_blockers:
                side = float(blk.get("side", 0.0))
                if abs(side) < 1e-6:
                    continue
                dist = float(blk.get("dist_to_line", clearance))
                dist = max(ball_radius, dist)
                score -= side * (ball_radius / dist)
            for blk in near_blockers:
                side = float(blk.get("side", 0.0))
                if abs(side) < 1e-6:
                    continue
                dist = float(blk.get("dist_to_target", clearance))
                dist = max(ball_radius, dist)
                score -= side * 1.4 * (ball_radius / dist)
            if abs(score) < 0.15:
                return 0.0
            return math.copysign(1.0, score)
        def classify_geom_tag(a_val: float, b_val: float, plain_th: float = 0.04) -> str:
            if abs(a_val) <= plain_th and abs(b_val) <= plain_th:
                return "plain"
            if abs(b_val) >= abs(a_val):
                return "follow_soft" if b_val >= 0.0 else "draw_soft"
            return "side_right" if a_val >= 0.0 else "side_left"

        # --- fusion extras: templated spins ---
        spin_profiles = [
            {"name": "follow_soft", "a": 0.0, "b": 0.15, "theta": (0.0, 7.0), "v": (0.95, 1.1)},
            {"name": "draw_soft", "a": 0.0, "b": -0.15, "theta": (0.0, 8.0), "v": (0.95, 1.1)},
            {"name": "side_left", "a": -0.15, "b": 0.0, "theta": (0.0, 7.0), "v": (0.95, 1.1)},
            {"name": "side_right", "a": 0.15, "b": 0.0, "theta": (0.0, 7.0), "v": (0.95, 1.1)},
        ]
        follow_draw_profiles = [spin_profiles[0], spin_profiles[1]]
        side_profiles = [spin_profiles[2], spin_profiles[3]]

        def add_priority_geom_actions(base_phi_deg, base_v0, base_cost, ball_xy, pocket_xy, aim_xy):
            # High-priority geom: low spin, low theta, small angle jitter.
            for off in priority_offsets:
                phi = (base_phi_deg + off) % 360.0
                action = {
                    "V0": float(np.clip(np.random.uniform(base_v0 * 0.95, base_v0 * 1.05), 0.5, 8.0)),
                    "phi": float(phi),
                    "theta": float(np.random.uniform(0.0, 3.0)),
                    "a": float(np.random.uniform(-0.02, 0.02)),
                    "b": float(np.random.uniform(-0.02, 0.02)),
                    "candidate_type": "geom_plain",
                }
                action["prior_penalty"] = prior_penalty_for_action(
                    action, ball_xy, pocket_xy, aim_xy, is_detour=False
                )
                candidates.append((float(base_cost - 0.15), action))

        def add_templated_geom_actions(base_phi_deg, base_v0, base_cost, ball_xy, pocket_xy, aim_xy):
            chosen = [
                random.choice(follow_draw_profiles),
                random.choice(side_profiles),
                random.choice(follow_draw_profiles + side_profiles),
            ]
            for prof in chosen:
                vmin, vmax = prof["v"]
                tmin, tmax = prof["theta"]
                a_center = prof["a"]
                b_center = prof["b"]
                phi = (base_phi_deg + random.choice(angle_offsets)) % 360.0
                action = {
                    "V0": float(np.clip(np.random.uniform(base_v0 * vmin, base_v0 * vmax), 0.5, 8.0)),
                    "phi": float(phi),
                    "theta": float(np.random.uniform(tmin, tmax)),
                    "a": float(np.random.uniform(a_center - 0.04, a_center + 0.04)),
                    "b": float(np.random.uniform(b_center - 0.04, b_center + 0.04)),
                    "candidate_type": f"geom_{prof['name']}",
                }
                extra_cost = 0.0
                if prof["name"] in ("follow_soft", "draw_soft"):
                    extra_cost = 0.08
                elif prof["name"] in ("side_left", "side_right"):
                    extra_cost = 0.12
                action["prior_penalty"] = prior_penalty_for_action(
                    action, ball_xy, pocket_xy, aim_xy, is_detour=False
                )
                candidates.append((float(base_cost + extra_cost), action))

        def angle_diff_deg(a_deg: float, b_deg: float) -> float:
            diff = (a_deg - b_deg + 180.0) % 360.0 - 180.0
            return abs(diff)

        def unit_vec(vec):
            norm = float(np.linalg.norm(vec))
            if norm < 1e-9:
                return np.array([1.0, 0.0], dtype=float), 0.0
            return vec / norm, norm

        def rot90(vec):
            return np.array([-vec[1], vec[0]], dtype=float)

        def sector_risk(
            origin_xy,
            phi_deg: float,
            length: float,
            v0: float,
            v0_std: float,
            dphi_deg: float,
            black_xy,
        ) -> float:
            if length <= 1e-6:
                return 0.0
            v0 = max(float(v0), 1e-3)
            vmin = max(0.1, v0 - 3.0 * v0_std)
            vmax = v0 + 3.0 * v0_std
            ratio_min = max(0.4, vmin / v0)
            ratio_max = min(1.6, vmax / v0)
            l_min = length * ratio_min
            l_max = length * ratio_max
            if l_min > l_max:
                l_min, l_max = l_max, l_min

            vec = black_xy - origin_xy
            r = float(np.linalg.norm(vec))
            if r < 1e-9:
                return 1.0
            ang = float(math.degrees(math.atan2(vec[1], vec[0])) % 360.0)
            dtheta = angle_diff_deg(ang, phi_deg)
            dphi = max(float(dphi_deg), 0.5)

            if dtheta <= dphi:
                if r < l_min:
                    dist = l_min - r
                elif r > l_max:
                    dist = r - l_max
                else:
                    dist = 0.0
            else:
                angle_excess = math.radians(dtheta - dphi)
                r_clamp = min(max(r, l_min), l_max)
                lateral = r_clamp * math.sin(angle_excess)
                if r < l_min:
                    radial = l_min - r
                elif r > l_max:
                    radial = r - l_max
                else:
                    radial = 0.0
                dist = math.hypot(lateral, radial)

            scale = max(ball_radius * 2.0, length * math.tan(math.radians(dphi)))
            scale = max(scale, 1e-4)
            risk = math.exp(-0.5 * (dist / scale) ** 2)
            return float(max(0.0, min(1.0, risk)))

        def _ray_hit_cushion(origin_xy, dir_xy, axis: str, c_val: float, max_len: float):
            if axis == "x":
                if abs(dir_xy[0]) < 1e-9:
                    return None
                t = (c_val - origin_xy[0]) / dir_xy[0]
                if t <= 0.0 or t > max_len:
                    return None
                y = origin_xy[1] + dir_xy[1] * t
                if 0.0 <= y <= table.l:
                    return t
                return None
            if abs(dir_xy[1]) < 1e-9:
                return None
            t = (c_val - origin_xy[1]) / dir_xy[1]
            if t <= 0.0 or t > max_len:
                return None
            x = origin_xy[0] + dir_xy[0] * t
            if 0.0 <= x <= table.w:
                return t
            return None

        def _reflect_point(point_xy, axis: str, c_val: float):
            if axis == "x":
                return np.array([2.0 * c_val - point_xy[0], point_xy[1]], dtype=float)
            return np.array([point_xy[0], 2.0 * c_val - point_xy[1]], dtype=float)

        def cushion_reflect_risk(
            origin_xy,
            phi_deg: float,
            length: float,
            v0: float,
            v0_std: float,
            dphi_deg: float,
            black_xy,
        ) -> float:
            if length <= 1e-6:
                return 0.0
            v0 = max(float(v0), 1e-3)
            vmin = max(0.1, v0 - 3.0 * v0_std)
            vmax = v0 + 3.0 * v0_std
            ratio_max = min(1.6, vmax / v0)
            l_max = length * ratio_max

            ang = math.radians(phi_deg)
            dir_xy = np.array([math.cos(ang), math.sin(ang)], dtype=float)
            risk = 0.0

            for axis, c_val in (("x", 0.0), ("x", float(table.w)), ("y", 0.0), ("y", float(table.l))):
                t_hit = _ray_hit_cushion(origin_xy, dir_xy, axis, c_val, l_max)
                if t_hit is None:
                    continue
                mirrored_black = _reflect_point(black_xy, axis, c_val)
                risk = max(
                    risk,
                    sector_risk(
                        origin_xy,
                        phi_deg,
                        length,
                        v0,
                        v0_std,
                        dphi_deg + 2.0,
                        mirrored_black,
                    ),
                )
            return float(max(0.0, min(1.0, risk)))

        def elastic_after_state(
            cue_xy: np.ndarray,
            phi_deg: float,
            black_xy: np.ndarray,
            ball_radius: float,
            dphi_deg: float,
        ):
            phi_rad = math.radians(phi_deg)
            u = np.array([math.cos(phi_rad), math.sin(phi_rad)], dtype=float)
            vec_cb = black_xy - cue_xy
            dist_cb = float(np.linalg.norm(vec_cb))
            if dist_cb < 1e-6:
                return None
            t_proj = float(np.dot(vec_cb, u))
            if t_proj <= 0.0:
                return None
            offset = vec_cb - u * t_proj
            d_perp = float(np.linalg.norm(offset))
            extra = dist_cb * math.sin(math.radians(max(dphi_deg, 1.0)))
            hit_radius = 2.0 * ball_radius + extra
            if d_perp > hit_radius:
                return None
            d_clamp = min(max(d_perp, 0.0), 2.0 * ball_radius - 1e-6)
            t_contact = t_proj - math.sqrt(max(0.0, (2.0 * ball_radius) ** 2 - d_clamp ** 2))
            if t_contact <= 0.0:
                t_contact = t_proj
            contact_xy = cue_xy + u * t_contact
            n_vec = contact_xy - black_xy
            n, n_len = unit_vec(n_vec)
            if n_len < 1e-6:
                return None
            dot_un = float(np.dot(u, n))
            v_tan = u - dot_un * n
            v_tan_norm = float(np.linalg.norm(v_tan))
            if v_tan_norm < 1e-6:
                return contact_xy, None, 0.0
            v_after_dir = v_tan / v_tan_norm
            return contact_xy, v_after_dir, v_tan_norm

        def prior_penalty_for_action(action, ball_xy, pocket_xy, aim_xy, is_detour: bool) -> float:
            black_ball = balls.get("8", None)
            if black_ball is None or getattr(black_ball.state, "s", 0) == 4:
                return 0.0
            only_black = len(my_targets) == 1 and my_targets[0] == "8"
            black_xy = np.array(black_ball.state.rvw[0], dtype=float)[:2]

            phi_std = float(self.noise_std.get("phi", 0.1))
            v0_std = float(self.noise_std.get("V0", 0.1))
            dphi_deg = 3.0 * phi_std
            spin_mag = abs(float(action.get("a", 0.0))) + abs(float(action.get("b", 0.0)))
            theta_val = abs(float(action.get("theta", 0.0)))
            dphi_deg += 12.0 * spin_mag + 0.3 * theta_val
            if is_detour and detour_penalty:
                dphi_deg *= 1.4
            dphi_deg = max(dphi_deg, 1.0)

            ctype = str(action.get("candidate_type", "")).lower()
            type_weights = {
                "cue": 1.0,
                "obj": 1.0,
                "after": 1.0,
                "cushion": 1.0,
                "co_pocket": 1.0,
                "near_ball": 1.0,
                "near_aim": 1.0,
                "near_pocket": 1.0,
            }
            dphi_scale = 1.0
            if detour_penalty and "detour" in ctype:
                type_weights["cue"] *= 1.1
                type_weights["after"] *= 1.2
                type_weights["cushion"] *= 1.35
                dphi_scale *= 1.2
            if "side" in ctype:
                type_weights["cue"] *= 1.05
                type_weights["cushion"] *= 1.2
                type_weights["near_aim"] *= 1.1
                dphi_scale *= 1.15
            if "follow" in ctype:
                type_weights["after"] *= 1.25
                type_weights["co_pocket"] *= 1.2
                dphi_scale *= 1.05
            if "draw" in ctype:
                type_weights["after"] *= 1.15
                dphi_scale *= 1.05
            if "plain" in ctype:
                type_weights["cushion"] *= 0.9
            if ctype in ("opening_safe", "safe_fallback", "random"):
                for key in type_weights:
                    type_weights[key] *= 0.85
                dphi_scale *= 0.9
            dphi_deg *= dphi_scale

            cue_dir, cue_len = unit_vec(aim_xy - cue_xy)
            obj_dir, obj_len = unit_vec(pocket_xy - ball_xy)
            vec_cb = ball_xy - cue_xy
            vec_bp = pocket_xy - ball_xy
            dist_cb = float(np.linalg.norm(vec_cb))
            dist_bp = float(np.linalg.norm(vec_bp))
            cut_angle_deg = 0.0
            if dist_cb > 1e-6 and dist_bp > 1e-6:
                cos_cut = float(np.dot(vec_cb, vec_bp) / (dist_cb * dist_bp))
                cos_cut = max(-1.0, min(1.0, cos_cut))
                cut_angle_deg = float(math.degrees(math.acos(cos_cut)))

            phi_deg = float(action.get("phi", 0.0)) % 360.0
            cue_risk = sector_risk(
                cue_xy, phi_deg, cue_len, action.get("V0", 0.0), v0_std, dphi_deg, black_xy
            )
            action["prior_cue_risk"] = float(cue_risk)

            throw_deg = min(25.0, 0.25 * cut_angle_deg + 8.0 * spin_mag)
            obj_phi = float(math.degrees(math.atan2(obj_dir[1], obj_dir[0])) % 360.0)
            obj_risk = sector_risk(
                ball_xy,
                obj_phi,
                obj_len,
                action.get("V0", 0.0),
                v0_std,
                max(dphi_deg + throw_deg, 2.0),
                black_xy,

            )
            cue_cushion_risk = cushion_reflect_risk(
                cue_xy, phi_deg, cue_len, action.get("V0", 0.0), v0_std, dphi_deg, black_xy
            )
            obj_cushion_risk = cushion_reflect_risk(
                ball_xy,
                obj_phi,
                obj_len,
                action.get("V0", 0.0),
                v0_std,
                max(dphi_deg + throw_deg, 2.0),
                black_xy,

            )
            cross_z = cue_dir[0] * obj_dir[1] - cue_dir[1] * obj_dir[0]
            side = -math.copysign(1.0, cross_z) if abs(cross_z) > 1e-6 else 1.0
            stun_dir = rot90(obj_dir) * side
            b_val = float(action.get("b", 0.0))
            follow_ratio = max(-1.0, min(1.0, b_val / 0.2))
            if cut_angle_deg < 5.0:
                base_after_dir = obj_dir if b_val >= 0.0 else -obj_dir
            else:
                if follow_ratio >= 0.0:
                    mix = (1.0 - follow_ratio) * stun_dir + follow_ratio * obj_dir
                else:
                    mix = (1.0 - abs(follow_ratio)) * stun_dir + abs(follow_ratio) * (-obj_dir)
                base_after_dir, _ = unit_vec(mix)

            after_phi = float(math.degrees(math.atan2(base_after_dir[1], base_after_dir[0])) % 360.0)
            base_after_len = max(ball_radius * 6.0, 0.35 * cue_len)
            speed_scale = max(0.4, min(1.8, float(action.get("V0", 0.0)) / 3.0))
            spin_scale = 1.0 + 0.8 * abs(b_val)
            cue_after_len = base_after_len * speed_scale * spin_scale
            max_len = 0.7 * min(table.w, table.l)
            cue_after_len = min(cue_after_len, max_len)
            dphi_after = dphi_deg + 6.0 * abs(float(action.get("a", 0.0))) + 0.4 * theta_val
            cue_after_risk = sector_risk(
                ball_xy,
                after_phi,
                cue_after_len,
                action.get("V0", 0.0),
                v0_std,
                max(dphi_after, 2.0),
                black_xy,
            )
            cue_after_cushion_risk = cushion_reflect_risk(
                ball_xy,
                after_phi,
                cue_after_len,
                action.get("V0", 0.0),
                v0_std,
                max(dphi_after, 2.0),
                black_xy,

            )
            d_black_ball = float(np.linalg.norm(black_xy - ball_xy))
            near_ball_risk = math.exp(-0.5 * (d_black_ball / max(2.5 * ball_radius, 1e-4)) ** 2)
            d_black_aim = float(np.linalg.norm(black_xy - aim_xy))
            near_aim_risk = math.exp(-0.5 * (d_black_aim / max(2.2 * ball_radius, 1e-4)) ** 2)
            d_black_pocket = float(np.linalg.norm(black_xy - pocket_xy))
            pocket_risk = math.exp(-0.5 * (d_black_pocket / max(1.8 * ball_radius, 1e-4)) ** 2)

            v0_val = float(action.get("V0", 0.0))
            b_val = float(action.get("b", 0.0))
            co_pocket_risk = 0.0
            hit_black_risk = 0.0
            elastic_state = elastic_after_state(cue_xy, phi_deg, black_xy, ball_radius, dphi_deg)
            if elastic_state is not None and dist_cb > 1e-6:
                contact_xy, elastic_dir, sin_theta = elastic_state
                hit_black_risk = sector_risk(
                    cue_xy, phi_deg, dist_cb, v0_val, v0_std, dphi_deg, black_xy
                )
                if hit_black_risk > 1e-6:
                    follow_ratio = max(-1.0, min(1.0, b_val / 0.2))
                    if elastic_dir is None:
                        if follow_ratio > 0.2:
                            after_dir = cue_dir
                        elif follow_ratio < -0.2:
                            after_dir = -cue_dir
                        else:
                            after_dir = None
                    else:
                        after_dir = elastic_dir
                        if abs(follow_ratio) > 0.05:
                            mix = (1.0 - abs(follow_ratio)) * after_dir + abs(follow_ratio) * (
                                cue_dir if follow_ratio > 0.0 else -cue_dir
                            )
                            after_dir, _ = unit_vec(mix)
                    if after_dir is not None:
                        after_phi_elastic = float(
                            math.degrees(math.atan2(after_dir[1], after_dir[0])) % 360.0
                        )
                        v_after = max(0.3, v0_val * max(0.1, sin_theta))
                        dphi_after_elastic = max(2.0, dphi_deg + 6.0 * (1.0 - sin_theta))
                        for pocket in table.pockets.values():
                            p_xy = np.array(pocket.center, dtype=float)[:2]
                            d_black_p = float(np.linalg.norm(black_xy - p_xy))
                            black_p_risk = math.exp(
                                -0.5 * (d_black_p / max(1.6 * ball_radius, 1e-4)) ** 2
                            )
                            if black_p_risk < 1e-3:
                                continue
                            after_len_p = float(np.linalg.norm(p_xy - contact_xy))
                            cue_after_p = sector_risk(
                                contact_xy,
                                after_phi_elastic,
                                after_len_p,
                                v_after,
                                v0_std,
                                dphi_after_elastic,
                                p_xy,
                            )
                            cue_after_cushion = cushion_reflect_risk(
                                contact_xy,
                                after_phi_elastic,
                                after_len_p,
                                v_after,
                                v0_std,
                                dphi_after_elastic,
                                p_xy,
                            )
                            cue_after_p = max(cue_after_p, cue_after_cushion)
                            co_pocket_risk = max(
                                co_pocket_risk, hit_black_risk * black_p_risk * cue_after_p
                            )
            follow_bias = 1.0 + 0.8 * max(0.0, b_val) + 0.35 * max(0.0, v0_val - 2.0) / 3.0
            co_pocket_risk = min(1.0, co_pocket_risk * min(2.0, follow_bias))
            action["prior_co_pocket_risk"] = float(co_pocket_risk)

            def _w(risk_val: float, key: str) -> float:
                return max(0.0, min(1.0, risk_val * type_weights[key]))

            if only_black:
                penalty = -500.0 * _w(co_pocket_risk, "co_pocket")
                return float(max(-500.0, min(0.0, penalty)))

            cushion_risk = max(cue_cushion_risk, obj_cushion_risk, cue_after_cushion_risk)
            risk = 1.0 - (
                (1.0 - _w(cue_risk, "cue"))
                * (1.0 - _w(obj_risk, "obj"))
                * (1.0 - _w(cue_after_risk, "after"))
                * (1.0 - _w(cushion_risk, "cushion"))
                * (1.0 - _w(co_pocket_risk, "co_pocket"))
                * (1.0 - _w(near_ball_risk, "near_ball"))
                * (1.0 - _w(near_aim_risk, "near_aim"))
                * (1.0 - _w(pocket_risk, "near_pocket"))
            )
            penalty = -500.0 * risk
            return float(max(-500.0, min(0.0, penalty)))


        # ------- 单弯点 detour 优化：在几何上寻找最佳弯点 X -------

        def plan_detour_actions_for_pair(
            bid: str,
            aim_xy: np.ndarray,
            base_v0: float,
            vec_ca: np.ndarray,
            blockers_for_pair,
            prefer_side_hint: float = 0.0,
            max_actions_per_pair: int = 3,
        ):
            """
            对 (cue, 目标球 bid, 该 pocket) 这对组合：
            在平面上搜索一个或多个“弯点” X，使得：
            - C -> X -> aim_xy 两段线段尽量不靠近任何障碍球；
            - 路径长度尽量短。
            然后根据弯曲程度转换为 detour 类型的击球参数。
            """
            C = cue_xy
            G = aim_xy
            v = G - C
            L = float(np.linalg.norm(v))
            if L < 1e-4:
                return []

            # 构造局部坐标系：e_para 沿 C->G，e_perp 垂直
            e_para = v / L
            e_perp = np.array([-e_para[1], e_para[0]], dtype=float)

            # t 在 [0.15L, 0.85L] 的中间带；s 为法向偏移
            t_min = 0.15 * L
            t_max = 0.85 * L

            # 允许的最大偏移，既要能绕过球，也别太离谱
            s_max = min(max(3.0 * clearance, 0.25 * L), 0.8 * L)

            # 从 blockers 中推断“优选绕行侧”
            prefer_side = float(prefer_side_hint)
            if abs(prefer_side) < 1e-6 and blockers_for_pair:
                nearest = blockers_for_pair[0]
                side = nearest.get("side", 0.0)
                # 一般往阻挡球的反侧绕
                if abs(side) > 1e-6:
                    prefer_side = -side

            # 障碍球列表：所有存活球（除了 cue 和当前目标球）
            obstacles = []
            for oid, ob in balls.items():
                if oid in ("cue", bid):
                    continue
                if getattr(ob.state, "s", 0) == 4:
                    continue
                pos_xy = np.array(ob.state.rvw[0], dtype=float)[:2]
                obstacles.append((oid, pos_xy))

            if not obstacles:
                # 场上几乎没球，不必特意 detour
                return []

            # 目标函数：路径长度 + 碰撞惩罚
            lambda_coll = 250.0  # 碰撞惩罚权重
            min_required_clearance = clearance  # 期望的中心线-球心最小距离

            def eval_objective(t, s):
                """
                t: 沿 C->G 的距离
                s: 法向偏移
                """
                X = C + t * e_para + s * e_perp
                # 路径长度
                path_len = float(np.linalg.norm(X - C) + np.linalg.norm(G - X))
                penalty = 0.0
                min_clear = float("inf")

                for _, O in obstacles:
                    # C->X
                    d1, tau1, _ = segment_distance(C, X, O)
                    # X->G
                    d2, tau2, _ = segment_distance(X, G, O)

                    # 只对线段内部的最近点计惩罚
                    d_candidates = []
                    if 0.0 < tau1 < 1.0:
                        d_candidates.append(d1)
                    if 0.0 < tau2 < 1.0:
                        d_candidates.append(d2)

                    if not d_candidates:
                        continue

                    d = min(d_candidates)
                    min_clear = min(min_clear, d)

                    if d < min_required_clearance:
                        # hard penalty：越近惩罚越大
                        penalty += 1.0 + (min_required_clearance - d) / max(min_required_clearance, 1e-4)

                obj = path_len + lambda_coll * penalty
                if min_clear == float("inf"):
                    min_clear = 999.0
                return obj, min_clear, X

            # --------- 随机多启动搜索 ---------
            num_samples = 80  # 全局采样次数
            best_list = []     # 保存若干最优 (obj, min_clear, t, s, X)

            for i in range(num_samples):
                # t 均匀采样在中段
                t = float(np.random.uniform(t_min, t_max))

                # s 带有优选侧的偏分布
                if prefer_side == 0.0:
                    side = random.choice([-1.0, 1.0])
                else:
                    # 70% 概率取优选侧，30% 取另一侧以防局部最优
                    if random.random() < 0.7:
                        side = prefer_side
                    else:
                        side = -prefer_side
                s_abs = float(np.random.uniform(0.4 * clearance, s_max))
                s = side * s_abs

                obj, min_clear, X = eval_objective(t, s)

                best_list.append((obj, min_clear, t, s, X))

            # 按 obj 排序，选出前若干作为候选弯点
            best_list.sort(key=lambda x: x[0])
            # 优先选“完全无碰撞”的
            good_points = [item for item in best_list if item[1] >= min_required_clearance]
            if not good_points:
                # 放宽一点要求，允许略小于 clearance 的情况
                good_points = [item for item in best_list if item[1] >= 0.7 * min_required_clearance]

            if not good_points:
                return []

            max_pool = max_actions_per_pair * 6
            if len(good_points) > max_pool:
                good_points = good_points[:max_pool]

            # --------- map (C -> X -> G) path to detour action ---------
            detour_actions = []
            L_direct = max(L, 1e-4)

            def detour_metrics(item):
                _, min_clear, _, _, X = item
                diff = X - C
                t_comp = float(np.dot(diff, e_para))
                s_comp = float(np.dot(diff, e_perp))
                curvature = abs(s_comp) / L_direct
                path_len = float(np.linalg.norm(X - C) + np.linalg.norm(G - X))
                length_ratio = path_len / L_direct
                vec_cx = X - C
                vec_len = float(np.linalg.norm(vec_cx))
                if vec_len < 1e-6:
                    turn_angle = 180.0
                else:
                    cos_turn = float(np.dot(vec_cx / vec_len, e_para))
                    cos_turn = max(-1.0, min(1.0, cos_turn))
                    turn_angle = float(math.degrees(math.acos(cos_turn)))
                return {
                    "min_clear": min_clear,
                    "t_comp": t_comp,
                    "s_comp": s_comp,
                    "curvature": curvature,
                    "path_len": path_len,
                    "length_ratio": length_ratio,
                    "turn_angle": turn_angle,
                }

            priority_points = []
            regular_points = []
            for idx, item in enumerate(good_points):
                metrics = detour_metrics(item)
                min_clear = metrics["min_clear"]
                if (
                    min_clear >= 1.15 * min_required_clearance
                    and metrics["curvature"] <= 0.28
                    and metrics["length_ratio"] <= 1.35
                    and metrics["turn_angle"] <= 40.0
                ):
                    key = (
                        metrics["length_ratio"],
                        metrics["curvature"],
                        -min_clear,
                        metrics["turn_angle"],
                    )
                    priority_points.append((key, idx, item, metrics))
                regular_points.append((item[0], idx, item, metrics))

            priority_points.sort(key=lambda x: x[0])
            priority_points = priority_points[:max_actions_per_pair]
            used_idx = {p[1] for p in priority_points}
            remain = max_actions_per_pair - len(priority_points)
            if remain > 0:
                regular_points = [r for r in regular_points if r[1] not in used_idx]
                regular_points.sort(key=lambda x: x[0])
                regular_points = regular_points[:remain]

            selected_points = [(True, p[2], p[3]) for p in priority_points]
            selected_points += [(False, r[2], r[3]) for r in regular_points]

            for is_priority, item, metrics in selected_points:
                _, _, _, _, X = item
                curvature = metrics["curvature"]
                path_len = metrics["path_len"]
                length_ratio = metrics["length_ratio"]
                s_comp = metrics["s_comp"]

                # power scaling by path length
                if is_priority:
                    v0_scale = float(np.clip(0.85 + 0.20 * (length_ratio - 1.0), 0.75, 1.10))
                else:
                    v0_scale = float(np.clip(0.9 + 0.25 * (length_ratio - 1.0), 0.8, 1.35))
                V0_center = base_v0 * v0_scale

                vec_cx = X - C
                detour_phi = float(math.degrees(math.atan2(vec_cx[1], vec_cx[0])) % 360.0)

                if is_priority:
                    if curvature < 0.18:
                        a_min, a_max = 0.04, 0.10
                        theta_min, theta_max = 2.5, 6.0
                    else:
                        a_min, a_max = 0.08, 0.16
                        theta_min, theta_max = 4.0, 8.0
                    b_min, b_max = -0.04, 0.04
                    cost_scale = 0.85
                    ctype = "detour_pri"
                else:
                    if curvature < 0.15:
                        a_min, a_max = 0.05, 0.12
                        theta_min, theta_max = 3.0, 7.0
                    elif curvature < 0.3:
                        a_min, a_max = 0.12, 0.20
                        theta_min, theta_max = 6.0, 11.0
                    else:
                        a_min, a_max = 0.20, 0.30
                        theta_min, theta_max = 8.0, 14.0
                    b_min, b_max = -0.06, 0.06
                    cost_scale = 1.0
                    ctype = "detour"

                side_spin_sign = math.copysign(1.0, s_comp) if abs(s_comp) > 1e-4 else random.choice([-1.0, 1.0])

                detour_action = {
                    "V0": float(np.random.uniform(0.9 * V0_center, 1.1 * V0_center)),
                    "phi": detour_phi,
                    "theta": float(np.random.uniform(theta_min, theta_max)),
                    "a": float(np.random.uniform(a_min, a_max) * side_spin_sign),
                    "b": float(np.random.uniform(b_min, b_max)),
                    "candidate_type": ctype,
                }
                detour_actions.append((path_len * cost_scale, detour_action))

            return detour_actions

        # --- cheap prefilter for (ball, pocket) pairs ---
        pair_infos = []
        for bid in my_targets:
            if bid == "cue":
                continue
            ball = balls.get(bid, None)
            if ball is None or getattr(ball.state, "s", 0) == 4:
                continue
            ball_xy = np.array(ball.state.rvw[0], dtype=float)[:2]
            for pocket in table.pockets.values():
                pocket_xy = np.array(pocket.center, dtype=float)[:2]
                vec_bp = pocket_xy - ball_xy
                dist_bp = float(np.linalg.norm(vec_bp))
                if dist_bp < 1e-6:
                    continue
                dir_bp = vec_bp / dist_bp
                aim_xy = ball_xy - dir_bp * (ball_radius * 2.0)
                vec_ca = aim_xy - cue_xy
                dist_ca = float(np.linalg.norm(vec_ca))
                if dist_ca < 1e-6:
                    continue
                vec_cb = ball_xy - cue_xy
                dist_cb = float(np.linalg.norm(vec_cb))
                if dist_cb < 1e-6:
                    continue
                blockers = find_blockers(aim_xy)
                blocked_any_line = has_blocker_on_segment(cue_xy, aim_xy, threshold=block_clearance)
                near_blockers = find_near_target_blockers(bid, ball_xy, aim_xy, dist_ca)
                prefer_side_hint = compute_prefer_side(blockers, near_blockers)
                blockers_all = sorted(blockers + near_blockers, key=lambda x: x.get("along", 0.0))
                blocked_any = blocked_any_line or bool(near_blockers)
                cos_cut = float(np.dot(vec_cb, vec_bp) / (dist_cb * dist_bp))
                cos_cut = max(-1.0, min(1.0, cos_cut))
                cut_penalty = 1.0 - max(0.0, cos_cut)
                near_penalty = 0.1 * len(near_blockers) if near_blockers else 0.0
                score = (
                    dist_bp
                    + 0.6 * dist_ca
                    + 0.25 * len(blockers)
                    + (0.6 if blocked_any_line else 0.0)
                    + near_penalty
                    + 1.5 * cut_penalty
                )
                pair_infos.append(
                    (
                        score,
                        bid,
                        ball_xy,
                        pocket_xy,
                        aim_xy,
                        vec_ca,
                        dist_bp,
                        blockers_all,
                        blocked_any,
                        prefer_side_hint,

                    )
                )
        if not pair_infos:
            return []

        pair_infos.sort(key=lambda x: x[0])
        pair_budget = min(len(pair_infos), max(6, int(num_candidates * 1.0)))
        selected_pairs = pair_infos[:pair_budget]

        # --- offensive candidates: geom + detour ---
        for _, bid, ball_xy, pocket_xy, aim_xy, vec_ca, dist_bp, blockers, blocked_any, prefer_side_hint in selected_pairs:
            base_phi_deg = math.degrees(math.atan2(vec_ca[1], vec_ca[0])) % 360.0
            base_v0 = float(np.clip(dist_bp * 4.5, 1.0, 6.5))
            risk_penalty = 0.25 * len(blockers)
            base_cost = dist_bp + risk_penalty

            add_priority_geom_actions(base_phi_deg, base_v0, base_cost, ball_xy, pocket_xy, aim_xy)

            for off in angle_offsets:
                phi = (base_phi_deg + off) % 360.0
                a_val = float(np.random.uniform(-0.12, 0.12))
                b_val = float(np.random.uniform(-0.12, 0.12))
                geom_tag = classify_geom_tag(a_val, b_val)
                if geom_tag == "plain":
                    continue
                action = {
                    "V0": float(np.random.uniform(base_v0 * 0.9, base_v0 * 1.1)),
                    "phi": float(phi),
                    "theta": float(np.random.uniform(0.0, 8.0)),
                    "a": a_val,
                    "b": b_val,
                    "candidate_type": f"geom_{geom_tag}",
                }
                action["prior_penalty"] = prior_penalty_for_action(
                    action, ball_xy, pocket_xy, aim_xy, is_detour=False
                )
                candidates.append((base_cost, action))

            add_templated_geom_actions(base_phi_deg, base_v0, base_cost, ball_xy, pocket_xy, aim_xy)

            if (not geom_only) and blocked_any:
                detour_for_pair = plan_detour_actions_for_pair(
                    bid=bid,
                    aim_xy=aim_xy,
                    base_v0=base_v0,
                    vec_ca=vec_ca,
                    blockers_for_pair=blockers,
                    prefer_side_hint=prefer_side_hint,
                    max_actions_per_pair=3,
                )
                for cost_like, act in detour_for_pair:
                    act["prior_penalty"] = prior_penalty_for_action(
                        act, ball_xy, pocket_xy, aim_xy, is_detour=True
                    )
                    detour_candidates.append((base_cost + 0.1 + cost_like, act))
        candidates.sort(key=lambda x: x[0])
        detour_candidates.sort(key=lambda x: x[0])
        geom_actions = [act for _, act in candidates]
        detour_actions = [act for _, act in detour_candidates]

        if geom_only:
            target_detour = 0
            target_geom = max(1, num_candidates)
        else:
            target_detour = max(1, int(num_candidates * 0.2))
            target_geom = max(1, num_candidates - target_detour)

        keep_geom = geom_actions[: min(len(geom_actions), target_geom)]
        keep_detour = [] if geom_only else detour_actions[: min(len(detour_actions), target_detour)]

        final_candidates = list(keep_geom + keep_detour)
        used = len(final_candidates)

        # 从剩余 geom/detour 中继续补足
        leftover_scored = candidates[len(keep_geom):]
        if not geom_only:
            leftover_scored += detour_candidates[len(keep_detour):]
        leftover_scored.sort(key=lambda x: x[0])
        for _, act in leftover_scored:
            if used >= num_candidates:
                break
            final_candidates.append(act)
            used += 1

        # 如果仍未满足数量，用随机动作补齐
        while used < num_candidates:
            rand_act = self._random_action()
            rand_act["candidate_type"] = "random"
            rand_act["prior_penalty"] = 0.0
            final_candidates.append(rand_act)
            used += 1

        # 如果候选数超出了，随机下采样到 num_candidates
        if len(final_candidates) > num_candidates:
            final_candidates = final_candidates[:num_candidates]

        return final_candidates