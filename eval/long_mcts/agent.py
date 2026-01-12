import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import time
import signal
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from mcts import MCTSSolver


# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=20):
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
                    if not simulate_with_timeout(shot, timeout=20):
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
        num_candidates: int = 64,
        num_simulations: int = 400,
        max_depth: int = 2,           # 保留参数，占位
        exploration_c: float = 1.4,     
        rollout_per_leaf: int = 2,    # 保留参数，占位
        risk_aversion: float = 0.4,   # 均值-λ*方差 中的 λ
        num_workers=8,                # CPU 并行线程数，None 表示自动
    ):
        super().__init__()
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

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or my_targets is None or table is None:
            return self._random_action()
        try:
            # 若己方普通球已全部进袋，则只剩黑8
            remaining_own = [
                bid for bid in my_targets
                if bid in balls and getattr(balls[bid].state, "s", 0) != 4 and bid != "8"
            ]
            if len(remaining_own) == 0 and "8" not in my_targets:
                my_targets = ["8"]
                print("[NewAgent] 目标球已清空，切换为黑8")
            remaining_report = [
                bid for bid in my_targets
                if bid in balls and getattr(balls[bid].state, "s", 0) != 4
            ]
            print(f"[NewAgent] 剩余目标球: {remaining_report}")

            candidates = self._generate_candidate_actions(
                balls, my_targets, table, self.num_candidates
            )
            if not candidates:
                return self._random_action()

            action = self.mcts.search(
                balls=balls,
                my_targets=my_targets,
                table=table,
                candidate_actions=candidates,
            )
            if action is None:
                return self._random_action()
            ctype = action.get("candidate_type", "unknown")
            print(
                f"[NewAgent] action({ctype}): "
                f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                f"theta={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}"
            )
            return action
        except Exception as e:
            print(f"[NewAgent] error, fallback to random. Reason: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

    def _generate_candidate_actions(self, balls, my_targets, table, num_candidates: int):
        """
        生成候选动作：
        - geom: 几何直线进球
        - detour: 绕球/曲线球（使用单弯点优化）
        """
        candidates = []  # (cost, action)
        detour_candidates = []  # (cost, action)
        kick_candidates = []  # (cost, action)
        cue_ball = balls.get("cue", None)
        if cue_ball is None:
            return []

        cue_pos = np.array(cue_ball.state.rvw[0], dtype=float)
        cue_xy = cue_pos[:2]
        ball_radius = 0.028575
        clearance = ball_radius * 2.1   # 安全间隔（中心线与球心的最小距离）
        angle_offsets = [-4.0, -2.0, 0.0, 2.0, 4.0]
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

        def has_blocker_on_segment(p_start: np.ndarray, p_end: np.ndarray, allow_ids=None) -> bool:
            """
            segment_distance 判定线段上是否有阻挡。
            allow_ids: 允许被碰到的球（如目标球），其他存活球视为阻挡。
            """
            allow_ids = allow_ids or set()
            for oid, ob in balls.items():
                if oid == "cue" or getattr(ob.state, "s", 0) == 4:
                    continue
                if oid in allow_ids:
                    continue
                pos_xy = np.array(ob.state.rvw[0], dtype=float)[:2]
                dist, t, _ = segment_distance(p_start, p_end, pos_xy)
                if 0.0 < t < 1.0 and dist < clearance:
                    return True
            return False

        def classify_geom_tag(a_val: float, b_val: float, plain_th: float = 0.04) -> str:
            if abs(a_val) <= plain_th and abs(b_val) <= plain_th:
                return "plain"
            if abs(b_val) >= abs(a_val):
                return "follow_soft" if b_val >= 0.0 else "draw_soft"
            return "side_right" if a_val >= 0.0 else "side_left"

        # --- fusion extras: templated spins + simple one-cushion kick candidates ---
        spin_profiles = [
            {"name": "plain", "a": 0.0, "b": 0.0, "theta": (0.0, 6.0), "v": (0.9, 1.1)},
            {"name": "follow_soft", "a": 0.0, "b": 0.15, "theta": (0.0, 7.0), "v": (0.95, 1.1)},
            {"name": "draw_soft", "a": 0.0, "b": -0.15, "theta": (0.0, 8.0), "v": (0.95, 1.1)},
            {"name": "side_left", "a": -0.15, "b": 0.0, "theta": (0.0, 7.0), "v": (0.95, 1.1)},
            {"name": "side_right", "a": 0.15, "b": 0.0, "theta": (0.0, 7.0), "v": (0.95, 1.1)},
        ]
        plain_prof = spin_profiles[0]
        follow_draw_profiles = [spin_profiles[1], spin_profiles[2]]
        side_profiles = [spin_profiles[3], spin_profiles[4]]

        def add_templated_geom_actions(base_phi_deg, base_v0, base_cost):
            chosen = [
                plain_prof,
                random.choice(follow_draw_profiles),
                random.choice(side_profiles),
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
                candidates.append((float(base_cost + extra_cost), action))

        pocket_centers = [np.array(p.center, dtype=float)[:2] for p in table.pockets.values()]
        bank_points = []
        if pocket_centers:
            pcs = np.stack(pocket_centers, axis=0)
            min_x, min_y = pcs.min(axis=0)
            max_x, max_y = pcs.max(axis=0)
            mid_x = 0.5 * (min_x + max_x)
            mid_y = 0.5 * (min_y + max_y)
            bank_points = [
                np.array([mid_x, min_y], dtype=float),
                np.array([mid_x, max_y], dtype=float),
                np.array([min_x, mid_y], dtype=float),
                np.array([max_x, mid_y], dtype=float),
            ]

        def generate_kick_action(bp_xy, aim_xy, dist_bp, risk_penalty):
            if has_blocker_on_segment(cue_xy, bp_xy) or has_blocker_on_segment(bp_xy, aim_xy):
                return None
            phi = float(math.degrees(math.atan2(bp_xy[1] - cue_xy[1], bp_xy[0] - cue_xy[0])) % 360.0)
            total_dist = float(np.linalg.norm(bp_xy - cue_xy) + np.linalg.norm(aim_xy - bp_xy))
            v0 = float(np.clip(total_dist * 3.5, 0.8, 6.5))
            action = {
                "V0": v0,
                "phi": phi,
                "theta": float(np.random.uniform(0.0, 7.0)),
                "a": float(np.random.uniform(-0.15, 0.15)),
                "b": float(np.random.uniform(-0.12, 0.12)),
                "candidate_type": "kick",
            }
            cost = float(dist_bp + risk_penalty + 0.6 * total_dist + 0.8)
            return (cost, action)

        # ------- 单弯点 detour 优化：在几何上寻找最佳弯点 X -------

        def plan_detour_actions_for_pair(
            bid: str,
            aim_xy: np.ndarray,
            base_v0: float,
            vec_ca: np.ndarray,
            blockers_for_pair,
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
            prefer_side = 0.0
            if blockers_for_pair:
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
            num_samples = 160  # 全局采样次数
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

            good_points = good_points[:max_actions_per_pair]

            # --------- 将 (C -> X -> G) 几何路径映射为 detour 击球参数 ---------
            detour_actions = []
            L_direct = max(L, 1e-4)

            for obj, min_clear, t_val, s_val, X in good_points:
                diff = X - C
                # 分解回 (t,s)，可用于估计弯曲程度
                t_comp = float(np.dot(diff, e_para))
                s_comp = float(np.dot(diff, e_perp))
                curvature = abs(s_comp) / L_direct  # 粗略曲率指标

                # 力度：根据路径拉长比例调整
                path_len = float(np.linalg.norm(X - C) + np.linalg.norm(G - X))
                length_ratio = path_len / L_direct
                v0_scale = float(np.clip(0.9 + 0.25 * (length_ratio - 1.0), 0.8, 1.35))
                V0_center = base_v0 * v0_scale

                # 角度：朝向 C->X
                vec_cx = X - C
                detour_phi = float(math.degrees(math.atan2(vec_cx[1], vec_cx[0])) % 360.0)

                # 根据曲率大小选择合适的抬杆角和侧旋范围
                if curvature < 0.15:
                    a_min, a_max = 0.05, 0.12
                    theta_min, theta_max = 3.0, 7.0
                elif curvature < 0.3:
                    a_min, a_max = 0.12, 0.20
                    theta_min, theta_max = 6.0, 11.0
                else:
                    a_min, a_max = 0.20, 0.30
                    theta_min, theta_max = 8.0, 14.0

                # 侧旋方向与 s 的符号一致
                side_spin_sign = math.copysign(1.0, s_comp) if abs(s_comp) > 1e-4 else random.choice([-1.0, 1.0])

                detour_action = {
                    "V0": float(np.random.uniform(0.9 * V0_center, 1.1 * V0_center)),
                    "phi": detour_phi,
                    "theta": float(np.random.uniform(theta_min, theta_max)),
                    "a": float(np.random.uniform(a_min, a_max) * side_spin_sign),
                    "b": float(np.random.uniform(-0.06, 0.06)),
                    "candidate_type": "detour",
                }
                detour_actions.append((path_len, detour_action))

            return detour_actions

        # --- 进攻型候选：geom + detour ---
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
                dist_bp = np.linalg.norm(vec_bp)
                if dist_bp < 1e-6:
                    continue
                dir_bp = vec_bp / dist_bp

                # 鬼球点（ghost ball），即 aim_xy
                aim_xy = ball_xy - dir_bp * (ball_radius * 2.0)

                vec_ca = aim_xy - cue_xy
                if np.linalg.norm(vec_ca) < 1e-6:
                    continue

                base_phi_deg = math.degrees(math.atan2(vec_ca[1], vec_ca[0])) % 360.0
                base_v0 = float(np.clip(dist_bp * 4.5, 1.0, 6.5))

                # 先看看 C->aim 是否有阻挡
                blockers = find_blockers(aim_xy)
                blocked_any = has_blocker_on_segment(cue_xy, aim_xy)
                risk_penalty = 0.25 * len(blockers)
                base_cost = dist_bp + risk_penalty

                # 直接几何进攻（无论是否有阻挡，都可以尝试）
                for off in angle_offsets:
                    phi = (base_phi_deg + off) % 360.0
                    a_val = float(np.random.uniform(-0.12, 0.12))
                    b_val = float(np.random.uniform(-0.12, 0.12))
                    geom_tag = classify_geom_tag(a_val, b_val)
                    action = {
                        "V0": float(np.random.uniform(base_v0 * 0.9, base_v0 * 1.1)),
                        "phi": float(phi),
                        "theta": float(np.random.uniform(0.0, 8.0)),
                        "a": a_val,
                        "b": b_val,
                        "candidate_type": f"geom_{geom_tag}",
                    }
                    candidates.append((base_cost, action))

                # add a few structured spin templates (plain/follow/draw/side)
                add_templated_geom_actions(base_phi_deg, base_v0, base_cost)

                # add a few kick/bank candidates when the direct line is blocked
                if bank_points and blocked_any:
                    for bp in bank_points:
                        kick = generate_kick_action(bp, aim_xy, dist_bp, risk_penalty)
                        if kick is not None:
                            kick_candidates.append(kick)

                # 若存在阻挡，用“单弯点优化算法”生成 detour 候选
                if blocked_any:
                    detour_for_pair = plan_detour_actions_for_pair(
                        bid=bid,
                        aim_xy=aim_xy,
                        base_v0=base_v0,
                        vec_ca=vec_ca,
                        blockers_for_pair=blockers,
                        max_actions_per_pair=3,
                    )
                    for cost_like, act in detour_for_pair:
                        detour_candidates.append((base_cost + 0.1 + cost_like, act))

        candidates.sort(key=lambda x: x[0])
        detour_candidates.sort(key=lambda x: x[0])
        kick_candidates.sort(key=lambda x: x[0])
        geom_actions = [act for _, act in candidates]
        detour_actions = [act for _, act in detour_candidates]
        kick_actions = [act for _, act in kick_candidates]

        target_detour = max(1, int(num_candidates * 0.15))
        target_kick = max(0, int(num_candidates * 0.15))
        target_geom = max(1, num_candidates - target_detour - target_kick)

        keep_geom = geom_actions[: min(len(geom_actions), target_geom)]
        keep_detour = detour_actions[: min(len(detour_actions), target_detour)]
        keep_kick = kick_actions[: min(len(kick_actions), target_kick)]

        final_candidates = list(keep_geom + keep_detour + keep_kick)
        used = len(final_candidates)

        # 从剩余 geom/detour 中继续补足
        leftover_scored = (
            candidates[len(keep_geom):]
            + detour_candidates[len(keep_detour):]
            + kick_candidates[len(keep_kick):]
        )
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
            final_candidates.append(rand_act)
            used += 1

        # 如果候选数超出了，随机下采样到 num_candidates
        if len(final_candidates) > num_candidates:
            final_candidates = final_candidates[:num_candidates]

        return final_candidates
