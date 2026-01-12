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
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...]
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8）, -30（首球/碰库犯规）
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue']
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2:  # 只有白球和8号球时不算犯规
             foul_first_hit = True
    else:
        remaining_own_before = [bid for bid in player_targets if last_state[bid].state.s != 4]
        opponent_plus_eight = [bid for bid in last_state.keys() if bid not in player_targets and bid not in ['cue']]
        if ('8' not in opponent_plus_eight):
            opponent_plus_eight.append('8')
            
        if len(remaining_own_before) > 0 and first_contact_ball_id in opponent_plus_eight:
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
        
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 100 if is_targeting_eight_ball_legally else -150
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
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
                    
                    # 关键：使用 pooltool 物理引擎 (世界A)
                    pt.simulate(shot, inplace=True)
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

class NewAgent(Agent):
    """基于物理模拟和几何计算的智能 Agent（噪声鲁棒版）
    
    核心策略：
    1. 精确几何计算：利用球半径、球袋位置和半径精确计算击球角度
    2. 障碍检测：检查白球到目标球、目标球到球袋的路径（增加安全边距）
    3. 评分系统：综合考虑距离、切球角度、球袋难度等因素
    4. 物理模拟验证：使用 pooltool 模拟微调参数（含噪声测试）
    5. 危险规避：检测黑8和对手球的危险位置
    """
    
    def __init__(self):
        super().__init__()
        # 这些参数会在 decision 时从 balls/table 动态获取
        self.ball_radius = None
        self.table_width = None
        self.table_length = None
        self.pockets_info = None  # {pocket_id: {'center': np.array, 'radius': float}}
        
        # 噪声参数（与环境一致）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        
        # 安全边距系数（用于噪声容错）
        self.safety_margin = 1.5  # 路径检测时增加的安全系数
        
        print("NewAgent (Physics-based, Noise-Robust) 已初始化。")
    
    def decision(self, balls=None, my_targets=None, table=None):
        """基于几何计算和物理模拟的决策方法"""
        if balls is None or my_targets is None or table is None:
            print("[NewAgent] 缺少必要信息，使用随机动作。")
            return self._random_action()
        
        try:
            # 1. 初始化物理参数
            self._init_physics_params(balls, table)
            
            # 2. 获取白球信息
            cue_ball = balls['cue']
            if cue_ball.state.s == 4:  # 白球进袋
                return self._random_action()
            cue_pos = cue_ball.state.rvw[0][:2]
            
            # 3. 确定目标球列表
            remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_targets) == 0:
                remaining_targets = ['8']  # 己方球清空，打黑8
            
            # 4. 识别危险球（黑8和对手球）
            dangerous_balls = self._identify_dangerous_balls(balls, my_targets, remaining_targets)
            
            # 5. 评估所有可能的击球方案
            all_shots = []
            for target_id in remaining_targets:
                if balls[target_id].state.s == 4:
                    continue
                target_pos = balls[target_id].state.rvw[0][:2]
                
                for pocket_id, pocket_info in self.pockets_info.items():
                    shot_info = self._evaluate_shot(
                        cue_pos, target_pos, target_id,
                        pocket_info, pocket_id, balls,
                        dangerous_balls
                    )
                    if shot_info is not None:
                        all_shots.append(shot_info)
            
            # 6. 按得分排序，选择最佳方案
            all_shots.sort(key=lambda x: x['score'], reverse=True)
            
            if all_shots and all_shots[0]['score'] > 10:  # 提高阈值，更保守
                best_shot = all_shots[0]
                # 使用物理模拟微调（含噪声测试）
                action = self._optimize_with_simulation(best_shot, balls, my_targets, table)
                print(f"[NewAgent] 进攻: 目标={best_shot['target_id']}→{best_shot['pocket_id']}, "
                      f"切角={best_shot['cut_angle']:.1f}°, 得分={best_shot['score']:.1f}")
                return action
            else:
                # 没有好的进球机会，防守
                action = self._defensive_shot(cue_pos, balls, my_targets, table)
                print(f"[NewAgent] 防守: phi={action['phi']:.1f}°, V0={action['V0']:.2f}")
                return action
                
        except Exception as e:
            print(f"[NewAgent] 决策异常: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()
    
    def _init_physics_params(self, balls, table):
        """从 balls 和 table 中提取物理参数"""
        # 球半径
        self.ball_radius = balls['cue'].params.R
        
        # 球桌尺寸
        self.table_width = table.w
        self.table_length = table.l
        
        # 球袋信息（包含位置和半径）
        self.pockets_info = {}
        for pocket_id, pocket in table.pockets.items():
            self.pockets_info[pocket_id] = {
                'center': np.array(pocket.center[:2]),
                'radius': pocket.radius,
                'is_corner': pocket_id in ['lb', 'lt', 'rb', 'rt'],  # 角袋
                'is_side': pocket_id in ['lc', 'rc']  # 中袋
            }
    
    def _identify_dangerous_balls(self, balls, my_targets, remaining_targets):
        """识别危险球位置
        
        返回：危险球列表 [(ball_id, position, danger_level), ...]
        """
        dangerous = []
        
        for ball_id, ball in balls.items():
            if ball.state.s == 4:  # 已进袋
                continue
            if ball_id == 'cue':
                continue
            if ball_id in remaining_targets:
                continue
            
            pos = np.array(ball.state.rvw[0][:2])
            
            # 黑8特别危险（己方球未清空时）
            if ball_id == '8' and remaining_targets != ['8']:
                dangerous.append((ball_id, pos, 3.0))  # 最高危险等级
            # 对手球
            elif ball_id not in my_targets:
                dangerous.append((ball_id, pos, 1.5))  # 中等危险
        
        return dangerous
    
    def _evaluate_shot(self, cue_pos, target_pos, target_id, pocket_info, pocket_id, balls, dangerous_balls):
        """评估一个击球方案的可行性和得分（含噪声鲁棒性评估）
        
        返回: dict 或 None（不可行）
        """
        cue_pos = np.array(cue_pos)
        target_pos = np.array(target_pos)
        pocket_center = pocket_info['center']
        pocket_radius = pocket_info['radius']
        
        # 1. 计算目标球到球袋的方向
        target_to_pocket = pocket_center - target_pos
        dist_to_pocket = np.linalg.norm(target_to_pocket)
        
        if dist_to_pocket < self.ball_radius:  # 球已经在袋口
            return None
            
        target_to_pocket_unit = target_to_pocket / dist_to_pocket
        
        # 2. 计算"幽灵球"位置（白球需要到达的位置才能把目标球打进袋）
        ghost_ball_pos = target_pos - target_to_pocket_unit * (2 * self.ball_radius)
        
        # 3. 计算白球到幽灵球的方向和距离
        cue_to_ghost = ghost_ball_pos - cue_pos
        dist_cue_to_ghost = np.linalg.norm(cue_to_ghost)
        
        if dist_cue_to_ghost < 2 * self.ball_radius:  # 太近，无法击打
            return None
            
        cue_to_ghost_unit = cue_to_ghost / dist_cue_to_ghost
        
        # 4. 计算切球角度
        dot_product = np.dot(cue_to_ghost_unit, target_to_pocket_unit)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        cut_angle = math.degrees(math.acos(dot_product))
        
        # 噪声鲁棒性：降低大角度切球的容忍度（原70度降至55度）
        max_cut_angle = 55
        if cut_angle > max_cut_angle:
            return None
        
        # 5. 检查路径障碍（增加安全边距）
        # 5a. 白球到幽灵球的路径
        if self._path_has_obstacle_safe(cue_pos, ghost_ball_pos, balls, ['cue', target_id]):
            return None
        
        # 5b. 目标球到球袋的路径
        if self._path_has_obstacle_safe(target_pos, pocket_center, balls, ['cue', target_id]):
            return None
        
        # 6. 检查危险球风险
        danger_penalty = self._calculate_danger_penalty(
            cue_pos, ghost_ball_pos, target_pos, pocket_center,
            dangerous_balls, dist_cue_to_ghost
        )
        
        # 如果危险太高，放弃这个方案
        if danger_penalty > 50:
            return None
        
        # 7. 计算击球角度 phi
        phi = math.degrees(math.atan2(cue_to_ghost[1], cue_to_ghost[0]))
        if phi < 0:
            phi += 360
        
        # 8. 计算得分（含鲁棒性评估）
        score = self._calculate_shot_score_robust(
            dist_cue_to_ghost, dist_to_pocket, cut_angle,
            pocket_info, pocket_radius, danger_penalty
        )
        
        # 9. 估算击球力度（更保守）
        V0 = self._estimate_velocity_safe(dist_cue_to_ghost, dist_to_pocket, cut_angle)
        
        return {
            'target_id': target_id,
            'pocket_id': pocket_id,
            'phi': phi,
            'V0': V0,
            'theta': 0,
            'a': 0,
            'b': 0,
            'score': score,
            'cut_angle': cut_angle,
            'ghost_pos': ghost_ball_pos,
            'dist_cue_to_target': dist_cue_to_ghost,
            'dist_target_to_pocket': dist_to_pocket
        }
    
    def _path_has_obstacle_safe(self, start_pos, end_pos, balls, exclude_ids):
        """检查路径上是否有障碍球（增加安全边距）
        
        使用更大的安全边距来容忍噪声
        """
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)
        
        path_vec = end_pos - start_pos
        path_length = np.linalg.norm(path_vec)
        
        if path_length < 0.001:
            return False
            
        path_unit = path_vec / path_length
        
        # 障碍检测的临界距离：增加安全边距
        critical_dist = 2 * self.ball_radius * self.safety_margin
        
        for ball_id, ball in balls.items():
            if ball_id in exclude_ids:
                continue
            if ball.state.s == 4:  # 已进袋
                continue
            
            ball_pos = np.array(ball.state.rvw[0][:2])
            start_to_ball = ball_pos - start_pos
            proj_length = np.dot(start_to_ball, path_unit)
            
            # 扩大检测范围
            margin = self.ball_radius * self.safety_margin
            if proj_length < -margin or proj_length > path_length + margin:
                continue
            
            # 计算垂直距离
            if proj_length < 0:
                dist = np.linalg.norm(start_to_ball)
            elif proj_length > path_length:
                dist = np.linalg.norm(ball_pos - end_pos)
            else:
                closest_point = start_pos + path_unit * proj_length
                dist = np.linalg.norm(ball_pos - closest_point)
            
            if dist < critical_dist:
                return True
        
        return False
    
    def _calculate_danger_penalty(self, cue_pos, ghost_pos, target_pos, pocket_pos, dangerous_balls, shot_dist):
        """计算危险球惩罚分
        
        检查白球路径和目标球路径附近是否有危险球
        """
        penalty = 0
        
        for ball_id, ball_pos, danger_level in dangerous_balls:
            # 检查白球路径附近的危险球
            dist_to_cue_path = self._point_to_line_distance(ball_pos, cue_pos, ghost_pos)
            if dist_to_cue_path < 3 * self.ball_radius:
                penalty += danger_level * 20 * (1 - dist_to_cue_path / (3 * self.ball_radius))
            
            # 检查目标球路径附近的危险球
            dist_to_target_path = self._point_to_line_distance(ball_pos, target_pos, pocket_pos)
            if dist_to_target_path < 3 * self.ball_radius:
                penalty += danger_level * 15 * (1 - dist_to_target_path / (3 * self.ball_radius))
            
            # 距离越近，危险越高
            dist_to_target = np.linalg.norm(ball_pos - target_pos)
            if dist_to_target < 4 * self.ball_radius:
                penalty += danger_level * 10
        
        return penalty
    
    def _point_to_line_distance(self, point, line_start, line_end):
        """计算点到线段的最短距离"""
        point = np.array(point)
        line_start = np.array(line_start)
        line_end = np.array(line_end)
        
        line_vec = line_end - line_start
        line_length = np.linalg.norm(line_vec)
        
        if line_length < 0.001:
            return np.linalg.norm(point - line_start)
        
        line_unit = line_vec / line_length
        start_to_point = point - line_start
        proj = np.dot(start_to_point, line_unit)
        
        if proj < 0:
            return np.linalg.norm(start_to_point)
        elif proj > line_length:
            return np.linalg.norm(point - line_end)
        else:
            closest = line_start + line_unit * proj
            return np.linalg.norm(point - closest)
    
    def _calculate_shot_score_robust(self, dist_cue_target, dist_target_pocket, cut_angle, 
                                      pocket_info, pocket_radius, danger_penalty):
        """计算击球得分（噪声鲁棒版）
        
        更保守的评分策略
        """
        # 距离得分：短距离更可靠
        dist1_score = 100 * math.exp(-dist_cue_target / 1.2)  # 更快衰减
        dist2_score = 100 * math.exp(-dist_target_pocket / 0.8)
        
        # 切球角度得分：更严格惩罚大角度
        # 0度=100分，30度=60分，55度=0分
        angle_score = max(0, 100 * (1 - cut_angle / 55))
        
        # 球袋类型加成
        if pocket_info['is_corner']:
            pocket_bonus = 1.05
        else:
            pocket_bonus = 0.9  # 中袋更难，降低更多
        
        # 进球容错空间
        tolerance_bonus = pocket_radius / 0.06
        
        # 综合得分（提高角度权重）
        total_score = (
            dist1_score * 0.20 +
            dist2_score * 0.25 +
            angle_score * 0.55  # 更重视角度
        ) * pocket_bonus * tolerance_bonus
        
        # 切球角度严格惩罚
        if cut_angle > 40:
            total_score *= (55 - cut_angle) / 15
        
        # 远距离惩罚（噪声影响更大）
        if dist_cue_target > 1.0:
            total_score *= 0.8
        if dist_cue_target > 1.5:
            total_score *= 0.7
        
        # 扣除危险惩罚
        total_score -= danger_penalty
        
        return max(0, total_score)
    
    def _estimate_velocity_safe(self, dist_cue_target, dist_target_pocket, cut_angle):
        """估算击球力度（更保守，减少误伤）
        
        使用较小的力度，减少球的后续滚动
        """
        total_dist = dist_cue_target + dist_target_pocket
        
        # 更保守的力度公式
        base_v0 = 1.3 + total_dist * 1.0
        
        # 切球角度补偿（但不过度补偿）
        cut_angle_rad = math.radians(cut_angle)
        efficiency = max(0.4, math.cos(cut_angle_rad))
        adjusted_v0 = base_v0 / efficiency
        
        # 更低的上限，减少意外
        return np.clip(adjusted_v0, 1.0, 5.0)
    
    def _optimize_with_simulation(self, shot_info, balls, my_targets, table):
        """使用物理模拟微调击球参数（含多次噪声测试）"""
        base_phi = shot_info['phi']
        base_v0 = shot_info['V0']
        
        best_action = {
            'V0': base_v0,
            'phi': base_phi,
            'theta': 0,
            'a': 0,
            'b': 0
        }
        best_score = -float('inf')
        
        # 更细致的搜索网格
        phi_range = [-2, -1, 0, 1, 2]
        v0_range = [-0.5, -0.25, 0, 0.25, 0.5]
        
        for dphi in phi_range:
            for dv0 in v0_range:
                test_phi = (base_phi + dphi) % 360
                test_v0 = np.clip(base_v0 + dv0, 0.5, 8.0)
                
                # 多次模拟取平均（考虑噪声）
                total_score = 0
                n_trials = 3
                for _ in range(n_trials):
                    score = self._simulate_with_noise(
                        test_v0, test_phi, 0, 0, 0,
                        balls, my_targets, table
                    )
                    total_score += score
                
                avg_score = total_score / n_trials
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_action = {
                        'V0': test_v0,
                        'phi': test_phi,
                        'theta': 0,
                        'a': 0,
                        'b': 0
                    }
        
        return best_action
    
    def _simulate_with_noise(self, V0, phi, theta, a, b, balls, my_targets, table):
        """执行物理模拟（添加噪声）"""
        try:
            # 添加噪声
            V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
            phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
            theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
            a_noisy = a + np.random.normal(0, self.noise_std['a'])
            b_noisy = b + np.random.normal(0, self.noise_std['b'])
            
            # 限制范围
            V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
            phi_noisy = phi_noisy % 360
            theta_noisy = np.clip(theta_noisy, 0, 90)
            a_noisy = np.clip(a_noisy, -0.5, 0.5)
            b_noisy = np.clip(b_noisy, -0.5, 0.5)
            
            # 创建模拟环境
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id="cue")
            
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
            pt.simulate(shot, inplace=True)
            
            last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            score = analyze_shot_for_reward(shot, last_state, my_targets)
            
            return score
        except Exception:
            return -500
    
    def _defensive_shot(self, cue_pos, balls, my_targets, table):
        """防守策略（更安全）
        
        优先避免打到对手球和黑8
        """
        cue_pos = np.array(cue_pos)
        
        # 获取剩余目标球
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining_targets) == 0:
            remaining_targets = ['8']
        
        best_action = None
        best_score = -float('inf')
        
        # 识别危险球
        dangerous_balls = self._identify_dangerous_balls(balls, my_targets, remaining_targets)
        
        # 策略：向最安全的目标球轻推
        for target_id in remaining_targets:
            target_pos = np.array(balls[target_id].state.rvw[0][:2])
            direction = target_pos - cue_pos
            dist = np.linalg.norm(direction)
            
            if dist < 0.01:
                continue
            
            # 检查路径是否有危险球
            path_danger = 0
            for ball_id, ball_pos, danger_level in dangerous_balls:
                d = self._point_to_line_distance(ball_pos, cue_pos, target_pos)
                if d < 3 * self.ball_radius:
                    path_danger += danger_level
            
            if path_danger > 2:  # 太危险，跳过
                continue
                
            base_phi = math.degrees(math.atan2(direction[1], direction[0]))
            if base_phi < 0:
                base_phi += 360
            
            # 使用小力度轻推
            for v0 in [1.2, 1.5, 1.8]:
                for dphi in [-3, 0, 3]:
                    test_phi = (base_phi + dphi) % 360
                    
                    # 多次模拟取平均
                    total_score = 0
                    n_trials = 3
                    for _ in range(n_trials):
                        score = self._simulate_with_noise(
                            v0, test_phi, 0, 0, 0,
                            balls, my_targets, table
                        )
                        total_score += score
                    
                    avg_score = total_score / n_trials
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_action = {
                            'V0': v0,
                            'phi': test_phi,
                            'theta': 0,
                            'a': 0,
                            'b': 0
                        }
        
        if best_action is not None:
            return best_action
        
        # 后备：随机动作
        return self._random_action()