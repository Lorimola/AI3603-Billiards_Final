# MCTS / Bandit core for billiards

import copy
import math
import random
import warnings
from typing import Dict, List, Optional, Callable

import numpy as np
import pooltool as pt
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


class MCTSNode:
    """
    保留的树节点结构（目前 bandit-style 搜索并未真正展开多步树，仅作为兼容占位）。
    """

    def __init__(self, balls, table, my_targets, parent=None, action=None, depth=0):
        self.balls = balls
        self.table = table
        self.my_targets = list(my_targets)
        self.parent = parent
        self.action = action
        self.depth = depth
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value = 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def ucb_score(self, c=1.4):
        if self.visits == 0 or self.parent is None:
            return float("inf")
        return self.value / self.visits + c * math.sqrt(
            math.log(self.parent.visits + 1) / self.visits
        )


class CandidateStats:
    """
    记录单个候选动作的蒙特卡洛统计量：采样次数、均值和方差（Welford 在线算法）。
    """

    def __init__(self, action: dict):
        self.action = action
        self.n: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0  # 用于方差计算

    def update(self, reward: float):
        self.n += 1
        delta = reward - self.mean
        self.mean += delta / self.n
        delta2 = reward - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n <= 1:
            return 0.0
        return max(self.M2 / (self.n - 1), 0.0)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


class MCTSSolver:
    """
    Bandit-style 单杆蒙特卡洛搜索：

    - 输入一批 candidate_actions（由 NewAgent 产生：geom / detour / defend / random）
    - 对每个候选动作进行多次带噪声仿真
    - 使用 UCB 在候选之间分配仿真预算
    - 所有仿真阶段都通过 ThreadPoolExecutor 批量并行执行
    - 最终按 mean - λ * std（λ = risk_aversion）选出相对稳定的动作
    """

    def __init__(
        self,
        pbounds: Dict[str, tuple],
        reward_fn: Callable,
        num_simulations: int = 120,   # 总仿真预算（所有动作总和）
        max_depth: int = 2,           # 保留参数
        exploration_c: float = 1.4,   # UCB 探索系数
        rollout_per_leaf: int = 1,    # 保留参数
        enable_noise: bool = True,
        noise_std: Optional[Dict[str, float]] = None,
        first_hit_bonus: float = 6.0,
        first_hit_penalty: float = 5.0,
        risk_v0_softcap: float = 6.0,
        risk_spin_softcap: float = 0.12,
        risk_v0_weight: float = 0.4,
        risk_spin_weight: float = 8.0,
        risk_aversion: float = 0.4,   # 最终打分: mean - λ * std
        num_workers=None,             # CPU并行线程数
    ):
        self.pbounds = pbounds
        self.reward_fn = reward_fn
        self.num_simulations = max(0, int(num_simulations))
        self.max_depth = max_depth
        self.exploration_c = exploration_c
        self.rollout_per_leaf = rollout_per_leaf
        self.enable_noise = enable_noise
        self.noise_std = noise_std or {
            "V0": 0.1,
            "phi": 0.1,
            "theta": 0.1,
            "a": 0.003,
            "b": 0.003,
        }
        self.first_hit_bonus = first_hit_bonus
        self.first_hit_penalty = first_hit_penalty
        # 轻惩高风险动作：大力度、大侧旋在噪声下更容易失误
        self.risk_v0_softcap = risk_v0_softcap
        self.risk_spin_softcap = risk_spin_softcap
        self.risk_v0_weight = risk_v0_weight
        self.risk_spin_weight = risk_spin_weight

        self.risk_aversion = max(0.0, float(risk_aversion))

        # 并行 worker 数：默认最多 8 个（可以在 NewAgent 里显式传 num_workers）
        if num_workers is None or num_workers <= 0:
            cpu_cnt = os.cpu_count() or 1
            self.num_workers = max(1, min(cpu_cnt, 8))
        else:
            self.num_workers = int(num_workers)

    # ------------------------------------------------------------------
    # 对外接口：bandit-style 单杆蒙特卡洛搜索（完全并行化仿真）
    # ------------------------------------------------------------------
    def search(self, balls, my_targets, table, candidate_actions: List[dict]):
        """
        对给定的一批候选动作进行单杆蒙特卡洛评估：

        - 阶段1：初始均匀探索，每个动作至少评估若干次（若预算允许），用线程池并行执行
          初始每个动作的评估次数大致为: init_per_action ≈ num_simulations / (2 * num_candidates)
        - 阶段2：剩余预算通过 UCB 分配，按 batch 并行评估
        - 最终按 mean - risk_aversion * std 选择一个动作
        """
        num_actions = len(candidate_actions)
        if num_actions == 0:
            return None
        if self.num_simulations <= 0:
            return random.choice(candidate_actions)

        stats_list = [CandidateStats(act) for act in candidate_actions]
        total_budget = self.num_simulations

        # -------------------------------
        # 阶段1：初始均匀探索（并行）
        # -------------------------------
        budget_used = 0

        if total_budget < num_actions:
            # 预算太少，只能让部分动作各打一杆
            init_indices = list(range(num_actions))
            random.shuffle(init_indices)
            init_indices = init_indices[:total_budget]
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(
                        self._evaluate_candidate_once,
                        balls,
                        table,
                        my_targets,
                        stats_list[idx].action,
                        idx,
                    )
                    for idx in init_indices
                ]
                for fut in as_completed(futures):
                    try:
                        idx, reward = fut.result()
                    except Exception:
                        idx, reward = 0, -500.0
                    stats_list[idx].update(reward)
                    budget_used += 1
        else:
            # 预算足够覆盖一轮所有动作
            approx_init = max(1, total_budget // (2 * num_actions))
            max_possible_init = total_budget // num_actions
            init_per_action = max(1, min(approx_init, max_possible_init))

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for _ in range(init_per_action):
                    futures = []
                    for idx in range(num_actions):
                        futures.append(
                            executor.submit(
                                self._evaluate_candidate_once,
                                balls,
                                table,
                                my_targets,
                                stats_list[idx].action,
                                idx,
                            )
                        )
                    for fut in as_completed(futures):
                        try:
                            idx, reward = fut.result()
                        except Exception:
                            idx, reward = 0, -500.0
                        stats_list[idx].update(reward)
                        budget_used += 1

        # -------------------------------
        # 阶段2：UCB 分配剩余预算（批量并行）
        # -------------------------------
        budget_remaining = total_budget - budget_used
        if budget_remaining > 0:
            batch_base = max(1, self.num_workers * 2)
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                while budget_remaining > 0:
                    batch_size = min(batch_base, budget_remaining)
                    # 预先根据当前统计，用 UCB 选出本批要评估的动作索引
                    indices = []
                    # 虚拟计数避免同一批次反复选到首个未采样动作
                    virtual_counts = [0] * len(stats_list)
                    for _ in range(batch_size):
                        total_n = sum(s.n for s in stats_list) + sum(virtual_counts)
                        idx = self._select_with_ucb(
                            stats_list, total_n, virtual_counts
                        )
                        indices.append(idx)
                        virtual_counts[idx] += 1

                    futures = [
                        executor.submit(
                            self._evaluate_candidate_once,
                            balls,
                            table,
                            my_targets,
                            stats_list[idx].action,
                            idx,
                        )
                        for idx in indices
                    ]

                    for fut in as_completed(futures):
                        try:
                            idx, reward = fut.result()
                        except Exception:
                            idx, reward = 0, -500.0
                        stats_list[idx].update(reward)
                        budget_remaining -= 1
                        if budget_remaining <= 0:
                            break

        # -------------------------------
        # 阶段3：根据 mean - λ * std 选择最终动作
        # -------------------------------
        best_idx = None
        best_score = -float("inf")
        for i, s in enumerate(stats_list):
            if s.n == 0:
                continue
            # 对低采样的动作设置std下限，避免单次幸运样本占优
            effective_std = s.std if s.n >= 2 else max(50.0, s.std)
            score = s.mean - self.risk_aversion * effective_std
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            return random.choice(candidate_actions)
        return stats_list[best_idx].action

    # ------------------------------------------------------------------
    # 内部工具函数
    # ------------------------------------------------------------------
    def _select_with_ucb(
        self,
        stats_list: List[CandidateStats],
        total_n: int,
        virtual_counts: Optional[List[int]] = None,
    ) -> int:
        """
        对当前统计信息使用 UCB 选出一个动作索引。
        若某个动作尚未被采样 (n=0)，则优先采样它（UCB=+inf）。
        """
        best_ucb = -float("inf")
        best_idx = 0
        for i, s in enumerate(stats_list):
            vcount = 0 if virtual_counts is None else virtual_counts[i]
            n_eff = s.n + vcount
            if n_eff == 0:
                return i  # 优先给没被评估过的动作一次机会
            mean = s.mean
            ucb = mean + self.exploration_c * math.sqrt(
                math.log(max(total_n, 1) + 1.0) / n_eff
            )
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = i
        return best_idx

    def _evaluate_candidate_once(self, balls, table, my_targets, action, idx: int):
        """
        供并行调用的单次评估函数，返回 (idx, reward)。
        """
        reward, _, _ = self._simulate_action(balls, table, my_targets, action)
        return idx, reward

    def _simulate_action(self, balls, table, my_targets, action):
        """
        对单个动作做一次仿真：拷贝当前桌面和球状态，在噪声下击球并返回奖励。
        这里会把 action['candidate_type'] 传给 reward_fn，用于区分 defend / 非 defend。
        """
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

        act = self._apply_noise(action) if self.enable_noise else action
        try:
            cue.set_state(
                V0=act["V0"],
                phi=act["phi"],
                theta=act["theta"],
                a=act["a"],
                b=act["b"],
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "error",
                    category=RuntimeWarning,
                    module=r"pooltool\.ptmath\.roots\.core",
                )
                pt.simulate(shot, inplace=True)
        except Exception:
            # 仿真失败视为非常差的动作
            return -500.0, balls, table

        cand_type = None
        if isinstance(action, dict):
            cand_type = action.get("candidate_type", None)

        # 注意：奖励函数现在可以根据 candidate_type 区分进攻/防守
        reward = self.reward_fn(
            shot=shot,
            last_state=balls,
            player_targets=my_targets,
            candidate_type=cand_type,
        )
        # 额外轻微 shaping + 风险惩罚
        if cand_type != "defend":
            reward += self._first_hit_shaping(shot, my_targets)
        reward -= self._risk_penalty(act)
        return reward, shot.balls, sim_table

    def _first_hit_shaping(self, shot, my_targets):
        """
        对首碰球做一个轻微 shaping：优先碰到自己球。
        """
        first_contact_ball_id = None
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, "ids") else []
            if ("cushion" not in et) and ("pocket" not in et) and ("cue" in ids):
                other_ids = [i for i in ids if i != "cue"]
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break
        if first_contact_ball_id is None:
            return -self.first_hit_penalty
        return (
            self.first_hit_bonus
            if first_contact_ball_id in my_targets
            else -self.first_hit_penalty
        )

    def _risk_penalty(self, action: dict) -> float:
        """
        Softly penalize high speed / large side spin shots that are fragile under noise.
        """
        v0_over = max(0.0, float(action.get("V0", 0.0)) - self.risk_v0_softcap)
        spin_mag = abs(float(action.get("a", 0.0))) + abs(float(action.get("b", 0.0)))
        spin_over = max(0.0, spin_mag - self.risk_spin_softcap)
        return v0_over * self.risk_v0_weight + spin_over * self.risk_spin_weight

    def _apply_noise(self, action: dict) -> dict:
        noisy = dict(action)
        for k, std in self.noise_std.items():
            noisy[k] = float(
                np.clip(noisy[k] + np.random.normal(0, std), *self.pbounds[k])
            )
        if "phi" in noisy:
            noisy["phi"] = noisy["phi"] % 360.0
        return noisy

    def _random_action(self):
        return {
            "V0": random.uniform(
                max(0.8, self.pbounds["V0"][0]), min(6.5, self.pbounds["V0"][1])
            ),
            "phi": random.uniform(*self.pbounds["phi"]),
            "theta": random.uniform(0.0, min(12.0, self.pbounds["theta"][1])),
            "a": random.uniform(
                max(-0.2, self.pbounds["a"][0]), min(0.2, self.pbounds["a"][1])
            ),
            "b": random.uniform(
                max(-0.2, self.pbounds["b"][0]), min(0.2, self.pbounds["b"][1])
            ),
        }
