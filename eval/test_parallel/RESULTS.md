# 并行评估测试结果

**测试时间**: 2024-12-27 (更新)  
**测试环境**: 128 CPU cores, 100 workers

---

## 0. Physics-based Agent (物理模拟) vs BasicAgent

| 指标 | 值 |
|------|-----|
| **总局数** | 200 |
| **有效局数** | 200 |
| **错误/超时** | 0 |
| **总耗时** | 8.0 分钟 |
| **平均速度** | 0.42 局/秒 |
| **平均击球数** | 39.1 |

| Agent | 胜场 | 胜率 |
|-------|------|------|
| **Physics Agent** | 120 | **60.00%** |
| BasicAgent | 64 | 32.00% |
| 平局 | 16 | 8.00% |

**说明**: 物理模拟 Agent 使用几何计算和物理引擎模拟来决策。该工作后来由队友以 MCTS 方法延续改进。

---

## 测试配置

| 参数 | 值 |
|------|-----|
| 测试局数 | 1000 |
| 并行进程 | 80-100 |
| 超时时间 | 600s |
| Agent 来源 | `time_limit_mcts/agent.py`, `future_agent/agent.py` |

---

## 测试结果

### 1. merge_basic (NewAgent) vs BasicAgent

| 指标 | 值 |
|------|-----|
| **总局数** | 1000 |
| **有效局数** | 1000 |
| **错误/超时** | 0 |
| **总耗时** | 22.5 分钟 |
| **平均速度** | 0.74 局/秒 |
| **平均击球数** | 28.6 |

| Agent | 胜场 | 胜率 |
|-------|------|------|
| **NewAgent (merge_basic)** | 904 | **90.40%** |
| BasicAgent | 80 | 8.00% |
| 平局 | 16 | 1.60% |

**结论**: NewAgent 对 BasicAgent 有压倒性优势，胜率超过 90%。

---

### 2. merge_basic (NewAgent) vs BasicAgentPro

| 指标 | 值 |
|------|-----|
| **已完成局数** | 990/1000 |
| **状态** | 最后 10 局卡住（可能是 Pro 模拟超时） |

| Agent | 胜场 | 胜率 (基于 990 局) |
|-------|------|------|
| **NewAgent (merge_basic)** | 556 | **56.2%** |
| BasicAgentPro | 434 | 43.8% |
| 平局 | 0 | 0% |

**结论**: NewAgent 对 BasicAgentPro 有一定优势，胜率约 56%。

---

### 3. future (NewAgent) vs BasicAgentPro

| 指标 | 值 |
|------|-----|
| **总局数** | 1000 |
| **有效局数** | 1000 |
| **错误/超时** | 0 |
| **总耗时** | 19.5 分钟 |
| **平均速度** | 0.85 局/秒 |
| **平均击球数** | 19.2 |

| Agent | 胜场 | 胜率 |
|-------|------|------|
| **NewAgent (future)** | 582 | **58.20%** |
| BasicAgentPro | 417 | 41.70% |
| 平局 | 1 | 0.10% |

**结论**: Future 分支比 merge_basic 有约 2% 的提升 (56.2% → 58.2%)，且平均击球数更少 (28.6 → 19.2)，说明决策效率更高。

---

### 4. long_mcts (NewAgent) vs BasicAgentPro ⭐ 最新最强

| 指标 | 值 |
|------|-----|
| **总局数** | 1000 |
| **有效局数** | 1000 |
| **错误/超时** | 0 |
| **总耗时** | 65.5 分钟 |
| **平均速度** | 0.25 局/秒 |
| **平均击球数** | 15.6 |

| Agent | 胜场 | 胜率 |
|-------|------|------|
| **NewAgent (long_mcts)** | 806 | **80.60%** |
| BasicAgentPro | 194 | 19.40% |
| 平局 | 0 | 0.00% |

**结论**: Long MCTS 模型以 80.6% 的胜率大幅领先 BasicAgentPro，比 future 分支提升 22.4%！虽然决策速度较慢（0.25 vs 0.85 局/秒），但胜率和击球效率（15.6 vs 19.2）均显著提升。

---

## 各分支胜率汇总

| 分支 | 对手 | 胜率 | 平均击球数 | 备注 |
|------|------|------|------|------|
| time_limit_mcts | BasicAgent | 93.75% | - | 队友报告 |
| robust | BasicAgent | 95% | - | 队友报告 |
| merge_basic | BasicAgent | 90.40% | 28.6 | 本次测试 |
| merge_basic | BasicAgentPro | 56.2% | - | 本次测试 |
| future | BasicAgentPro | 58.20% | 19.2 | 本次测试 |
| **long_mcts** | BasicAgentPro | **80.60%** | **15.6** | ⭐ 最新最强 |

---

## 备注

1. **BasicAgent**: 基于贝叶斯优化的智能 Agent（30 次迭代）
2. **BasicAgentPro**: 基于 MCTS 的进阶 Agent（50 次模拟）
3. **NewAgent (merge_basic)**: 融合了 robust 分支的 MCTS 和 BasicAgentPro 策略
4. **NewAgent (future)**: 包含 future reward 和 eight penalty 优化，带时间限制
5. **NewAgent (long_mcts)**: 长时间 MCTS 模型（64 候选，400 模拟），不限时间，追求最高胜率

---

## 测试命令

```bash
# merge_basic vs BasicAgent
python test_parallel/parallel_evaluate.py \
    --num_games 1000 --num_workers 80 \
    --agent_a merge_basic --agent_b basic

# merge_basic vs BasicAgentPro
python test_parallel/parallel_evaluate.py \
    --num_games 1000 --num_workers 80 \
    --agent_a merge_basic --agent_b pro

# future vs BasicAgentPro
python test_parallel/parallel_evaluate.py \
    --num_games 1000 --num_workers 100 --cpu_cores 128 \
    --agent_a future --agent_b pro

# long_mcts vs BasicAgentPro (最强)
python test_parallel/parallel_evaluate.py \
    --num_games 1000 --num_workers 100 --cpu_cores 128 \
    --agent_a long --agent_b pro --timeout 600
```
