# uavsim4security 接入 RLlib 方案

本文档说明如何把 `uavsim4security` 作为第二个仿真后端接入当前 `rl-bridge`，并继续使用 RLlib/PPO 进行训练、评估和可视化。

> 当前状态：仓库 `abadhelenkockoamplin-bit/uavsim4security` 可以通过 GitHub 解析到远端，但本机 GitHub HTTPS 代理配置为 `127.0.0.1:7890`，当前代理端口未响应；SSH clone 也出现长时间等待。因此本文先给出稳定的接入架构。等仓库源码成功拉到本地后，再按本文的接口清单补齐 simulator 调用细节。

## 1. 总体原则

当前 `rl-bridge` 已经把仿真核心和强化学习接口拆开：

- `gradys_uav_service_env.py`：只负责仿真、状态、动作、奖励、指标。
- `gymnasium_adapter.py`：把 core env 包成 Gymnasium 单智能体/集中式环境。
- `rllib_multiagent_adapter.py`：把 core env 包成 RLlib `MultiAgentEnv`。
- `main_rllib_ppo_train.py`：注册 RLlib 环境，配置 PPO，训练并和 baseline 对比。

接入 `uavsim4security` 时，不建议直接改现有 GrADyS 环境；应该新增一个同形状的 core env：

```text
uavsim4security simulator
        |
        v
UAVSim4SecurityCoreEnv
        |
        v
RLlib MultiAgentEnv adapter
        |
        v
RLlib PPO / 后续 MAPPO-style shared policy
```

这样做的好处是：两个仿真后端可以共享训练脚本、评估脚本、日志格式、可视化流程和 PPO 调参经验。

## 2. 推荐目录结构

所有新增文件继续放在 `showcases/rl-bridge` 下：

```text
showcases/rl-bridge/
  external/
    uavsim4security/                       # 本地 clone 或子模块，是否提交取决于许可证和体积
  uavsim4security_core_env.py              # 新仿真后端的核心环境
  rllib_uavsim4security_adapter.py         # RLlib MultiAgentEnv 适配器
  main_uavsim4security_random.py           # 随机策略 smoke test
  main_uavsim4security_rllib_ppo_train.py  # PPO 训练入口
  main_uavsim4security_rllib_ppo_eval.py   # checkpoint 评估入口
  main_uavsim4security_visualize.py        # 轨迹/安全事件可视化
  docs 或 outputs/                         # 训练输出和图像仍放在 rl-bridge 内
```

第三方仓库建议先不要直接提交进主仓库。更稳妥的方式是：

1. 开发期 clone 到 `showcases/rl-bridge/external/uavsim4security`。
2. 若依赖稳定，再选择 Git submodule。
3. 若它支持 Python 包安装，则优先使用 `pip install git+...` 或本地 editable install。

## 3. Core Env 统一接口

`UAVSim4SecurityCoreEnv` 应尽量模仿现有 `GradysUAVServiceCoreEnv`，暴露以下接口：

```python
class UAVSim4SecurityCoreEnv:
    agents: list[str]

    @property
    def action_size(self) -> int: ...

    @property
    def observation_size(self) -> int: ...

    @property
    def time(self) -> float: ...

    def reset(self, seed: int | None = None) -> dict[str, list[float]]: ...

    def step(self, actions: dict[str, int]) -> StepResult: ...

    def close(self) -> None: ...

    def metrics_snapshot(self) -> dict[str, float]: ...

    def visualization_snapshot(self) -> dict[str, object]: ...
```

`StepResult` 可以直接复用当前 `gradys_uav_service_env.py` 中的形状：

```python
@dataclass
class StepResult:
    observations: dict[str, list[float]]
    rewards: dict[str, float]
    terminated: bool
    truncated: bool
    infos: dict[str, dict]
```

一旦 core env 形状统一，RLlib 适配层只需要替换 `self.core = ...`，其余 reset/step/action mask/team reward 基本都能复用。

## 4. 状态、动作、奖励设计

具体设计要看 `uavsim4security` 的场景定义。对于安全巡逻/防护类 UAV 仿真，建议从下面这个最小 MDP 开始。

### 4.1 Observation

每架 UAV 的 observation 建议包含：

- 自身状态：`x, y, z, vx, vy, battery/energy, busy/cooldown`
- 目标/威胁状态：最近 K 个 threat/intruder/attack/event 的相对位置、距离、速度、剩余时间
- 保护对象状态：base/asset 的相对位置、风险值、是否被攻击
- 全局进度：当前时间比例、剩余时间、已检测数量、已漏检数量

为了和现有 action-mask PPO 接轨，建议固定成定长向量：

```text
self_features + K * candidate_features
```

候选不足 K 个时用 0 padding，同时用 action mask 屏蔽无效候选。

### 4.2 Action

第一阶段建议使用离散动作，而不是一上来做连续控制：

```text
0: hover / patrol
1..K: 选择第 K 个候选威胁或任务
```

core env 内部再把“选择目标”转换成 simulator 的移动、拦截、巡逻或检测命令。

如果 `uavsim4security` 本身只支持连续控制，则可以先离散化：

```text
0 hover
1 north
2 south
3 west
4 east
5 move_to_nearest_threat
6 move_to_highest_risk_asset
```

先把 PPO 闭环跑通，再考虑连续动作空间，例如 RLlib PPO/SAC 对 `Box` action space 的训练。

### 4.3 Reward

推荐先用稠密+事件混合奖励：

```text
+R_detect       成功检测威胁
+R_intercept    成功拦截/处置威胁
+R_protect      保护对象未受损或风险下降
-R_breach       威胁突破/攻击成功
-R_miss         威胁超时未处理
-R_collision    碰撞或越界
-c_energy       能耗惩罚
-c_wait         未处理风险随时间积累的惩罚
+shaping        距离威胁变近、覆盖风险区域、减少响应时间
```

第一版不要把奖励设计得太复杂。建议先保证 nearest/rule-based baseline 有明确含义，然后让 PPO 以“超过 baseline”为目标。

## 5. RLlib 适配层

新增 `rllib_uavsim4security_adapter.py`，结构可以复制当前 `rllib_multiagent_adapter.py`：

```python
class UAVSim4SecurityRLlibEnv(MultiAgentEnv):
    def __init__(self, config=None):
        self.core = UAVSim4SecurityCoreEnv(env_config)
        self.possible_agents = list(self.core.agents)
        self.observation_space = spaces.Dict({
            "observations": spaces.Box(...),
            "action_mask": spaces.Box(...),
        })
        self.action_space = spaces.Discrete(self.core.action_size)

    def reset(self, *, seed=None, options=None):
        observations = self.core.reset(seed=seed)
        return self._as_observations(observations), infos

    def step(self, action_dict):
        result = self.core.step(action_dict)
        return observations, rewards, terminateds, truncateds, infos
```

保持 action mask 的原因是：安全场景中经常出现“当前没有可处置目标”“某 UAV 正在执行任务”“候选目标不足 K 个”等情况。mask 可以显著减少 PPO 早期探索的无效动作。

## 6. 训练流程

建议分四个阶段推进。

### 阶段 A：只跑仿真

目标：确认 simulator 可以 reset、step、关闭，并且状态可读取。

命令形态：

```bash
python showcases/rl-bridge/main_uavsim4security_random.py --episodes 3
```

验收标准：

- 不崩溃。
- 每个 episode 有固定时长或终止条件。
- 输出 time、reward、detect/intercept/miss、energy 等指标。

### 阶段 B：接 Gymnasium/RLlib smoke test

目标：确认 RLlib 能采样 rollout。

命令形态：

```bash
python showcases/rl-bridge/main_uavsim4security_rllib_ppo_train.py \
  --iterations 1 \
  --eval-episodes 2 \
  --num-uavs 2 \
  --episode-duration 30
```

验收标准：

- RLlib 能 build algorithm。
- 能完成至少 1 次 `algorithm.train()`。
- 能写出 `training_metrics.csv` 和 `comparison_summary.csv`。

### 阶段 C：PPO 对比 baseline

目标：比较 PPO 与 random/nearest/risk-first 等规则策略。

推荐 baseline：

- `random`：随机选合法动作。
- `nearest-threat`：选择最近威胁。
- `highest-risk`：选择风险最高的威胁或保护对象。
- `deadline-first`：如果有威胁剩余时间/攻击倒计时，优先处理最紧急者。

评估要固定随机种子集合，例如 50 或 100 个 episode：

```bash
python showcases/rl-bridge/main_uavsim4security_rllib_ppo_train.py \
  --iterations 50 \
  --eval-episodes 50 \
  --baseline-policies random,nearest-threat,highest-risk,deadline-first \
  --checkpoint \
  --output-dir outputs/uavsim4security_rllib_ppo_train
```

### 阶段 D：可视化和误差分析

目标：不是只看平均 reward，而是看 PPO 到底在怎样调度 UAV。

每个 episode 建议记录：

- UAV 轨迹。
- 威胁/入侵者轨迹。
- 检测、拦截、漏检、攻击成功事件。
- 每一步 action、action mask、reward 分解。
- episode 总 reward、success rate、breach rate、energy、response time。

输出文件建议：

```text
outputs/uavsim4security_visualization/
  ppo_vs_nearest_seedXXXX.png
  ppo_vs_nearest_seedXXXX.gif
  trajectory_seedXXXX.csv
  step_trace_seedXXXX.jsonl
```

## 7. 评价指标

安全类 UAV 场景建议不要只看 reward。至少同时看：

- `episode_reward_mean`：训练主指标。
- `detection_rate`：威胁检测率。
- `intercept_rate`：成功处置率。
- `breach_rate`：威胁突破率，越低越好。
- `mean_response_time`：平均响应时间。
- `energy_used`：总能耗。
- `collision_count` / `out_of_bounds_count`：安全约束。
- `coverage_ratio`：巡逻/监控覆盖率。

最终报告建议使用“均值 ± 标准差/置信区间”，并固定同一批评估种子比较 PPO 与 baseline。

## 8. 当前需要手动处理的网络问题

本机当前 Git 配置为：

```bash
git config --global --get http.proxy
git config --global --get https.proxy
# http://127.0.0.1:7890
```

如果你的代理软件没有启动，HTTPS clone 会失败。可以选择下面任一方式：

```bash
# 方式 1：启动本机代理，确保 127.0.0.1:7890 可用
git clone --depth 1 https://github.com/abadhelenkockoamplin-bit/uavsim4security.git \
  showcases/rl-bridge/external/uavsim4security

# 方式 2：使用 SSH
git clone --depth 1 git@github.com:abadhelenkockoamplin-bit/uavsim4security.git \
  showcases/rl-bridge/external/uavsim4security

# 方式 3：临时禁用 Git 代理直连
git -c http.proxy= -c https.proxy= clone --depth 1 \
  https://github.com/abadhelenkockoamplin-bit/uavsim4security.git \
  showcases/rl-bridge/external/uavsim4security
```

仓库成功拉取后，下一步就是检查它的 README、requirements、核心 simulator class、是否已有 Gym/Gymnasium 环境。如果已经有 Gym 环境，可以直接包一层 RLlib；如果只有 simulator class，则按本文新增 `UAVSim4SecurityCoreEnv`。

## 9. 最小实现清单

第一轮实现建议只做这些：

1. `uavsim4security_core_env.py`：封装 reset/step/state/action/reward。
2. `rllib_uavsim4security_adapter.py`：接入 RLlib MultiAgentEnv。
3. `main_uavsim4security_random.py`：确认仿真闭环。
4. `main_uavsim4security_rllib_ppo_train.py`：跑 1 iteration smoke test。
5. `main_uavsim4security_visualize.py`：画 PPO 与规则 baseline 的轨迹对比。

等这个闭环跑通后，再考虑：

- 多策略训练。
- 更复杂 action space。
- centralized critic。
- PettingZoo 并行接口。
- MAPPO 风格结构。
