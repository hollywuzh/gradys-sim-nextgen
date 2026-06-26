# smoke_sqmr_dataset_transitions.csv 字段说明

对应数据文件：

- [smoke_sqmr_dataset_transitions.csv](/C:/Users/ASUS/Desktop/项目1/UavNetSim-master%20(3)/extracted/UavNetSim-master/results/smoke_sqmr_dataset_transitions.csv)

该表中的每一行表示一次单步路由决策样本，可写成标准强化学习转移形式：

\[
(s, a, r, s')
\]

其中：

- \(s\)：动作执行前的当前状态
- \(a\)：当前节点选择的动作
- \(r\)：动作执行后的即时奖励
- \(s'\)：动作执行后的下一状态

---

## 1. 样本标识字段

### `protocol`

- 含义：该样本所对应的路由协议名称
- 取值示例：`SQMR`

### `seed`

- 含义：该次仿真运行所使用的随机种子
- 作用：用于实验复现和多随机种子统计分析

### `transition_id`

- 含义：单条状态转移样本的唯一标识
- 当前格式：通常写为 `当前节点ID:packet_id`
- 例子：`2:6`

### `packet_id`

- 含义：该条样本对应的数据包编号

### `src_id`

- 含义：该数据包的源节点 ID

### `dst_id`

- 含义：该数据包的目的节点 ID

### `current_node_id`

- 含义：当前执行下一跳选择的节点 ID
- 说明：该节点就是本条样本中做路由决策的智能体

### `time_us`

- 含义：本次决策发生时的仿真时间
- 单位：微秒（us）

### `feedback_time_us`

- 含义：本次决策收到反馈时的仿真时间
- 单位：微秒（us）
- 说明：反馈可能来自 ACK、watchdog 或超时惩罚

---

## 2. 动作字段 A

### `action_next_hop_id`

- 含义：本次决策选择的下一跳邻居 ID
- 作用：这是当前状态下执行的核心动作

### `action_has_route`

- 含义：本次决策是否找到了可用路由
- 可能取值：
  - `True`：找到下一跳
  - `False`：没有可用下一跳

---

## 3. 奖励与终止字段

### `reward`

- 含义：本次动作获得的即时奖励
- 说明：该奖励是后续 Q-learning 或 Dyna-Q 更新的核心量

### `terminal`

- 含义：该条转移是否可以看作终止样本
- 常见情况：
  - `ack_timeout`
  - `dropped`
  - `no_route`
  - `unfinished`

---

## 4. 反馈结果字段

### `ack_status`

- 含义：本次路由动作在链路层/网络层收到的 ACK 反馈状态
- 常见取值：
  - `ack_received`：成功收到 ACK
  - `ack_timeout`：等待 ACK 超时
  - `no_route`：没有下一跳可选
  - `unfinished`：仿真结束时还未完成反馈

### `forwarding_status`

- 含义：watchdog 对下一跳真实转发行为的观测结果
- 常见取值：
  - `forwarded`：观测到下一跳继续转发
  - `not_forwarded`：没有观测到继续转发
  - `not_observed`：没有足够证据完成观测

### `is_local_minimum`

- 含义：本次决策是否遇到局部最优/路由空洞问题
- 说明：常用于分析几何推进失败情形

### `max_q`

- 含义：本次更新时参考的后继最大 Q 值
- 作用：对应 Q-learning 更新中的 \(\max Q(s',a')\)

### `q_value_after`

- 含义：本次更新之后，该动作对应的 `q_value`
- 作用：用于观察学习更新后的动作价值变化

### `trust_score_after`

- 含义：本次更新之后，该邻居的 `trust_score`
- 作用：用于观察安全机制对下一跳信任值的修正结果

### `drop_reason`

- 含义：如果本次动作失败，对应的失败原因
- 常见示例：
  - `ack_timeout`

### `security_event`

- 含义：若该样本与某类安全事件直接关联，则在此给出事件标签

---

## 5. 攻击与安全标签字段

### `attack_names`

- 含义：本次样本所在仿真场景中激活的攻击名称列表
- 例子：`BLACKHOLE`、`GRAYHOLE`、`ACK_POISON`

### `attack_active`

- 含义：当前样本是否处于攻击场景
- 取值：
  - `True`
  - `False`

### `packet_attack_event_count`

- 含义：当前数据包在传输过程中关联到的攻击事件总数

### `watchdog_forwarded`

- 含义：watchdog 是否确认该下一跳真实执行了继续转发
- 取值：
  - `True`
  - `False`

### `watchdog_failed`

- 含义：watchdog 是否确认该下一跳未继续转发或存在可疑行为
- 取值：
  - `True`
  - `False`

---

## 6. 原始状态快照字段

### `raw_state_json`

- 含义：动作执行前的完整原始状态快照
- 内容包括：
  - 邻居总数
  - 主候选集
  - 次候选集
  - 每个邻居的 `LQ / Q / trust / delay / velocity / energy`
  - 被选邻居的详细快照

### `raw_next_state_json`

- 含义：动作执行后的完整下一状态快照
- 作用：适合后续重新定义状态空间、做更复杂特征工程或图神经网络输入构建

---

## 7. 当前状态字段 S：`s_` 前缀

这一组字段表示动作执行前的当前状态 \(s\)。

### 拓扑与候选集信息

#### `s_neighbor_count`

- 含义：当前邻居节点总数

#### `s_candidate_count`

- 含义：满足主候选条件的邻居数

#### `s_sub_candidate_count`

- 含义：次候选邻居数量

### 时限与分组属性

#### `s_required_velocity`

- 含义：当前数据包在剩余时限约束下所需的最小推进速度

#### `s_packet_deadline_remaining_us`

- 含义：当前数据包剩余可用时限
- 单位：微秒

#### `s_packet_ttl`

- 含义：当前数据包已使用的 TTL 状态

### 被选下一跳的连续特征

#### `s_chosen_next_hop_id`

- 含义：当前状态下被选中的下一跳节点 ID

#### `s_chosen_lq`

- 含义：被选邻居的链路质量 `LQ`

#### `s_chosen_k_factor`

- 含义：被选邻居的综合权重 `k`
- 说明：通常由链路质量和空间推进权重共同形成

#### `s_chosen_q_value`

- 含义：被选邻居在当前时刻的动作价值 `Q`

#### `s_chosen_trust_score`

- 含义：被选邻居在当前时刻的信任值 `trust_score`

#### `s_chosen_delay_us`

- 含义：被选邻居的一跳估计时延
- 单位：微秒

#### `s_chosen_remain_energy`

- 含义：被选邻居剩余能量

#### `s_chosen_actual_velocity`

- 含义：被选邻居相对于目的节点的实际推进速度

#### `s_chosen_is_candidate`

- 含义：被选邻居是否属于主候选集合

#### `s_chosen_is_sub_candidate`

- 含义：被选邻居是否属于次候选集合

---

## 8. 当前状态离散桶字段

这些字段是为了方便 `Dyna-Q`、表格型 RL 或离散化实验而自动生成的。

### `s_lq_bucket`

- 含义：链路质量分桶标签
- 常见类别：
  - `poor`
  - `fair`
  - `good`

### `s_delay_bucket`

- 含义：时延分桶标签
- 常见类别：
  - `small`
  - `medium`
  - `large`

### `s_trust_bucket`

- 含义：信任值分桶标签
- 常见类别：
  - `low`
  - `medium`
  - `high`

### `s_energy_bucket`

- 含义：剩余能量分桶标签

### `s_velocity_bucket`

- 含义：推进速度分桶标签
- 常见类别：
  - `negative`
  - `weak`
  - `strong`

### `s_attack_indicator`

- 含义：当前状态是否处于攻击场景
- 取值：
  - `0`
  - `1`

### `s_watchdog_failure_indicator`

- 含义：当前样本是否出现 watchdog 失败

### `s_watchdog_success_indicator`

- 含义：当前样本是否出现 watchdog 成功

### `s_state_label`

- 含义：状态标签
- 当前通常写为：`state`

---

## 9. 下一状态字段 S'：`sp_` 前缀

这一组字段表示动作执行后的下一状态 \(s'\)。

它们的语义与 `s_` 开头字段完全对应，只是时间点从“动作前”变成了“动作后”。

例如：

- `sp_neighbor_count`：下一状态中的邻居总数
- `sp_candidate_count`：下一状态中的主候选邻居数
- `sp_sub_candidate_count`：下一状态中的次候选邻居数
- `sp_required_velocity`：下一状态的最小所需推进速度
- `sp_chosen_lq`：下一状态下被选邻居的链路质量
- `sp_chosen_q_value`：下一状态下对应的 Q 值
- `sp_chosen_trust_score`：下一状态下对应的 trust 值
- `sp_delay_bucket`：下一状态下的时延分桶
- `sp_trust_bucket`：下一状态下的信任值分桶

因此：

- `s_...` 表示当前状态
- `action_...` 表示动作
- `reward` 表示即时奖励
- `sp_...` 表示下一状态

这正好构成标准强化学习训练样本。

---

## 10. 从 SQMR-DynaQ 角度最关键的字段

如果后续要基于该表构建 `SQMR-DynaQ`，最核心的字段通常包括：

### 状态输入

- `s_neighbor_count`
- `s_candidate_count`
- `s_required_velocity`
- `s_chosen_lq`
- `s_chosen_k_factor`
- `s_chosen_q_value`
- `s_chosen_trust_score`
- `s_chosen_delay_us`
- `s_chosen_remain_energy`
- `s_chosen_actual_velocity`

### 动作

- `action_next_hop_id`

### 奖励

- `reward`

### 下一状态

- `sp_chosen_lq`
- `sp_chosen_q_value`
- `sp_chosen_trust_score`
- `sp_delay_bucket`

### 安全标签

- `ack_status`
- `forwarding_status`
- `watchdog_failed`
- `attack_active`

---

## 11. 一行样本的直观理解

一条样本可以读作：

> 在时刻 `time_us`，节点 `current_node_id` 针对数据包 `packet_id` 做出一次下一跳决策，选择了 `action_next_hop_id` 作为动作。该决策时刻的路由状态由 `s_` 开头的一组特征描述，动作执行后得到奖励 `reward`，并进一步观测到 ACK、watchdog 与下一状态 `sp_` 特征，从而形成完整的状态转移样本。

这也是后续构建 `SQMR-DynaQ`、离散状态空间、奖励函数和安全约束策略的直接数据基础。
