# SQMR-DynaQ 数学模型与算法步骤

## 1. 设计动机

在当前项目中，`QMR` 利用时限约束、链路质量和 `Q` 值完成下一跳选择，`SQMR` 进一步引入了转发看门狗与邻居信任值 `trust_score`，能够识别“返回 ACK 但不继续转发”的黑洞和灰洞节点。  
不过，现有 `SQMR` 仍然主要依赖真实交互样本完成在线更新，存在两个局限：

1. 无人机拓扑变化快，单纯依赖实时样本会导致收敛较慢。
2. 在攻击出现初期，可疑邻居的负反馈样本仍然不足，导致协议绕开恶意节点的速度不够快。

为此，可在当前 `SQMR` 基础上进一步设计 `SQMR-DynaQ`。其核心思想是：

- 保留 `SQMR` 的安全感知路由骨架，即 `ACK + forwarding watchdog + trust score`。
- 将每次真实转发得到的经验写入局部模型 `Model`。
- 在每个时隙利用 `n_plan` 次规划更新对历史关键状态动作进行“模拟回放”。
- 使节点不仅从真实攻击反馈中学习，还能从内部模型中加速传播“某邻居不可信”的经验。

因此，`SQMR-DynaQ` 本质上是一种面向攻击环境的安全增强型模型学习路由方法。

---

## 2. 问题建模

### 2.1 网络与决策主体

考虑由若干无人机构成的自组网，记节点集合为

\[
\mathcal{U}=\{1,2,\dots,N\}.
\]

在离散决策时隙 \(k\) 内，当前节点 \(i\) 需要为待发送数据包选择下一跳。  
与原始 `QMR/SQMR` 一致，节点只能依据本地可观测信息进行分布式决策，因此可将单节点路由过程建模为局部马尔可夫决策过程：

\[
\mathcal{M}=\langle \mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma \rangle.
\]

其中：

- \(\mathcal{S}\) 为状态空间；
- \(\mathcal{A}\) 为动作空间；
- \(\mathcal{P}\) 为状态转移概率；
- \(\mathcal{R}\) 为即时奖励函数；
- \(\gamma\in(0,1)\) 为折扣因子。

在当前项目中，`SQMR` 已经显式维护以下关键信息：

- 邻居位置 `recorded_pos`
- 邻居速度 `recorded_vel`
- 邻居剩余能量 `remain_energy`
- 链路质量 `lq`
- 一跳估计时延 `delay`
- 学习价值 `q_value`
- 安全信誉 `trust_score`

因此，`SQMR-DynaQ` 的建模必须直接兼容这些局部观测量。

### 2.2 决策目标

对任意源节点而言，路由决策目标不是单一最短路，而是联合优化以下多目标性能：

\[
\max \; \mathbb{E}\left[\sum_{k=0}^{\infty}\gamma^k r^{(k)}\right],
\]

其中即时奖励 \(r^{(k)}\) 同时刻画：

- 成功交付收益；
- 时延代价；
- 能耗代价；
- 链路风险；
- 邻居安全可信度；
- 攻击惩罚。

换言之，`SQMR-DynaQ` 的目标是在高动态、高干扰和存在恶意中继的环境中，学习“既能到达、又能避险、还能快速收敛”的下一跳选择策略。

---

## 3. 状态空间设计

### 3.1 邻居集合

在时隙 \(k\)，节点 \(i\) 的可用邻居集合记为

\[
\mathcal{N}_i^{(k)}=\{j \mid j \text{ 在节点 } i \text{ 通信范围内}\}.
\]

记邻居数为

\[
n_i^{(k)} = |\mathcal{N}_i^{(k)}|.
\]

### 3.2 邻居级原始特征

对任一邻居 \(j\in \mathcal{N}_i^{(k)}\)，定义其局部特征向量为

\[
\mathbf{x}_{ij}^{(k)}=
\left[
d_{ij}^{(k)},
\;v_{ij}^{(k)},
\;E_j^{(k)},
\;LQ_{ij}^{(k)},
\;D_{ij}^{(k)},
\;Q_{ij}^{(k)},
\;T_{ij}^{(k)}
\right],
\]

其中：

- \(d_{ij}^{(k)}\) 为节点 \(i\) 到邻居 \(j\) 的欧氏距离；
- \(v_{ij}^{(k)}\) 为邻居相对运动速度；
- \(E_j^{(k)}\) 为邻居剩余能量；
- \(LQ_{ij}^{(k)}\) 为链路质量；
- \(D_{ij}^{(k)}\) 为估计一跳总时延；
- \(Q_{ij}^{(k)}\) 为当前路由价值；
- \(T_{ij}^{(k)}\) 为当前安全信任值。

### 3.3 安全与攻击观测

为了与当前 `SQMR` 的防御模块对应，引入局部安全观测量：

\[
\mathbf{z}_{ij}^{(k)}=
\left[
\hat a_{ij}^{(k)},
\;c_{ij}^{(k)},
\;u_{ij}^{(k)}
\right],
\]

其中：

- \(\hat a_{ij}^{(k)}\in\{0,1\}\) 表示最近一次转发是否触发异常标记；
- \(c_{ij}^{(k)}\) 表示邻居 \(j\) 的累计成功转发确认次数；
- \(u_{ij}^{(k)}\) 表示邻居 \(j\) 的累计失效或疑似丢包次数。

这三项可直接映射到当前项目中的：

- `trust_success_count`
- `trust_failure_count`
- watchdog 超时触发记录

### 3.4 状态压缩

由于邻居集合大小是动态变化的，若直接对原始状态使用表格型强化学习会导致维数过大。因此 `SQMR-DynaQ` 采用“候选动作驱动的局部状态抽象”，仅对每个候选邻居构造离散状态：

\[
s_{ij}^{(k)}=
\left(
b_1(LQ_{ij}^{(k)}),
b_2(D_{ij}^{(k)}),
b_3(E_j^{(k)}),
b_4(T_{ij}^{(k)}),
b_5(\Delta d_{ij}^{(k)}),
b_6(\hat a_{ij}^{(k)})
\right),
\]

其中 \(b_m(\cdot)\) 表示分桶量化函数，\(\Delta d_{ij}^{(k)}\) 表示邻居相对目的节点的推进收益。  
例如可设：

- `LQ` 分为 `poor / fair / good`
- `delay` 分为 `large / medium / small`
- `trust` 分为 `low / medium / high`
- 推进收益分为 `negative / weak / strong`

这样可以显著降低状态空间规模，更符合“用 Dyna-Q 代替 DRL 以追求更快收敛”的目标。

### 3.5 总体状态定义

故节点 \(i\) 在时隙 \(k\) 的决策状态可表示为

\[
s_i^{(k)}=\left\{\left(s_{ij}^{(k)},j\right)\mid j\in\mathcal{N}_i^{(k)}\right\}.
\]

在实际执行时，节点不需要对整个邻居排列组合建立全局大状态，只需对每个候选邻居维护“离散局部状态 - 动作价值”映射即可。

---

## 4. 动作空间设计

在当前项目的 `QMR/SQMR` 实现中，每次发送一个数据包时本质上要做的是“从邻居集合中选一个下一跳”。因此，`SQMR-DynaQ` 的基础动作空间定义为

\[
\mathcal{A}_i^{(k)} = \mathcal{N}_i^{(k)} \cup \{a_{\emptyset}\},
\]

其中：

- 动作 \(a=j\) 表示选择邻居 \(j\) 作为下一跳；
- 动作 \(a_{\emptyset}\) 表示本时隙暂不转发或继续等待。

若后续你要进一步扩展到“下一跳 + 功率控制”联合决策，则可扩展为

\[
a_i^{(k)}=(j,p_\ell),\qquad
j\in \mathcal{N}_i^{(k)},\; p_\ell \in \mathcal{P},
\]

其中 \(\mathcal{P}\) 为离散功率等级集合。  
但基于你当前项目的实现基础，第一版 `SQMR-DynaQ` 更建议先采用单动作版本，即只学习下一跳选择，这样最容易直接落地。

---

## 5. 候选筛选与安全路由度量

### 5.1 时限约束筛选

沿用 `QMR/SQMR` 的核心约束。对待转发数据包，设其剩余生存时间为 \(T_{\text{rem}}^{(k)}\)，当前节点到目的节点距离为 \(D(i,d)\)，则所需最小推进速度为

\[
v_{\text{req}}^{(k)}=\frac{D(i,d)}{T_{\text{rem}}^{(k)}}.
\]

对邻居 \(j\)，利用未来位置预测得到其实际推进速度

\[
v_{\text{act}}^{(k)}(i,j)=
\frac{D(i,d)-D(j_{t+\tau},d)}{\tau_{ij}^{(k)}},
\]

其中 \(\tau_{ij}^{(k)}\) 为估计一跳时延。仅当

\[
v_{\text{act}}^{(k)}(i,j)\ge v_{\text{req}}^{(k)}
\]

时，邻居 \(j\) 才进入候选集 \(\mathcal{C}_i^{(k)}\)。

### 5.2 综合链路权重

与原始实现一致，记距离权重为

\[
m_{ij}^{(k)}=1-\frac{d_{ij}^{(k)}}{R},
\]

其中 \(R\) 为通信半径。链路质量为

\[
LQ_{ij}^{(k)} = d_f^{(k)} \cdot d_r^{(k)},
\]

故综合链路权重为

\[
k_{ij}^{(k)}=m_{ij}^{(k)}LQ_{ij}^{(k)}.
\]

### 5.3 SQMR-DynaQ 安全评分

为兼容 `SQMR` 中“信任驱动绕开恶意邻居”的思想，定义动作 \(a=j\) 的安全启发评分为

\[
H_{ij}^{(k)} = k_{ij}^{(k)} \cdot T_{ij}^{(k)}.
\]

进一步，将 Dyna-Q 学到的状态动作价值 \(Q(s_{ij}^{(k)},j)\) 与该启发项融合，得到最终决策分数：

\[
F_{ij}^{(k)} = H_{ij}^{(k)} \cdot Q(s_{ij}^{(k)},j).
\]

路由选择规则为

\[
a_i^{(k)}=\arg\max_{j\in\mathcal{C}_i^{(k)}} F_{ij}^{(k)}.
\]

若候选集合为空，则回退到次候选集或等待动作，这一点与当前 `QMR/SQMR` 的回退机制保持一致。

---

## 6. 奖励函数设计

### 6.1 即时奖励分解

定义动作 \(a=j\) 执行后的即时奖励为

\[
r_{ij}^{(k)}=
\lambda_1 r_{\text{succ}}
\lambda_2 r_{\text{prog}}
\lambda_3 r_{\text{delay}}
\lambda_4 r_{\text{energy}}
\lambda_5 r_{\text{trust}}
\lambda_6 r_{\text{attack}},
\]

其中 \(\lambda_m \ge 0\) 且 \(\sum_m \lambda_m = 1\)。

### 6.2 成功交付项

若下一跳成功返回 `ACK`，则给出基础成功奖励：

\[
r_{\text{succ}}=
\begin{cases}
1, & \text{收到 ACK}\\
-1, & \text{ACK 超时或发送失败}
\end{cases}
\]

### 6.3 推进收益项

若邻居能有效缩短到目的节点距离，则

\[
r_{\text{prog}}=
\frac{D(i,d)-D(j,d)}{R}.
\]

该项鼓励路由持续向目的节点推进。

### 6.4 时延代价项

设一跳估计时延为 \(\tau_{ij}^{(k)}\)，则可定义

\[
r_{\text{delay}}=-\frac{\tau_{ij}^{(k)}}{\tau_{\max}}.
\]

### 6.5 能量代价项

记邻居剩余能量占比为

\[
e_j^{(k)}=\frac{E_j^{(k)}}{E_{\text{init}}},
\]

则能量项可写为

\[
r_{\text{energy}} = e_j^{(k)}.
\]

这与当前 `QMR` 奖励中“时延 + 剩余能量”联合优化的思想一致。

### 6.6 信任增强项

若看门狗在观察窗口内确认邻居 \(j\) 确实继续转发，则

\[
r_{\text{trust}}=+\eta_s;
\]

若只收到 `ACK` 但未观察到继续转发，则

\[
r_{\text{trust}}=-\eta_f,
\qquad \eta_f>\eta_s>0.
\]

### 6.7 攻击惩罚项

若节点触发黑洞、灰洞或 ACK 欺骗可疑事件，则定义

\[
r_{\text{attack}}=-\xi_{ij}^{(k)},
\]

其中 \(\xi_{ij}^{(k)}\) 可由如下方式构造：

\[
\xi_{ij}^{(k)}=
\omega_1 \mathbb{I}_{\text{watchdog-fail}}
\omega_2 \mathbb{I}_{\text{ack-anomaly}}
\omega_3 \big(1-T_{ij}^{(k)}\big).
\]

这样能够将“显式异常”和“长期低信誉”同时纳入奖励惩罚。

### 6.8 与当前 QMR 奖励的兼容表达

若希望与项目内已有 `QMR` 奖励函数完全对接，也可写成

\[
r_{ij}^{(k)}=
\beta_1\Big(\omega e^{-\tau_{ij}^{(k)}}+(1-\omega)e_j^{(k)}\Big)
\beta_2 T_{ij}^{(k)}
\beta_3 r_{\text{attack}},
\]

即在当前 `QMR` 奖励骨架外，额外叠加信任与攻击惩罚项。

---

## 7. Dyna-Q 学习与规划模型

### 7.1 真实经验更新

节点执行动作 \(a_i^{(k)}\) 后，获得四元组经验

\[
\big(s_i^{(k)},a_i^{(k)},r_i^{(k)},s_i^{(k+1)}\big).
\]

对所选邻居 \(j\)，执行标准 Q-learning 更新：

\[
Q(s_i^{(k)},a_i^{(k)}) \leftarrow
Q(s_i^{(k)},a_i^{(k)}) +
\alpha
\left[
r_i^{(k)} +
\gamma \max_{a'} Q(s_i^{(k+1)},a') -
Q(s_i^{(k)},a_i^{(k)})
\right].
\]

其中 \(\alpha\) 为学习率。若你希望完全继承当前 `QMR` 的实现风格，则 \(\alpha\) 仍可依据一跳时延波动自适应调整。

### 7.2 局部环境模型

在 Dyna-Q 中，每次真实交互后，将经验写入局部模型

\[
\mathcal{M}_i:
(s,a)\mapsto (\hat r,\hat s').
\]

对当前项目而言，该模型不需要复杂神经网络，可直接采用表格式或字典式经验缓存：

\[
\mathcal{M}_i[s,a]=(\hat r_{sa}, \hat s'_{sa}, \nu_{sa}),
\]

其中 \(\nu_{sa}\) 表示该状态动作对的访问次数或优先级。

### 7.3 规划更新

在每次真实更新后，从模型中抽取 \(n_{\text{plan}}\) 个历史状态动作对进行模拟回放。对第 \(m\) 个被抽中的样本 \((\tilde s,\tilde a)\)，设模型给出的回报和后继状态分别为 \((\hat r,\tilde s')\)，则执行

\[
Q(\tilde s,\tilde a) \leftarrow
Q(\tilde s,\tilde a) +
\alpha_p
\left[
\hat r + \gamma \max_{a'}Q(\tilde s',a') - Q(\tilde s,\tilde a)
\right].
\]

其中 \(\alpha_p\) 为规划学习率。

### 7.4 安全优先采样

为增强对攻击的快速适应能力，规划阶段不应均匀随机采样，而应优先选择：

- 最近触发 watchdog 失败的状态动作；
- `trust_score` 显著下降的邻居动作；
- `ACK` 异常但真实转发未确认的经验；
- 高流量、高关键性链路上的经验。

可定义优先级

\[
\Pi(s,a)=
\mu_1 \Delta^{-}_{trust}(s,a) +
\mu_2 \mathbb{I}_{\text{watchdog-fail}} +
\mu_3 |\delta(s,a)|,
\]

其中：

- \(\Delta^{-}_{trust}(s,a)\) 为该动作导致的信任下降幅度；
- \(\delta(s,a)\) 为 TD 误差。

规划样本按 \(\Pi(s,a)\) 从大到小选取，即形成“安全感知 Prioritized Dyna-Q”。

### 7.5 信任值更新

延续当前 `SQMR` 机制。若在看门狗窗口内确认继续转发，则

\[
T_{ij}^{(k+1)}=
\min(T_{\max},T_{ij}^{(k)}+\Delta_s).
\]

若未确认继续转发，则

\[
T_{ij}^{(k+1)}=
\max(T_{\min},T_{ij}^{(k)}-\Delta_f).
\]

其中在当前项目中可直接继承已使用参数：

- \(T_{\max}=1.0\)
- \(T_{\min}=0.05\)
- \(\Delta_s=0.03\)
- \(\Delta_f=0.25\)

### 7.6 攻击下的 Q 值快速压制

若邻居 \(j\) 被判断为疑似恶意节点，则对相应动作值施加额外抑制：

\[
Q(s_{ij},j)\leftarrow \max(Q_{\min}, \rho Q(s_{ij},j)),
\]

其中 \(\rho\in(0,1)\)。  
当前 `SQMR` 中已经采用了 \(\rho=0.6\) 的思想，这一机制可直接沿用到 `SQMR-DynaQ`。

---

## 8. 策略与收敛解释

### 8.1 行为策略

在行为策略上，建议采用安全约束下的 \(\epsilon\)-greedy：

\[
\pi(a|s)=
\begin{cases}
1-\epsilon+\dfrac{\epsilon}{|\mathcal{A}_s|}, & a=\arg\max_{a'}F(s,a')\\[1ex]
\dfrac{\epsilon}{|\mathcal{A}_s|}, & \text{otherwise}
\end{cases}
\]

但仅允许在“信任值不低于阈值 \(T_{\text{safe}}\)”的邻居中探索。  
即当

\[
T_{ij}^{(k)} < T_{\text{safe}}
\]

时，动作 \(a=j\) 直接从可探索集合中剔除。

### 8.2 快速收敛原因

相对于深度强化学习，`SQMR-DynaQ` 更适合你当前项目，原因在于：

1. 当前路由决策本身已经有较强的结构先验，如 `v_req`、`v_act`、`LQ`、`trust_score`，不必依赖深网络自行抽特征。
2. 局部离散状态 + 候选筛选能把状态动作规模压到可控范围内。
3. Dyna-Q 的规划回放可在攻击出现后快速传播负经验，比纯在线 `SQMR` 收敛更快。
4. 对博士论文而言，这条路线兼具“可解释性、可实现性、可做理论分析、可做攻击实验”。

---

## 9. SQMR-DynaQ 算法步骤

### 步骤 1：初始化

对每个节点 \(i\)：

1. 初始化邻居表、链路质量表和 `trust_score`。
2. 初始化动作价值表 \(Q(s,a)\)。
3. 初始化局部环境模型 \(\mathcal{M}_i\)。
4. 设置学习率 \(\alpha\)、折扣因子 \(\gamma\)、探索率 \(\epsilon\)。
5. 设置规划次数 \(n_{\text{plan}}\)、信任阈值 \(T_{\text{safe}}\) 和 watchdog 窗口。

### 步骤 2：邻居发现与状态维护

节点周期性发送 `HELLO` 报文，更新：

- 邻居位置与速度；
- 剩余能量；
- 链路质量 `LQ`；
- 一跳时延；
- `trust_score`；
- 成功/失败转发计数。

### 步骤 3：候选邻居筛选

对待发数据包：

1. 计算剩余时限 \(T_{\text{rem}}\)。
2. 计算所需推进速度 \(v_{\text{req}}\)。
3. 对每个邻居预测未来位置，计算 \(v_{\text{act}}\)。
4. 仅保留满足 \(v_{\text{act}}\ge v_{\text{req}}\) 的邻居。
5. 剔除 \(T_{ij}<T_{\text{safe}}\) 的低信誉邻居。

### 步骤 4：构造局部离散状态

对每个候选邻居 \(j\)，将 `LQ`、`delay`、`energy`、`trust`、推进收益和异常标志分桶，得到离散状态 \(s_{ij}\)。

### 步骤 5：动作选择

对每个候选动作 \(a=j\)，计算

\[
F_{ij}=k_{ij} \cdot T_{ij} \cdot Q(s_{ij},j).
\]

采用安全约束的 \(\epsilon\)-greedy 选择动作：

- 以概率 \(1-\epsilon\) 选择最大评分动作；
- 以概率 \(\epsilon\) 在安全候选动作中随机探索。

### 步骤 6：执行转发并记录经验

节点将数据发送至下一跳，并注册 watchdog 观察记录，随后等待：

- `ACK` 反馈；
- 下一跳真实继续转发的证据。

### 步骤 7：更新信任与即时奖励

1. 若成功收到 `ACK`，记录正向交互结果。
2. 若观察到下一跳继续转发，则提升 `trust_score`。
3. 若 watchdog 超时未观测到继续转发，则降低 `trust_score` 并压制相应 `Q` 值。
4. 根据成功率、推进收益、时延、能量和攻击异常计算即时奖励 \(r\)。

### 步骤 8：真实样本 Q 更新

利用真实经验 \((s,a,r,s')\) 执行一次 Q-learning 更新。

### 步骤 9：模型写入

将 \((s,a)\mapsto(r,s')\) 写入局部模型 \(\mathcal{M}_i\)，并更新该样本优先级。

### 步骤 10：规划回放

重复 \(n_{\text{plan}}\) 次：

1. 从模型中按优先级抽取一个历史状态动作对；
2. 读取其模拟奖励与后继状态；
3. 执行一次规划型 Q 更新。

### 步骤 11：循环执行

当后续数据包到来时重复步骤 2 至步骤 10，直至仿真结束。

---

## 10. 论文中的方法命名建议

你可以把该方法写成：

> `SQMR-DynaQ: A Secure QMR Routing Scheme with Trust-Aware Dyna-Q Planning for UAV Ad Hoc Networks`

或者中文表达为：

> 一种基于信任感知 Dyna-Q 规划的安全增强型无人机自组网路由方法

---

## 11. 与 QMR、SQMR 的关系

三者的层次关系可以概括为：

- `QMR`：时限约束 + 链路质量 + Q-learning
- `SQMR`：`QMR` + watchdog + trust score
- `SQMR-DynaQ`：`SQMR` + 环境模型 + 规划回放 + 安全优先采样

因此，`SQMR-DynaQ` 不是推翻你现在的项目结构，而是在现有 `SQMR` 之上做一次“安全学习加速器”升级。这一点非常适合写成博士论文里的连续演进路线：

1. 先证明 `QMR` 在黑洞/灰洞下脆弱。
2. 再提出 `SQMR` 做安全增强。
3. 最后提出 `SQMR-DynaQ` 解决 `SQMR` 收敛速度与攻击适应速度问题。

---

## 12. 可直接落地到当前项目的实现建议

若下一步你要我继续实现第一版代码，最稳妥的落地方式是：

1. 在 `routing/sqmr_dynaq/` 下新增 `sqmr_dynaq.py`、`sqmr_dynaq_table.py`、`sqmr_dynaq_config.py`。
2. 复用当前 `SQMR` 的 `HELLO`、`ACK`、watchdog 和 `trust_score` 逻辑。
3. 新增一个离散状态编码器 `encode_state(neighbor_id, packet)`。
4. 新增 `model_memory[(state, action)] = (reward, next_state, priority)`。
5. 在每次真实 `ACK/watchdog` 更新后执行 `n_plan` 次 Dyna-Q 回放。
6. 在下一跳选择时用 `k * trust * Q(state, action)` 取代当前单一 `k * q_value * trust_score`。

如果你愿意，我下一步可以直接继续帮你做两件事之一：

1. 把这份内容再转成论文可直接粘贴的 `LaTeX` 小节。
2. 直接在当前项目里实现第一版 `SQMR-DynaQ` 原型代码。
