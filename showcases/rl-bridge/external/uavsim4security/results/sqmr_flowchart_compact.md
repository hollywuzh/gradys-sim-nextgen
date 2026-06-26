# SQMR 简洁版流程图

```mermaid
flowchart TD
    A["初始化 SQMR
    邻居表 / Q 值 / trust_score / 看门狗参数"] --> B["周期广播 HELLO"]
    B --> C["接收 HELLO
    更新邻居位置、速度、能量与链路质量 LQ"]
    C --> D["有数据包待转发"]
    D --> E["计算剩余时限
    得到所需推进速度 v_req"]
    E --> F["预测邻居未来位置
    计算实际推进速度 v_act"]
    F --> G["筛选满足 v_act >= v_req 的候选邻居"]
    G --> H["计算安全联合指标
    Score = k · Q · trust_score"]
    H --> I["选择下一跳
    argmax(k · Q · trust_score)"]
    I --> J["发送 DATA
    并注册 forwarding watchdog"]
    J --> K["收到 ACK
    按原始 QMR 更新 reward / alpha / Q"]
    K --> L["监听下一跳是否继续转发"]
    L --> M{"是否观测到继续转发?"}
    M -->|是| N["提高 trust_score
    T <- min(T_max, T + Δs)"]
    M -->|否| O["降低 trust_score
    T <- max(T_min, T - Δf)
    并压低 Q 值"]
    N --> D
    O --> D
```

## 图注建议

图 X 展示了 SQMR 协议的整体流程。与原始 QMR 相比，SQMR 在保留邻居发现、候选筛选、Q-learning 更新等基本结构的同时，新增了面向下一跳真实转发行为的看门狗观测与信任更新机制。节点在完成一次数据发送后，不仅依据 ACK 反馈更新 Q 值，还进一步监听下一跳是否继续转发该数据包，并据此动态调整该邻居的信任值，从而实现对黑洞和灰洞攻击的抑制。
