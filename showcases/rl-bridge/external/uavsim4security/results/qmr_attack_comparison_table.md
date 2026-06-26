# QMR 场景下基线与多类攻击结果对比

仿真设置统一为：`QMR + CSMA/CA + 3D Gauss-Markov + seed=2025 + 30 s`。  
网络层攻击与物理层干扰的有效时间窗统一为 `10 s` 到 `18 s`。

## 结果表

| 场景 | 攻击配置 | PDR (%) | 相对基线变化 (pp) | E2E Delay (ms) | Routing Load | Throughput (Kbps) | Hop Count | Collision | 安全事件 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 无攻击 | 94.03 | 0.00 | 26.00 | 0.846 | 638.43 | 2.447 | 39 | 无 |
| Blackhole | `BH@UAV5`, 丢包概率 `1.0` | 82.10 | -11.94 | 25.38 | 0.969 | 684.06 | 2.359 | 28 | `blackhole_drop_count=90` |
| Grayhole | `GH@UAV5`, 丢包概率 `0.5` | 87.67 | -6.37 | 25.69 | 0.908 | 657.32 | 2.415 | 68 | `grayhole_drop_count=48` |
| Blackhole + Grayhole | `BH@UAV5`, `GH@UAV1`, 丢包概率分别为 `1.0/0.5` | 80.24 | -13.79 | 25.16 | 0.992 | 692.50 | 2.342 | 28 | `blackhole_drop_count=78`, `grayhole_drop_count=27` |
| PHY Jamming | 关键中继链路附近干扰源，坐标 `(122.05, 262.49, 50.62)`，功率 `0.12 W`，半径 `120 m` | 67.77 | -26.26 | 39.23 | 1.174 | 589.64 | 2.429 | 1547 | `phy_jamming_interference_count=1513` |

## 图示文件

- 基线动图：[qmr_baseline_no_jamming.gif](/C:/Users/ASUS/Desktop/项目1/UavNetSim-master%20(3)/extracted/UavNetSim-master/qmr_baseline_no_jamming.gif)
- 黑洞动图：[qmr_blackhole_uav5.gif](/C:/Users/ASUS/Desktop/项目1/UavNetSim-master%20(3)/extracted/UavNetSim-master/qmr_blackhole_uav5.gif)
- 灰洞动图：[qmr_grayhole_uav5.gif](/C:/Users/ASUS/Desktop/项目1/UavNetSim-master%20(3)/extracted/UavNetSim-master/qmr_grayhole_uav5.gif)
- 联合攻击动图：[qmr_blackhole_grayhole_combo.gif](/C:/Users/ASUS/Desktop/项目1/UavNetSim-master%20(3)/extracted/UavNetSim-master/qmr_blackhole_grayhole_combo.gif)
- 物理层干扰动图：[qmr_key_relay_jamming.gif](/C:/Users/ASUS/Desktop/项目1/UavNetSim-master%20(3)/extracted/UavNetSim-master/qmr_key_relay_jamming.gif)

## 配图说明

### 图注推荐写法

图 X 展示了 `QMR` 协议在基线、黑洞攻击、灰洞攻击、黑洞-灰洞联合攻击以及关键中继链路附近物理层干扰五种场景下的动态仿真过程。图中蓝色箭头表示 `DATA` 转发链路，绿色连线表示 `ACK` 反馈链路；红色圆圈节点表示黑洞节点，橙色方框节点表示灰洞节点，节点上方的 `drop` 标记表示该节点在当前时间窗内刚刚丢弃过数据包；红色 `X` 及其球形作用域表示物理层干扰源及其有效干扰范围。

### 正文说明推荐写法

从动态仿真图可以观察到，基线场景下网络中的 `DATA/ACK` 转发链路较为连续，链路切换主要由机动性驱动。引入网络层攻击后，攻击节点附近的转发链路明显变稀疏，其中黑洞攻击由于对经过节点的转发数据包进行确定性丢弃，导致 `UAV5` 周边链路在攻击窗口内快速“断流”；灰洞攻击则表现为间歇性丢包，链路退化程度弱于黑洞，但仍能在局部区域观察到转发不稳定。联合攻击场景下，`UAV5` 的黑洞丢弃与 `UAV1` 的灰洞扰动共同作用，使得关键中继路径同时遭受强丢弃和概率性扰动，其整体投递率下降到 `80.24%`，低于单独黑洞和单独灰洞场景。

相比之下，物理层干扰并不直接修改网络层转发决策，而是通过恶化链路信噪比与增加冲突强度，导致更广范围的通信退化。由于干扰源被布置在关键中继链路附近，攻击窗口内大量 `DATA/ACK` 链路在空间上同步收缩，碰撞次数显著增大到 `1547`，最终使 `PDR` 下降到 `67.77%`。因此，网络层攻击更适合体现“定点恶意转发破坏”，而物理层干扰更适合体现“区域性链路失效与拥塞放大”。

### 简洁结论

- 就投递率破坏而言，物理层干扰最强，`PDR` 从 `94.03%` 降至 `67.77%`。
- 在网络层攻击中，黑洞的破坏强于灰洞，联合攻击又强于单独黑洞。
- 从可视化效果看，黑洞/灰洞更容易展示“恶意节点位置与丢包行为”，物理层干扰更容易展示“区域性链路塌缩与冲突激增”。
