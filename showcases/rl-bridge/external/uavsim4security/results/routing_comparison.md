# Routing Comparison

| Metric | DSDV | GRAd | Greedy | OPAR | QGeo | Q-routing | qfanet | qmr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generated_packets | 754 | 754 | 754 | 754 | 754 | 754 | 754 | 754 |
| packet_delivery_ratio_percent | 93.89920424403184 | 86.47214854111405 | 90.98143236074272 | 98.40848806366049 | 87.79840848806366 | 69.76127320954907 | 95.22546419098144 | 94.03183023872678 |
| average_end_to_end_delay_ms | 25.038505649717514 | 1037.8764064417178 | 31.0485 | 18.547378706199463 | 94.38263444108762 | 1941.1444429657795 | 40.13442339832869 | 26.000747531734838 |
| routing_load | 0.9180790960451978 | 2.2776073619631902 | 0.8746355685131195 | 0.0 | 0.9063444108761329 | 1.1406844106463878 | 0.8356545961002786 | 0.846262341325811 |
| average_throughput_kbps | 1043.3409170167306 | 790.0958913381 | 1073.8263089357347 | 1086.763585369632 | 627.5759594333874 | 98.16909348028265 | 767.9883425980594 | 638.4265176238816 |
| average_hop_count | 1.6454802259887005 | 1.7024539877300613 | 1.587463556851312 | 1.6644204851752022 | 2.392749244712991 | 3.5019011406844105 | 2.2242339832869082 | 2.447108603667137 |
| collision_count | 40 | 280 | 40 | 0 | 4 | 74 | 42 | 39 |
| average_mac_delay_ms | 5.7339377049180325 | nan | 6.603241796200345 | 4.3484923201293455 | 41.23393564582382 | 258.0928791903858 | 4.528774907749077 | 8.684430352303522 |

## Summary

- PDR delta (GRAd - DSDV): -7.4271 %
- PDR delta (Greedy - DSDV): -2.9178 %
- PDR delta (OPAR - DSDV): 4.5093 %
- PDR delta (QGeo - DSDV): -6.1008 %
- PDR delta (Q-routing - DSDV): -24.1379 %
- PDR delta (qfanet - DSDV): 1.3263 %
- PDR delta (qmr - DSDV): 0.1326 %
- Delay delta (GRAd - DSDV): 1012.8379 ms
- Delay delta (Greedy - DSDV): 6.0100 ms
- Delay delta (OPAR - DSDV): -6.4911 ms
- Delay delta (QGeo - DSDV): 69.3441 ms
- Delay delta (Q-routing - DSDV): 1916.1059 ms
- Delay delta (qfanet - DSDV): 15.0959 ms
- Delay delta (qmr - DSDV): 0.9622 ms
- Throughput delta (GRAd - DSDV): -253.2450 Kbps
- Throughput delta (Greedy - DSDV): 30.4854 Kbps
- Throughput delta (OPAR - DSDV): 43.4227 Kbps
- Throughput delta (QGeo - DSDV): -415.7650 Kbps
- Throughput delta (Q-routing - DSDV): -945.1718 Kbps
- Throughput delta (qfanet - DSDV): -275.3526 Kbps
- Throughput delta (qmr - DSDV): -404.9144 Kbps
- Routing load delta (GRAd - DSDV): 1.3595
- Routing load delta (Greedy - DSDV): -0.0434
- Routing load delta (OPAR - DSDV): -0.9181
- Routing load delta (QGeo - DSDV): -0.0117
- Routing load delta (Q-routing - DSDV): 0.2226
- Routing load delta (qfanet - DSDV): -0.0824
- Routing load delta (qmr - DSDV): -0.0718
- Collision delta (GRAd - DSDV): 240
- Collision delta (Greedy - DSDV): 0
- Collision delta (OPAR - DSDV): -40
- Collision delta (QGeo - DSDV): -36
- Collision delta (Q-routing - DSDV): 34
- Collision delta (qfanet - DSDV): 2
- Collision delta (qmr - DSDV): -1
