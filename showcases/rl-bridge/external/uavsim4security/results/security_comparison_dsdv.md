# Security Comparison for DSDV

| Scenario | PDR | Delay (ms) | Routing Load | Throughput (Kbps) | Hop Count | Collisions | MAC Delay (ms) | Security Metrics |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 100.0 | 49.30945833333334 | 0.4166666666666667 | 948.8009002966736 | 1.7458333333333333 | 20 | 4.429205250596659 | none |
| blackhole | 93.75 | 51.273315555555556 | 0.4444444444444444 | 984.7881340632556 | 1.6755555555555555 | 20 | 4.349376262626262 | blackhole_drop_count=15 |
| ack_spoof | 100.0 | 49.30945833333334 | 0.4166666666666667 | 948.8009002966736 | 1.7458333333333333 | 20 | 4.615219570405728 | ack_spoof_real_ack_suppressed_count=419, ack_spoof_success_count=419 |
