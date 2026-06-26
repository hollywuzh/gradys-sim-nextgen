# Joint Attack Matrix

| Protocol | Probability | Attackers | Scenario | PDR | Delay (ms) | Routing Load | Throughput (Kbps) | PDR Delta | Delay Delta (ms) | Security Metrics |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| QROUTING | 0.5 | 1,3 | baseline | 36.89320388349515 | 385.14513157894737 | 1.0526315789473684 | 97.98387401045684 | 0.0 | 0.0 | none |
| QROUTING | 0.5 | 1,3 | ack_poison | 27.184466019417474 | 401.6420714285714 | 1.4285714285714286 | 155.94926540781674 | -9.708737864077673 | 16.496939849624027 | ack_poison_count=61, ack_poison_qrouting_count=61 |
| QROUTING | 0.5 | 1,3 | grayhole | 46.601941747572816 | 160.2433125 | 0.8333333333333334 | 305.8422782981573 | 9.70873786407767 | -224.90181907894737 | grayhole_drop_count=32 |
| QROUTING | 0.5 | 1,3 | ack_poison_grayhole | 39.80582524271845 | 165.47651219512196 | 0.975609756097561 | 300.55998507279884 | 2.9126213592233015 | -219.66861938382542 | ack_poison_count=61, ack_poison_qrouting_count=61, grayhole_drop_count=45 |
| QMR | 0.5 | 1,3 | baseline | 93.20388349514563 | 47.586135416666664 | 0.4166666666666667 | 506.2760502917829 | 0.0 | 0.0 | none |
| QMR | 0.5 | 1,3 | ack_poison | 92.23300970873787 | 63.750778947368424 | 0.42105263157894735 | 434.78172479294847 | -0.9708737864077648 | 16.16464353070176 | ack_poison_count=56, ack_poison_qmr_count=56 |
| QMR | 0.5 | 1,3 | grayhole | 77.66990291262135 | 24.666275000000002 | 0.5 | 700.0565554220792 | -15.53398058252428 | -22.919860416666662 | grayhole_drop_count=17 |
| QMR | 0.5 | 1,3 | ack_poison_grayhole | 71.84466019417476 | 26.91436486486486 | 0.5405405405405406 | 658.5396552015195 | -21.359223300970868 | -20.671770551801803 | ack_poison_count=48, ack_poison_qmr_count=48, grayhole_drop_count=25 |
