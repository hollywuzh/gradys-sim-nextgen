from routing.qmr import qmr_config

hello_interval = qmr_config.hello_interval
discount_factor_update_interval = qmr_config.discount_factor_update_interval
history_packet_life_time = qmr_config.history_packet_life_time
communication_range = qmr_config.communication_range
which_exploration_mechanism = qmr_config.which_exploration_mechanism
eps_decay = qmr_config.eps_decay

forwarding_watchdog_timeout = int(0.4 * 1e6)
watchdog_check_interval = int(0.02 * 1e6)

initial_trust = 0.8
min_trust = 0.05
max_trust = 1.0
success_reward_step = 0.03
failure_penalty_step = 0.25
