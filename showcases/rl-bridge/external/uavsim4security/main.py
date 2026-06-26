import simpy
from utils import config
from simulator.simulator import Simulator

"""
  _   _                   _   _          _     ____    _             
 | | | |   __ _  __   __ | \ | |   ___  | |_  / ___|  (_)  _ __ ___  
 | | | |  / _` | \ \ / / |  \| |  / _ \ | __| \___ \  | | | '_ ` _ \ 
 | |_| | | (_| |  \ V /  | |\  | |  __/ | |_   ___) | | | | | | | | |
  \___/   \__,_|   \_/   |_| \_|  \___|  \__| |____/  |_| |_| |_| |_|
                                                                                                                                                                                                                                                                                           
"""

if __name__ == "__main__":
    # Simulation setup
    env = simpy.Environment()
    channel_states = {i: simpy.Resource(env, capacity=1) for i in range(config.NUMBER_OF_DRONES)}
    sim = Simulator(seed=config.SIMULATION_SEED, env=env, channel_states=channel_states, n_drones=config.NUMBER_OF_DRONES)

    print('Routing protocol:', config.ROUTING_PROTOCOL)
    print('Simulation seed:', config.SIMULATION_SEED)
    if config.ATTACK_NAMES:
        print('Security attacks:', ', '.join(config.ATTACK_NAMES))
        print('Security attackers:', list(config.ATTACKER_IDS))
        print('Security attack probability:', config.ATTACK_PROBABILITY)
        if 'BLACKHOLE' in config.ATTACK_NAMES:
            print('Security blackhole attackers:', list(config.BLACKHOLE_ATTACKER_IDS or config.ATTACKER_IDS))
            print('Security blackhole drop probability:', config.BLACKHOLE_DROP_PROBABILITY)
        if 'GRAYHOLE' in config.ATTACK_NAMES:
            print('Security grayhole attackers:', list(config.GRAYHOLE_ATTACKER_IDS or config.ATTACKER_IDS))
            print('Security grayhole drop probability:', config.GRAYHOLE_DROP_PROBABILITY)
        print('Security attack window(us):', (config.ATTACK_START_US, config.ATTACK_END_US))
        print('Security ACK poison mode:', config.ACK_POISON_MODE)
        print('Security location spoof ratio:', config.LOCATION_SPOOF_RATIO)
        print('Security PHY jammer power(W):', config.PHY_JAMMER_POWER_W)
        print('Security PHY jammer radius(m):', config.PHY_JAMMER_RADIUS_M)
        print('Security PHY jammer window(us):', (config.PHY_JAMMER_START_US, config.PHY_JAMMER_END_US))
        if config.PHY_JAMMER_COORDS:
            print('Security PHY jammer coords:', config.PHY_JAMMER_COORDS)
    else:
        print('Security attacks: NONE')

    visualizer = None
    if config.ENABLE_VISUALIZATION:
        from visualization.visualizer import SimulationVisualizer

        visualizer = SimulationVisualizer(sim, output_dir=".", vis_frame_interval=config.VIS_FRAME_INTERVAL)
        visualizer.run_visualization()

    # Run simulation
    env.run(until=config.SIM_TIME)

    if visualizer is not None:
        visualizer.finalize()
