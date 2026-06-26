import logging
import os
from utils.ieee_802_11 import IeeeStandard

IEEE_802_11 = IeeeStandard().b_802_11


def _get_env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() not in {"0", "false", "no", "off"}


def _get_env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default

    return int(value)


def _get_env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default

    return float(value)


def _get_env_str(name, default):
    value = os.getenv(name)
    if value is None:
        return default

    return value


def _get_env_csv(name, default=""):
    value = os.getenv(name)
    if value is None:
        value = default

    return tuple(item.strip() for item in value.split(",") if item.strip())

# --------------------- simulation parameters --------------------- #
MAP_LENGTH = 600  # m, length of the map
MAP_WIDTH = 600  # m, width of the map
MAP_HEIGHT = 100  # m, height of the map
SIM_TIME = _get_env_int('UAVNETSIM_SIM_TIME_US', int(30 * 1e6))  # us, total simulation time
NUMBER_OF_DRONES = _get_env_int('UAVNETSIM_NUM_DRONES', 10)  # number of drones in the network
GRID_RESOLUTION = 20  # grid the map for path planning
STATIC_CASE = 0  # whether to simulate a static network
HETEROGENEOUS = 0  # heterogeneous network support (in terms of speed)
LOGGING_LEVEL = logging.INFO  # whether to print the detail information during simulation
SIMULATION_SEED = _get_env_int('UAVNETSIM_SEED', 2025)
ROUTING_PROTOCOL = _get_env_str('UAVNETSIM_ROUTING_PROTOCOL', 'DSDV').upper()
ENABLE_VISUALIZATION = _get_env_bool('UAVNETSIM_ENABLE_VISUALIZATION', True)
VIS_FRAME_INTERVAL = _get_env_int('UAVNETSIM_VIS_FRAME_INTERVAL_US', 20000)
ENABLE_INTERACTIVE_VISUALIZATION = _get_env_bool('UAVNETSIM_ENABLE_INTERACTIVE_VISUALIZATION', True)
VIS_OUTPUT_BASENAME = _get_env_str('UAVNETSIM_VIS_OUTPUT_BASENAME', 'uav_network_simulation')
LOG_FILE = _get_env_str('UAVNETSIM_LOG_FILE', 'running_log.log')
DATASET_EXPORT_ENABLED = _get_env_bool('UAVNETSIM_DATASET_EXPORT_ENABLED', True)
DATASET_EXPORT_DIR = _get_env_str('UAVNETSIM_DATASET_EXPORT_DIR', 'results')
DATASET_EXPORT_BASENAME = _get_env_str('UAVNETSIM_DATASET_EXPORT_BASENAME', 'routing_training_dataset')
ATTACK_NAMES = tuple(item.upper() for item in _get_env_csv('UAVNETSIM_ATTACKS', ''))
ATTACKER_IDS = tuple(int(item) for item in _get_env_csv('UAVNETSIM_ATTACKER_IDS', ''))
ATTACK_START_US = _get_env_int('UAVNETSIM_ATTACK_START_US', 0)
ATTACK_END_US = _get_env_int('UAVNETSIM_ATTACK_END_US', SIM_TIME)
ATTACK_PROBABILITY = _get_env_float('UAVNETSIM_ATTACK_PROBABILITY', 1.0)
BLACKHOLE_ATTACKER_IDS = tuple(int(item) for item in _get_env_csv('UAVNETSIM_BLACKHOLE_ATTACKER_IDS', ''))
GRAYHOLE_ATTACKER_IDS = tuple(int(item) for item in _get_env_csv('UAVNETSIM_GRAYHOLE_ATTACKER_IDS', ''))
BLACKHOLE_DROP_PROBABILITY = _get_env_float('UAVNETSIM_BLACKHOLE_DROP_PROBABILITY', ATTACK_PROBABILITY)
GRAYHOLE_DROP_PROBABILITY = _get_env_float('UAVNETSIM_GRAYHOLE_DROP_PROBABILITY', ATTACK_PROBABILITY)
ACK_POISON_MODE = _get_env_str('UAVNETSIM_ACK_POISON_MODE', 'OPTIMISTIC').upper()
LOCATION_SPOOF_RATIO = _get_env_float('UAVNETSIM_LOCATION_SPOOF_RATIO', 0.8)
PHY_JAMMER_START_US = _get_env_int('UAVNETSIM_PHY_JAMMER_START_US', int(10 * 1e6))
PHY_JAMMER_END_US = _get_env_int('UAVNETSIM_PHY_JAMMER_END_US', int(18 * 1e6))
PHY_JAMMER_POWER_W = _get_env_float('UAVNETSIM_PHY_JAMMER_POWER_W', 0.12)
PHY_JAMMER_RADIUS_M = _get_env_float('UAVNETSIM_PHY_JAMMER_RADIUS_M', 120.0)
PHY_JAMMER_COORDS = tuple(float(item) for item in _get_env_csv('UAVNETSIM_PHY_JAMMER_COORDS', ''))

# ---------- hardware parameters of drone (rotary-wing) -----------#
PROFILE_DRAG_COEFFICIENT = 0.012
AIR_DENSITY = 1.225  # kg/m^3
ROTOR_SOLIDITY = 0.05  # defined as the ratio of the total blade area to disc area
ROTOR_DISC_AREA = 0.79  # m^2
BLADE_ANGULAR_VELOCITY = 400  # radians/second
ROTOR_RADIUS = 0.5  # m
INCREMENTAL_CORRECTION_FACTOR = 0.1
AIRCRAFT_WEIGHT = 100  # Newton
ROTOR_BLADE_TIP_SPEED = 500
MEAN_ROTOR_VELOCITY = 7.2  # mean rotor induced velocity in hover
FUSELAGE_DRAG_RATIO = 0.3
INITIAL_ENERGY = 20 * 1e3  # in joule
ENERGY_THRESHOLD = 2000  # in joule
MAX_QUEUE_SIZE = 200  # maximum size of drone's queue

# ----------------------- radio parameters ----------------------- #
TRANSMITTING_POWER = 0.1  # in Watt
LIGHT_SPEED = 3 * 1e8  # light speed (m/s)
CARRIER_FREQUENCY = IEEE_802_11['carrier_frequency']  # carrier frequency (Hz)
NOISE_POWER = 4 * 1e-11  # noise power (Watt)
RADIO_SWITCHING_TIME = 100  # us, the switching time of the transceiver mode
SNR_THRESHOLD = IEEE_802_11['snr_threshold']

# ---------------------- packet parameters ----------------------- #
VARIABLE_PAYLOAD_LENGTH = 0  # whether to consider random payload length of data packet
AVERAGE_PAYLOAD_LENGTH = 1024 * 8  # in bit, 1024 bytes
MAXIMUM_PAYLOAD_VARIATION = 1600  # in bit
MAX_TTL = NUMBER_OF_DRONES + 1  # maximum time-to-live value
PACKET_LIFETIME = 10 * 1e6  # 10s
IP_HEADER_LENGTH = 20 * 8  # header length in network layer, 20 byte
MAC_HEADER_LENGTH = 14 * 8  # header length in mac layer, 14 byte

# ---------------------- physical layer -------------------------- #
PATH_LOSS_EXPONENT = 2  # for large-scale fading
PLCP_PREAMBLE = 128 + 16  # including synchronization and SFD (start frame delimiter)
PLCP_HEADER = 8 + 8 + 16 + 16  # including signal, service, length and HEC (header error check)
PHY_HEADER_LENGTH = PLCP_PREAMBLE + PLCP_HEADER  # header length in physical layer, PLCP preamble + PLCP header

ACK_HEADER_LENGTH = 16 * 8  # header length of ACK packet, 16 byte
ACK_PACKET_LENGTH = ACK_HEADER_LENGTH + 14 * 8  # bit

HELLO_PACKET_PAYLOAD_LENGTH = 256  # bit
HELLO_PACKET_LENGTH = IP_HEADER_LENGTH + MAC_HEADER_LENGTH + PHY_HEADER_LENGTH + HELLO_PACKET_PAYLOAD_LENGTH

# define the range of "id" of different types of packets
"""
|--------------|--------------|--------------|--------------|--------------|
0            10000          20000          30000          40000    
|   data pkt   |   hello pkt  |    ack pkt   |    vf pkt    |   grad msg   |
"""
GL_ID_DATA_PACKET = 0
GL_ID_HELLO_PACKET = 10000
GL_ID_ACK_PACKET = 20000
GL_ID_VF_PACKET = 30000
GL_ID_GRAD_MESSAGE = 40000

# ------------------ physical layer parameters ------------------- #
BIT_RATE = IEEE_802_11['bit_rate']
BIT_TRANSMISSION_TIME = 1/BIT_RATE * 1e6
BANDWIDTH = IEEE_802_11['bandwidth']
SENSING_RANGE = 750  # in meter, defines the area where a sending node can disturb a transmission from a third node

# --------------------- mac layer parameters --------------------- #
SLOT_DURATION = IEEE_802_11['slot_duration']
SIFS_DURATION = IEEE_802_11['SIFS']
DIFS_DURATION = SIFS_DURATION + (2 * SLOT_DURATION)
CW_MIN = 31  # initial contention window size
ACK_TIMEOUT = ACK_PACKET_LENGTH / BIT_RATE * 1e6 + SIFS_DURATION + 50  # maximum waiting time for ACK, in us
MAX_RETRANSMISSION_ATTEMPT = 5
