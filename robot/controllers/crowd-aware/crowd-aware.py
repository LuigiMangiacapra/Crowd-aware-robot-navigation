#!/usr/bin/env python3

import os
import math
import torch
import numpy as np
import random
# import time
import yaml
import torch.nn.functional as F

from controller import Supervisor
from collections import deque

import TwoStream_RNN as RNN
import DeepQNetwork as DQN

# =========================================================
# MODALITÀ
# =========================================================
MODE = "finetune"   # "train" | "finetune" | "run"
MODEL_PATH = "crowd_model.pt"


# =========================================================
# DEBUG
# =========================================================
DEBUG_PRINT_EVERY = 30


# =========================================================
# COSTANTI
# =========================================================
MACRO_STEP = 6  # numero di step Webots per ogni azione del robot
MAX_SPEED = 6
UNUSED_POINT = 83
N_SECTOR = 5
ROBOT_RADIUS = 0.35

DANGER_DISTANCE = 1.5  # distanza minima per un ostacolo pericoloso
DANGER_LATERAL_DISTANCE = 0.3  # distanza per left e right
MIN_LATERAL_DISTANCE = 0.5

DANGER_DISTANCE_PEDESTRIAN = 0.7
MAX_DISTANCE_PEDESTRIAN = 3.5

MAX_DISTANCE = 10
MAX_LATERAL_DISTANCE = 2.5  # distanza massima considerata per normalizzazione
CRUISING_SPEED = 2.0    # era 3.0
TURN_SPEED     = 1.5    # era 2.0

# crowd parameters
NEAR_OBSTACLE_THRESHOLD = 0.3
FAR_OBSTACLE_THRESHOLD = 0.3

# robot parameters
WHEEL_RADIUS = 0.0985      # metri
WHEEL_BASE = 0.404         # metri distanza tra ruote

# Distanza minima dal goal
GOAL_THRESHOLD = 0.4


# REWARDS
NEUTRAL_PENALTY = -0.05

# TIme
TIME_PENALTY = -0.02

# Goal
PROGRESS_GAIN = -0.02
GOAL_REWARD = 10
TRACKING_GOAL_REWARD = -0.05
TRACKING_PROGRESS = -0.02

# Collision
COLLISION_PEDESTRIAN_PENALTY = -10
COLLISION_PENALTY = -10
NEAREST_PENALTY = -3
NEAR_PENALTY = -1

# Overcome
OVERCOME_REWARD = 0

# Regression
REGRESS_PENALTY = 0.5

ACTION_REPEAT = 6


# =========================================================
# LOG HELPER — scrive su file E stampa a schermo
# =========================================================

def log_line(msg):
    """Solo a schermo, mai su file."""
    print(msg)

def flush_log():
    pass  # non serve più

# =========================================================
# UTILS
# =========================================================
def check_speed(speed):
    return max(min(speed, MAX_SPEED), -MAX_SPEED)


def apply_action(left_speed, right_speed):
    left_wheel.setVelocity(left_speed)
    right_wheel.setVelocity(right_speed)


# =========================================================
# ROBOT (AGENT) SETUP
# =========================================================
robot = Supervisor()

# tempo di ogni step (32 ms)
# Nel paper ogni step è 0.2 s
time_step = int(robot.getBasicTimeStep())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

robot.step(time_step)

# Recupero del nodo robot
robot_node = robot.getSelf()
translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

# salva posa iniziale del robot per reset episodio
initial_translation = translation_field.getSFVec3f()
initial_rotation = rotation_field.getSFRotation()


class Agent:
    def __init__(self, config_path, set_name):

        with open(config_path, "r") as f:
            all_params = yaml.safe_load(f)

        if set_name not in all_params:
            raise ValueError(f"Set '{set_name}' non trovato nel config. Disponibili: {list(all_params.keys())}")

        hp = all_params[set_name]

        # =========================
        # CORE RL PARAMS
        # =========================
        self.memory_maxlen = hp["memory_maxlen"]
        self.batch_size = hp["batch_size"]
        self.discount_factor = hp["discount_factor"]
        self.epsilon_steps = hp["epsilon_steps"]
        self.min_epsilon = hp["min_epsilon"]
        self.n_episodes = hp["n_episodes"]
        self.learning_rate_dqn = float(hp["learning_rate_dqn"])
        self.learning_rate_rnn = float(hp["learning_rate_rnn"])

        # =========================
        # ARCHITECTURE / TRAINING
        # =========================
        self.n_pref_samples = hp["n_pref_samples"]
        self.target_update_freq = hp["target_update_freq"]
        self.steps_len = hp["steps_len"]
        self.max_timestep = hp["max_timestep"]

        # =========================
        # DEFAULT / OPTIONALS
        # =========================
        self.epsilon = hp.get("epsilon", 1.0)
        self.environment = hp.get("environment", "webots")


# =========================================================
# PEDESTRIAN
# =========================================================
pedestrian_node = robot.getFromDef("PEDESTRIAN")

ped_translation_field = pedestrian_node.getField("translation")
ped_rotation_field = pedestrian_node.getField("rotation")

initial_ped_translation = ped_translation_field.getSFVec3f()
initial_ped_rotation = ped_rotation_field.getSFRotation()


# =========================================================
# DISPOSITIVI
# =========================================================
lidar = robot.getDevice("Hokuyo URG-04LX-UG01")
lidar.enable(time_step)

left_wheel = robot.getDevice("wheel_left_joint")
right_wheel = robot.getDevice("wheel_right_joint")

left_wheel.setPosition(float('inf'))
right_wheel.setPosition(float('inf'))
left_wheel.setVelocity(0.0)
right_wheel.setVelocity(0.0)


# =========================================================
# POSIZIONE BRACCIO
# =========================================================
arm_positions = [0.07, -1.0, 1.0, 1.5, 1.0, -1.39, 0.0]

for i in range(1, 8):
    motor = robot.getDevice(f"arm_{i}_joint")
    if motor:
        motor.setPosition(
            max(min(arm_positions[i - 1], motor.getMaxPosition()),
                motor.getMinPosition())
        )


# =========================================================
# LIDAR SETUP
# =========================================================
urg04lx_width = lidar.getHorizontalResolution()
max_range = lidar.getMaxRange()
range_threshold = max_range / 2.0

sector_size = int((urg04lx_width - 2 * UNUSED_POINT - 1) / N_SECTOR)
sector_range = [UNUSED_POINT + (i + 1) * sector_size for i in range(N_SECTOR)]

previous_ranges = None
step_counter = 0

LIDAR_STATE_DIM = 20

# =========================================================
# CREAZIONE RETI
# =========================================================
rnn = RNN.CrowdNavNet(
    spatial_dim=LIDAR_STATE_DIM,   # 20
    temporal_dim=LIDAR_STATE_DIM,  # 20  (prima era 7)
    goal_dim=6,
    human_pref_dim=4,
    debug=False
).to(device)

policy_dqn = DQN.DQN(
    input_dim=196,
    hidden_dim=128,
    n_actions=6,
    n_objectives=4,
    debug=False
).to(device)

target_dqn = DQN.DQN(
    input_dim=196,
    hidden_dim=128,
    n_actions=6,
    n_objectives=4,
    debug=False
).to(device)


# =========================================================
# SCELTA AZIONE
# =========================================================
def select_action(state, preference, episode, step_counter, epsilon=0.1, policy_dqn=None):

    debug_print(f"Selezione azione con epsilon={epsilon:.2f}", step_counter, episode)

    if torch.rand(1).item() < epsilon:
        robot_action = torch.randint(0, 6, (1,))
        return robot_action.item()

    else:
        with torch.no_grad():
            q_values = policy_dqn(state)
            debug_print(f"\n[DQN] Q-values (raw):\n{q_values.detach().cpu().numpy()}", step_counter, episode)

            pref = torch.tensor(preference, dtype=torch.float32).unsqueeze(0).to(device)
            q_scalar = scalarize(q_values, pref)

            debug_print(f"[DQN] Q-values (scalarized):\n{q_scalar.detach().cpu().numpy()}", step_counter, episode)

            robot_action = torch.argmax(q_scalar, dim=1)
            return robot_action.item()


def action_to_speeds(action, factor, factor_frontal):

    if action == 0:      # LEFT
        return -TURN_SPEED, TURN_SPEED
    elif action == 1:    # FRONT LEFT
        return CRUISING_SPEED * factor, CRUISING_SPEED
    elif action == 2:    # FRONT
        return CRUISING_SPEED, CRUISING_SPEED
    elif action == 3:    # FRONT RIGHT
        return CRUISING_SPEED, CRUISING_SPEED * factor
    elif action == 4:    # RIGHT
        return TURN_SPEED, -TURN_SPEED
    elif action == 5:    # STOP
        return 0.0, 0.0


def stabilize_robot():
    pos = translation_field.getSFVec3f()
    if pos[2] > 0.3 or any(math.isnan(v) for v in pos):
        translation_field.setSFVec3f([pos[0], pos[1], 0.095])
        robot_node.resetPhysics()


def stop_pedestrian():
    pos = ped_translation_field.getSFVec3f()
    rot = ped_rotation_field.getSFRotation()
    ped_translation_field.setSFVec3f(pos)
    ped_rotation_field.setSFRotation(rot)
    pedestrian_node.resetPhysics()


def check_lidar(ranges, previous_ranges, spatial_info, temporal_info):

    for i in range(UNUSED_POINT, urg04lx_width - UNUSED_POINT - 1):

        if math.isinf(ranges[i]) or math.isinf(previous_ranges[i]):
            continue
        if ranges[i] >= range_threshold:
            continue

        spatial = 1.0 - ranges[i] / max_range
        delta = previous_ranges[i] - ranges[i]
        radial_velocity = delta / (time_step / 1000.0)

        if i < sector_range[0]:
            idx = 0
        elif i < sector_range[1]:
            idx = 1
        elif i < sector_range[2]:
            idx = 2
        elif i < sector_range[3]:
            idx = 3
        else:
            idx = 4

        spatial_info[idx] += spatial
        temporal_info[idx] += radial_velocity

    return spatial_info, temporal_info


def reset_robot():
    translation_field.setSFVec3f(initial_translation)
    rotation_field.setSFRotation(initial_rotation)
    robot_node.resetPhysics()
    robot.simulationResetPhysics()
    apply_action(0.0, 0.0)


def reset_robot_randomized(episode):
    goal_x, goal_y = 1.0, 1.0
    x, y = -2.0, 0.0
    base_angle = math.atan2(goal_y - y, goal_x - x)

    if episode < 200:
        final_angle = base_angle
    else:
        final_angle = base_angle + np.random.uniform(-0.5, 0.5)

    z = 0.096
    translation_field.setSFVec3f([x, y, z])
    rotation_field.setSFRotation([0.0, 0.0, 1.0, final_angle])
    robot_node.resetPhysics()
    robot.simulationResetPhysics()
    apply_action(0.0, 0.0)

    spawn_id = f"x{x:.2f}_y{y:.2f}"
    return [x, y, z], spawn_id


def reset_pedestrian(episode, recent_collision_rate, recent_goal_rate):
    controller_args_field = pedestrian_node.getField("controllerArgs")

    if episode < 400:
        speed = 0.06
    elif recent_collision_rate > 0.40:   # era 0.45
        speed = 0.04
    elif recent_collision_rate > 0.25:   # era 0.30
        speed = 0.06
    elif recent_collision_rate > 0.15:   # era 0.20
        speed = 0.08
    else:
        speed = 0.12                     # era 0.10

    x = np.random.uniform(-1.5, -0.5)
    controller_args_field.setMFString(0, f"--trajectory={x:.2f} 3, {x:.2f} 0")
    controller_args_field.setMFString(1, f"--speed={speed:.3f}")

    ped_translation_field.setSFVec3f(initial_ped_translation)
    ped_rotation_field.setSFRotation(initial_ped_rotation)
    pedestrian_node.resetPhysics()
    pedestrian_node.restartController()

    # ✅ ORA SCRIVE SUL LOG
    log_line(f"[PEDESTRIAN RESET] episode={episode}, speed={speed:.3f}, coll_rate={recent_collision_rate:.0%}")
    return speed


def detect_collision_lidar(ranges, factor, factor_frontal, ped_pos=None, robot_pos=None, theta=None):
    n = len(ranges)
    if n == 0:
        return False, float("inf"), float("inf"), factor, factor_frontal, False

    sector_size = n // 5

    left = ranges[0: sector_size]
    front_left = ranges[sector_size: 2 * sector_size]
    front = ranges[2 * sector_size: 3 * sector_size]
    front_right = ranges[3 * sector_size: 4 * sector_size]
    right = ranges[4 * sector_size: n]

    def valid(values):
        return [v for v in values if not math.isinf(v) and v > 0.01]

    def min_valid(values):
        vals = valid(values)
        return min(vals) if vals else MAX_DISTANCE

    def mean_valid(values):
        vals = valid(values)
        return sum(vals) / len(vals) if vals else MAX_DISTANCE

    d_left = min_valid(left)
    d_front_left = mean_valid(front_left)
    d_front = mean_valid(front)
    d_front_right = mean_valid(front_right)
    d_right = min_valid(right)

    min_dist = min(d_front_left, d_front, d_front_right)
    min_lateral = min(d_left, d_right)

    CRITICAL = 1.0
    WARNING  = 2.0

    collision = False
    near_obstacle = False

    if min_dist < CRITICAL:
        collision = True
        near_obstacle = True
        factor = 0.0
        factor_frontal = 0.0
    elif min_dist < WARNING:
        near_obstacle = True
        factor_frontal = np.clip((min_dist - CRITICAL) / (WARNING - CRITICAL), 0.2, 1.0)
    else:
        factor_frontal = 1.0

    if ped_pos is not None and robot_pos is not None and theta is not None:
        ped_dx = ped_pos[0] - robot_pos[0]
        ped_dy = ped_pos[1] - robot_pos[1]
        ped_dist = math.sqrt(ped_dx**2 + ped_dy**2)

        angle_to_ped = math.atan2(ped_dy, ped_dx) - theta
        angle_to_ped = math.atan2(math.sin(angle_to_ped), math.cos(angle_to_ped))

        ped_is_frontal = abs(angle_to_ped) < math.radians(30) and ped_dist < 2.0

        if ped_is_frontal:
            # Non sovrascrivere collision: se c'è un muro, rimane collision
            # Ridurre solo il rallentamento per non bloccarsi davanti al pedone
            factor = max(factor, 0.5)
            factor_frontal = max(factor_frontal, 0.5)

    return collision, min_dist, min_lateral, factor, factor_frontal, near_obstacle


def detect_goal(goal_distance):
    goal_reached = goal_distance < GOAL_THRESHOLD
    return goal_reached, goal_distance


def preference_function(reward_vector, base_reward, preference, n_pref):
    for i in range(n_pref):
        reward_vector[i] = base_reward * preference[i]
    return reward_vector


def normalize(value, min_val, max_val):
    return 2.0 * (value - min_val) / (max_val - min_val) - 1.0


def get_reward_paper_like(
    progress, goal_reached, collision,
    dist, ped_distance, ped_collision,
    goal_distance, prev_goal_dist,
    robot_pos, path_start_pos, goal_pos,
    angle_error, episode, ped_pos, theta, robot_action,
    prev_ped_distance=None    # ← nuovo parametro
):
    DP_MIN = 0.7
    DP_MAX = 2.0

    dx_ped = ped_pos[0] - robot_pos[0]
    dy_ped = ped_pos[1] - robot_pos[1]
    angle_to_ped = math.atan2(dy_ped, dx_ped) - theta
    angle_to_ped = math.atan2(math.sin(angle_to_ped), math.cos(angle_to_ped))
    frontal_factor = max(0.0, math.cos(angle_to_ped))

    # Velocità di avvicinamento: positiva = ci stiamo avvicinando
    if prev_ped_distance is not None:
        closing_speed = prev_ped_distance - ped_distance  # >0 se ci avviciniamo
    else:
        closing_speed = 0.0

    if ped_distance < DP_MIN or ped_collision:
        r_ped = -1.0
    elif ped_distance < DP_MAX:
        raw = -0.1 * (DP_MAX - ped_distance) / DP_MAX
        r_ped = raw * (0.3 + 0.7 * frontal_factor)

        # Penalità aggiuntiva se ci stiamo avvicinando e il pedone è frontale
        if closing_speed > 0.05:
            r_ped -= 0.15 * closing_speed * frontal_factor

        r_ped = float(np.clip(r_ped, -0.5, 0.0))
    else:
        r_ped = 0.0

    # === OSTACOLI STATICI — penalità progressiva invece di binaria ===
    # Soglia critica reale: 0.5m non 2.0m
    if collision or dist < 0.3:
        r_static = -1.0
    elif dist < 0.6:
        r_static = -0.5
    elif dist < 1.0:
        r_static = -0.1 * (1.0 - dist)  # massimo -0.1 vicino a 1m
    else:
        r_static = 0.0

    # === GOAL ===
    if goal_reached:
        r_goal = 1.0
    else:
        delta_dg = prev_goal_dist - goal_distance
        r_goal = np.clip(0.3 * delta_dg - 0.003, -0.2, 0.3)  # coefficiente 0.1 → 0.3
        r_goal += 0.08 * math.cos(angle_error)                 # peso orientamento aumentato

        if robot_action == 5:
            # Pedone lontano o già passato: stop ingiustificato
            if ped_distance > 1.5:
                r_goal -= 0.15
            # Pedone di lato (non frontale): stop comunque ingiustificato
            elif frontal_factor < 0.3 and ped_distance > 0.9:
                r_goal -= 0.10

        r_goal = float(np.clip(r_goal, -0.3, 0.35))

    # PATH TRACKING — disabilitato nei primi 500 episodi
    if episode < 2000:
        r_path = 0.0
    else:
        cte = path_tracking(robot_pos, path_start_pos, goal_pos)
        r_path = -0.02 * (1.0 - math.exp(-cte / 2.0))
        r_path = float(np.clip(r_path, -0.05, 0.0))

    reward = np.array([r_goal, r_static, r_ped, r_path], dtype=np.float32)
    return torch.tensor(reward, dtype=torch.float32, device=device)


# def get_reward(
#     progress, goal_reached, collision, near_obstacle,
#     dist, lateral,
#     ped_distance, goal_distance,
#     robot_pos, ped_pos, theta, goal_pos,
#     episode, step_counter, path_start_pos,
#     angle_error,
#     prev_dist,
#     prev_ped_dist,
#     ped_collision, robot_action
# ):
#     # =========================
#     # 1. GOAL / PROGRESS
#     # =========================
#     reward_goal = 6.0 * progress
#     reward_goal += 1.5 * np.cos(angle_error)

#     if robot_action == 5:
#         reward_goal -= 0.8
#         if ped_distance is not None and ped_distance > 3.0:
#             reward_goal -= 0.5

#     if progress > 0.05:
#         reward_goal += 0.3

#     if goal_reached:
#         reward_goal = 10.0

#     reward_goal = np.clip(reward_goal, -10.0, 10.0)

#     # =========================
#     # 2. OSTACOLI STATICI
#     # =========================
#     ped_dx = ped_pos[0] - robot_pos[0]
#     ped_dy = ped_pos[1] - robot_pos[1]
#     ped_dist_actual = math.sqrt(ped_dx**2 + ped_dy**2)

#     if ped_dist_actual < 3.0 and abs(dist - ped_dist_actual) < 1.0:
#         effective_dist = lateral
#     else:
#         effective_dist = dist

#     if effective_dist < 3.0:
#         reward_safety_object = -2.5 * np.exp(-effective_dist / 1.2)
#     else:
#         reward_safety_object = -0.05

#     if prev_dist is not None:
#         delta = effective_dist - prev_dist
#         reward_safety_object += 0.5 * np.tanh(delta)

#     if collision:
#         reward_safety_object -= 10.0

#     reward_safety_object = np.clip(reward_safety_object, -8.0, 0.0)

#     if lateral < 1.5:
#         reward_lateral = -0.8 * np.exp(-lateral / 0.6)
#     else:
#         reward_lateral = 0.0

#     # =========================
#     # 3. PEDONE / OVERTAKING
#     # =========================
#     reward_safety_pedestrian = 0.0

#     if ped_distance is not None:
#         reward_safety_pedestrian = -0.6 * np.exp(-ped_distance / 2.5)

#         if prev_ped_dist is not None:
#             delta_ped = ped_distance - prev_ped_dist
#             reward_safety_pedestrian += 0.2 * np.tanh(delta_ped)

#         if ped_collision:
#             reward_safety_pedestrian -= 2.0

#         reward_safety_pedestrian = np.clip(reward_safety_pedestrian, -1.0, 0.0)

#     # =========================
#     # 4. PATH TRACKING
#     # =========================
#     if ped_distance is not None and ped_distance < 2.0:
#         cross_track_error = 0
#         reward_path = 0.0
#     else:
#         cross_track_error = path_tracking(robot_pos, path_start_pos, goal_pos)
#         reward_path = -0.3 * (1.0 - np.exp(-cross_track_error / 1.5))
#         reward_path = np.clip(reward_path, -1.0, 0.0)

#     # =========================
#     # 5. NORMALIZZAZIONE
#     # =========================
#     r_goal   = np.clip(reward_goal / 10.0, -1.0, 1.0)
#     r_safety = np.clip((reward_safety_object + reward_lateral) / 8.0, -1.0, 1.0)
#     r_ped    = np.clip(reward_safety_pedestrian / 1.0, -1.0, 1.0)
#     r_path   = np.clip(reward_path / 1.0, -1.0, 1.0)

#     reward = np.array([r_goal, r_safety, r_ped, r_path], dtype=np.float32)

#     return torch.tensor(reward, dtype=torch.float32, device=device), cross_track_error


def get_yaw_from_webots_rotation(rotation):
    x, y, z, angle = rotation
    if z < 0:
        angle = -angle
    return angle


def differential_drive_kinematics(x, y, theta, wl, wr, wheel_radius, wheel_base, dt):
    v = wheel_radius * (wr + wl) / 2.0
    omega = wheel_radius * (wr - wl) / wheel_base
    x_new = x + v * math.cos(theta) * dt
    y_new = y + v * math.sin(theta) * dt
    theta_new = theta + omega * dt
    return x_new, y_new, theta_new, v, omega


def compute_goal_metrics(robot_pos, theta, goal_pos):
    x, y = robot_pos[0], robot_pos[1]
    dx = goal_pos[0] - x
    dy = goal_pos[1] - y
    goal_distance = math.sqrt(dx**2 + dy**2)

    dx_goal = math.cos(theta) * dx + math.sin(theta) * dy
    dy_goal = -math.sin(theta) * dx + math.cos(theta) * dy

    if goal_distance > 1e-6:
        goal_direction = np.array([dx, dy]) / goal_distance
    else:
        goal_direction = np.array([0.0, 0.0])

    robot_heading = np.array([math.cos(theta), math.sin(theta)])

    dot = np.clip(np.dot(robot_heading, goal_direction), -1.0, 1.0)
    cross = robot_heading[0] * goal_direction[1] - robot_heading[1] * goal_direction[0]
    angle_error = math.atan2(cross, dot)

    return goal_distance, angle_error, dx_goal, dy_goal


def path_tracking(robot_pos, start_pos, goal_pos):
    x, y = robot_pos[0], robot_pos[1]
    x1, y1 = start_pos[0], start_pos[1]
    x2, y2 = goal_pos[0], goal_pos[1]

    num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    den = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    if den < 1e-6:
        return 0.0
    return num / den


def decay_epsilon(global_step, epsilon_steps, min_epsilon=0.05):
    if global_step <= epsilon_steps:
        epsilon = max(min_epsilon, np.exp(-global_step / epsilon_steps))
    else:
        epsilon = min_epsilon
    return epsilon


def read_lidar(lidar):
    return lidar.getRangeImage()[::-1]


def detect_pedestrian_collision(robot_pos, ped_pos, theta):
    dx = ped_pos[0] - robot_pos[0]
    dy = ped_pos[1] - robot_pos[1]
    ped_distance = math.sqrt(dx**2 + dy**2)
    ped_collision = ped_distance < DANGER_DISTANCE_PEDESTRIAN
    # print("ped_distance: ", ped_distance)  ← rimuovi questa riga
    return ped_collision, ped_distance


def analyze_environment(ranges, factor, factor_frontal, goal_distance, ped_pos, robot_pos, theta):
    collision, min_dist, min_lateral, factor, factor_frontal, near_obstacle = detect_collision_lidar(
        ranges, factor, factor_frontal, ped_pos, robot_pos, theta
    )
    goal_reached, _ = detect_goal(goal_distance)
    ped_collision, ped_distance = detect_pedestrian_collision(robot_pos, ped_pos, theta)
    done = collision or ped_collision or goal_reached

    return collision, goal_reached, ped_collision, ped_distance, near_obstacle, min_dist, min_lateral, factor, factor_frontal, done


# =========================================================
# NUOVI PARAMETRI LIDAR STATE
# =========================================================
N_LIDAR_SECTORS = 20          # quanti "bin" dividi il lidar (paper usa 200 raw, noi usiamo 20)
LIDAR_STATE_DIM = N_LIDAR_SECTORS  # = spatial_dim = temporal_dim nella rete


def extract_lidar_state(ranges, previous_ranges):
    """
    Estrae lo stato spaziale e temporale dal lidar grezzo,
    avvicinandosi all'approccio del paper (o_t e z_t = o_t - o_{t-1}).

    Args:
        ranges: lista di float, scan lidar corrente (già reversed)
        previous_ranges: scan lidar del passo precedente

    Returns:
        spatial: lista[float] len=N_LIDAR_SECTORS, valori in [0,1]
        temporal: lista[float] len=N_LIDAR_SECTORS, valori in [-1,1]
    """
    # Usa solo i punti validi (esclude UNUSED_POINT su ogni lato)
    valid_ranges = ranges[UNUSED_POINT: len(ranges) - UNUSED_POINT - 1]
    valid_prev   = previous_ranges[UNUSED_POINT: len(previous_ranges) - UNUSED_POINT - 1]

    n_valid = len(valid_ranges)
    bin_size = max(n_valid // N_LIDAR_SECTORS, 1)

    spatial  = []
    temporal = []

    for i in range(N_LIDAR_SECTORS):
        start = i * bin_size
        end   = min(start + bin_size, n_valid)

        # --- Spatial: distanza minima nel settore, normalizzata in [0,1] ---
        sector_curr = [
            v for v in valid_ranges[start:end]
            if not math.isinf(v) and not math.isnan(v)
        ]
        sector_prev = [
            v for v in valid_prev[start:end]
            if not math.isinf(v) and not math.isnan(v)
        ]

        if sector_curr:
            min_curr = min(sector_curr)
            # Normalizza rispetto al max_range: 0 = vicino, 1 = lontano
            s = np.clip(min_curr / max_range, 0.0, 1.0)
        else:
            s = 1.0  # nessun ostacolo rilevato → lontano

        # --- Temporal: differenza (z_t = o_t - o_{t-1}), normalizzata ---
        if sector_curr and sector_prev:
            min_prev = min(sector_prev)
            delta = (min_curr - min_prev) / max_range   # positivo = ostacolo si allontana
            t = np.clip(delta, -1.0, 1.0)
        else:
            t = 0.0

        spatial.append(float(s))
        temporal.append(float(t))

    return spatial, temporal


def extract_components(
    robot_pos, ped_pos, prev_ped_pos, prev_robot_pos,
    theta, prev_theta, time_step, delta_speed
):
    EPS = 1e-6
    MAX_REL_SPEED = 2.0
    MAX_TTC = 5.0

    dt = (MACRO_STEP * time_step) / 1000.0
    dt = max(dt, EPS)

    dx = ped_pos[0] - robot_pos[0]
    dy = ped_pos[1] - robot_pos[1]
    distance = math.sqrt(dx**2 + dy**2)

    forward_offset = math.cos(theta) * dx + math.sin(theta) * dy
    lateral_offset = -math.sin(theta) * dx + math.cos(theta) * dy

    raw_angle = math.atan2(dy, dx) - theta
    angle = math.atan2(math.sin(raw_angle), math.cos(raw_angle))

    ttc = MAX_TTC
    rel_vx_robot = 0.0
    rel_vy_robot = 0.0
    closing_speed = 0.0
    ped_heading_x = 0.0
    ped_heading_y = 0.0
    crossing_intent = 0.0

    if prev_ped_pos is not None and prev_robot_pos is not None:
        ped_vx = (ped_pos[0] - prev_ped_pos[0]) / dt
        ped_vy = (ped_pos[1] - prev_ped_pos[1]) / dt
        robot_vx = (robot_pos[0] - prev_robot_pos[0]) / dt
        robot_vy = (robot_pos[1] - prev_robot_pos[1]) / dt

        rel_vx = ped_vx - robot_vx
        rel_vy = ped_vy - robot_vy

        rel_vx_robot = math.cos(theta) * rel_vx + math.sin(theta) * rel_vy
        rel_vy_robot = -math.sin(theta) * rel_vx + math.cos(theta) * rel_vy

        closing_speed = -rel_vx_robot

        ped_speed = math.sqrt(ped_vx**2 + ped_vy**2)
        if ped_speed > EPS:
            ped_heading_x = ped_vx / ped_speed
            ped_heading_y = ped_vy / ped_speed
            crossing_intent = abs(rel_vy_robot) / (abs(rel_vx_robot) + abs(rel_vy_robot) + EPS)

        relative_speed = forward_offset / dt
        if relative_speed < -0.01:
            ttc = distance / (abs(relative_speed) + EPS)

    distance_norm       = np.clip(distance / MAX_DISTANCE, 0.0, 1.0)
    lateral_offset_norm = np.clip(lateral_offset / 4.0, -1.0, 1.0)
    forward_offset_norm = np.clip(forward_offset / MAX_DISTANCE, -1.0, 1.0)
    ttc_norm            = np.clip(ttc / MAX_TTC, 0.0, 1.0)
    closing_speed_norm  = np.clip(closing_speed / MAX_REL_SPEED, -1.0, 1.0)

    spatial = [
        distance_norm,
        math.cos(angle),
        math.sin(angle),
        lateral_offset_norm,
        forward_offset_norm,
        ttc_norm,
        crossing_intent
    ]

    temporal = [
        np.clip(rel_vx_robot / MAX_REL_SPEED, -1.0, 1.0),
        np.clip(rel_vy_robot / MAX_REL_SPEED, -1.0, 1.0),
        closing_speed_norm,
        ped_heading_x,
        ped_heading_y,
        np.clip(delta_speed[0], -1.0, 1.0),
        np.clip(delta_speed[1], -1.0, 1.0),
    ]

    return spatial, temporal


def memorize_checkpoint(models_dir, optimizer_value, optimizer_advantage, optimizer_rnn,
                        policy_dqn, target_dqn, rnn, global_step, episode):
    checkpoint_path = os.path.join(models_dir, "last_model.pt")
    loaded = False

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_dqn.load_state_dict(checkpoint["policy_dqn"])
        target_dqn.load_state_dict(checkpoint["target_dqn"])
        rnn.load_state_dict(checkpoint["rnn"])
        optimizer_value.load_state_dict(checkpoint["optimizer_value"])
        optimizer_advantage.load_state_dict(checkpoint["optimizer_advantage"])
        optimizer_rnn.load_state_dict(checkpoint["optimizer_rnn"])
        global_step = checkpoint["global_step"]
        start_episode = checkpoint["episode"] + 1
        goal_reached_count = checkpoint.get("goal_reached_count", 0)
        best_success_rate  = checkpoint.get("best_success_rate", 0.0)
        best_reward_mean   = checkpoint.get("best_reward_mean", -float('inf'))  # ← aggiunta
        best_balanced_score = checkpoint.get("best_balanced_score", 0.0)        # ← aggiunta
        loaded = True
        log_line(f"[RESUME] Caricato da ep {start_episode}, step {global_step}")
    else:
        start_episode = 1
        global_step = 0
        goal_reached_count = 0
        best_success_rate = 0.0
        best_reward_mean = -float('inf')   # ← mancava
        best_balanced_score = 0.0          # ← mancava
        log_line("[RESUME] Nessun checkpoint trovato — training da zero")

    return global_step, start_episode, goal_reached_count, best_success_rate, best_reward_mean, best_balanced_score, loaded

def check_finetune(mode, optimizer_value, optimizer_advantage, optimizer_rnn,
                   policy_dqn, target_dqn, rnn, global_step, episode, models_dir):
    if mode == "finetune":
        global_step, start_episode, goal_reached_count, best_success_rate, best_reward_mean, best_balanced_score, loaded = memorize_checkpoint(
            models_dir, optimizer_value, optimizer_advantage, optimizer_rnn,
            policy_dqn, target_dqn, rnn, global_step, episode
        )
    else:
        start_episode = 1
        global_step = 0
        goal_reached_count = 0
        best_success_rate = 0.0

    return global_step, start_episode, goal_reached_count, best_success_rate, best_reward_mean, best_balanced_score


def debug_print(msg, step_counter=None, episode=None):
    if step_counter is None:
        print(msg)
    else:
        if step_counter % DEBUG_PRINT_EVERY == 0:
            prefix = ""
            if episode is not None:
                prefix += f"[EP {episode}] "
            prefix += f"[STEP {step_counter}] "
            print(prefix + str(msg))


def scalarize(q, pref):
    pref = pref / (pref.sum(dim=1, keepdim=True) + 1e-8)
    return (q * pref.unsqueeze(1)).sum(dim=2)


def chebyshev_scalarize(q, pref, tau=0.1):
    weights = pref.unsqueeze(1)
    weighted = weights * q
    return tau * torch.logsumexp(weighted / tau, dim=2)


def sample_preferences(k=5):
    alpha = np.ones(4)
    prefs = np.random.dirichlet(alpha, size=k)
    return torch.tensor(prefs, dtype=torch.float32, device=device)


def encode_state(spatial_seq, temporal_seq, goal, pref, training=False):
    if training:
        return rnn(spatial_seq, temporal_seq, goal, pref)
    else:
        with torch.no_grad():
            return rnn(spatial_seq, temporal_seq, goal, pref)


def train_step(
    memory, agent, policy_dqn, target_dqn,
    optimizer_value, optimizer_advantage, optimizer_rnn,
    device, global_step, step_counter, episode, debug_print=None
):
    if len(memory) < agent.batch_size:
        return None

    batch = memory.sample(agent.batch_size)
    (b_spatial, b_temporal, b_goal, b_pref_rnn,
     b_next_spatial, b_next_temporal, b_next_goal,
     b_actions, b_rewards, b_dones, b_pref) = zip(*batch)

    b_spatial    = torch.cat(b_spatial).to(device)
    b_temporal   = torch.cat(b_temporal).to(device)
    b_goal       = torch.cat(b_goal).to(device)
    b_pref_rnn   = torch.cat(b_pref_rnn).to(device)

    b_next_spatial  = torch.cat(b_next_spatial).to(device)
    b_next_temporal = torch.cat(b_next_temporal).to(device)
    b_next_goal     = torch.cat(b_next_goal).to(device)

    b_actions = torch.tensor(b_actions, dtype=torch.long, device=device).unsqueeze(1)
    b_rewards = torch.stack([r.to(device) for r in b_rewards])
    b_dones   = torch.tensor(b_dones, dtype=torch.float32, device=device)
    b_pref    = torch.stack([
        torch.tensor(p, dtype=torch.float32) if not isinstance(p, torch.Tensor) else p
        for p in b_pref
    ]).to(device)

    B = agent.batch_size

    b_state  = encode_state(b_spatial, b_temporal, b_goal, b_pref_rnn, training=True)
    q_values = policy_dqn(b_state)                                   # [B, A, K]
    current_q = q_values[torch.arange(B), b_actions.squeeze(1)]

    with torch.no_grad():
        b_next_state = encode_state(b_next_spatial, b_next_temporal, b_next_goal, b_pref_rnn)

        pref_sample_batch = sample_preferences(k=agent.n_pref_samples)  # [M, 4]
        q_next_target = target_dqn(b_next_state)                        # [B, A, K]

        best_q_next = torch.zeros(B, 4, device=device)

        for pref_w in pref_sample_batch:
            pref_w_exp = pref_w.unsqueeze(0).expand(B, -1)
            q_scalar_w = scalarize(q_next_target, pref_w_exp)
            best_a = q_scalar_w.argmax(dim=1)
            q_vec = q_next_target[torch.arange(B), best_a]

            val_current = (q_vec * b_pref).sum(dim=1)
            val_best    = (best_q_next * b_pref).sum(dim=1)

            mask = (val_current > val_best).unsqueeze(1).expand_as(best_q_next)
            best_q_next = torch.where(mask, q_vec, best_q_next)

        target_q = b_rewards + agent.discount_factor * best_q_next * (1 - b_dones.unsqueeze(1))
        target_q = torch.clamp(target_q, -20.0, 20.0)

    loss = F.smooth_l1_loss(current_q, target_q)

    # ✅ ORA SCRIVE SUL LOG
    if step_counter % 50 == 0:
        log_line(f"\n[DEBUG LOSS] {loss.item():.6f}")

    optimizer_value.zero_grad()
    optimizer_advantage.zero_grad()
    optimizer_rnn.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(rnn.parameters(), 0.5)
    optimizer_value.step()
    optimizer_advantage.step()
    optimizer_rnn.step()

    return loss.item()


def robot_speak(preference, step_counter, print_every=30):
    if step_counter % print_every != 0:
        return
    pref = np.array(preference)
    idx = np.argmax(pref)
    messages = [
        "MI CONCENTRO SUL GOAL",
        "PRIORITÀ: EVITARE OSTACOLI",
        "ATTENZIONE AI PEDONI",
        "SEGUO LA TRAIETTORIA"
    ]
    strength = pref[idx]
    tone = "!!!" if strength > 0.6 else ("!!" if strength > 0.4 else "!")
    msg = messages[idx] + " " + tone
    print(f"[STEP {step_counter}] 🤖 {msg}")


def stop_all():
    apply_action(0.0, 0.0)
    controller_args_field = pedestrian_node.getField("controllerArgs")
    controller_args_field.setMFString(1, "--speed=0.0")
    ped_translation_field.setSFVec3f(initial_ped_translation)
    ped_rotation_field.setSFRotation(initial_ped_rotation)
    pedestrian_node.resetPhysics()
    pedestrian_node.restartController()
    flush_log()
    log_line("\n[TRAINING COMPLETATO] Robot e pedone fermati.")


def log_to_file(
    models_dir, episode, epsilon, loss, total_reward,
    success_rate, goal_rate, episode_steps,
    end_reasons, collision_rate, ped_collision_rate,
    goal_reached_count, mean_q, max_q, min_q,
    q_action_spread, entropy,
):
    log_path = os.path.join(models_dir, "training_log.txt")
    total_ep = sum(end_reasons.values())

    with open(log_path, "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"EP {episode} | "
                f"Success: {goal_reached_count}/{episode} ({goal_reached_count/episode:.1%}) | "
                f"Window: {success_rate:.1%}\n")
        f.write(f"Epsilon: {epsilon:.4f} | Loss: {loss:.6f} | Steps: {episode_steps}\n")
        f.write(f"Collision: {collision_rate:.1%} | "
                f"Ped: {ped_collision_rate:.1%} | "
                f"Goal: {goal_rate:.1%} | "
                f"Timeout: {(end_reasons['timeout']/total_ep):.1%}\n")
        f.write(f"Q mean/max/min: {mean_q:.3f} / {max_q:.3f} / {min_q:.3f} | "
                f"Spread: {q_action_spread:.4f} | Entropy: {entropy:.3f}\n")
        f.write(f"Reward: {np.round(total_reward, 3)}\n")
        f.write(f"{'='*50}\n")


agent = Agent("hyperparameters.yml", "thiago")


# =========================================================
# TRAINING
# =========================================================
def train(goal_pos, mode):

    global global_step, _log_path_global
    global_step = 0
    episode = 0
    goal_reached_count = 0
    start_episode = 1
    best_success_rate = 0.0
    best_reward_mean = -float('inf')  # ← aggiunta
    best_balanced_score = 0.0         # ← aggiunta
    recent_successes = deque(maxlen=20)
    recent_rewards = deque(maxlen=100)
    prev_theta = None
    decision_step = 0
    end_reasons = {"goal": 0, "ped_collision": 0, "collision": 0, "timeout": 0}
    action_counts = [0] * 6
    action_names = ["LEFT", "FRONT_LEFT", "FRONT", "FRONT_RIGHT", "RIGHT", "STOP"]
    ped_distance = float('inf')
    reward_done = False
    action_counts_diag = [0] * 6
    spawn_id = "unknown"
    mean_q = 0.0
    max_q = 0.0
    min_q = 0.0
    state_mean_last = None
    state_std_last = None
    q_exploit_scalarized_last = None
    q_exploit_action_last = None

    memory = DQN.ReplayMemory(agent.memory_maxlen)
    target_dqn.load_state_dict(policy_dqn.state_dict())

    optimizer_value = torch.optim.Adam(
        list(policy_dqn.trunk.parameters()) +
        list(policy_dqn.value_stream.parameters()),
        lr=agent.learning_rate_dqn
    )
    optimizer_advantage = torch.optim.Adam(
        policy_dqn.advantage_stream.parameters(),
        lr=agent.learning_rate_dqn * 2.0
    )
    optimizer_rnn = torch.optim.Adam(rnn.parameters(), lr=agent.learning_rate_rnn)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # ✅ imposta il path globale per log_line()
    _log_path_global = os.path.join(models_dir, "training_log.txt")

    global_step, start_episode, goal_reached_count, best_success_rate, best_reward_mean, best_balanced_score = check_finetune(
        mode, optimizer_value, optimizer_advantage, optimizer_rnn,
        policy_dqn, target_dqn, rnn, global_step, start_episode, models_dir
    )

    epsilon_check = decay_epsilon(global_step, agent.epsilon_steps, agent.min_epsilon)
    log_line(f"[RESUME] global_step={global_step}, epsilon iniziale={epsilon_check:.4f}")

    # =========================================================
    # LOOP PRINCIPALE
    # =========================================================
    for episode in range(start_episode, agent.n_episodes + 1):

        reward_done = False
        prev_dist = None
        prev_ped_pos = None
        prev_robot_pos = None
        step_counter = 0
        factor = 1.0
        factor_frontal = 1.0
        total_rewards = 0
        prev_ped_dist = None
        prev_left_speed = 0.0
        prev_right_speed = 0.0
        episode_end_reason = "unknown"
        best_model_saved_this_ep = False
        q_exploit_scalarized_last = None
        q_exploit_action_last = None
        state_mean_last = None
        state_std_last = None
        success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0.0  # ← aggiunta

        debug_print("\n============================", step_counter, episode)
        debug_print(f"EPISODIO: {episode}", step_counter, episode)
        debug_print("============================\n", step_counter, episode)

        total_ep = max(sum(end_reasons.values()), 1)
        recent_collision_rate = end_reasons["collision"] / total_ep
        recent_goal_rate = end_reasons["goal"] / total_ep

        current_ped_speed = reset_pedestrian(episode, recent_collision_rate, recent_goal_rate)

        preference_distribution = generate_preferences(episode)
        # ✅ ORA SCRIVE SUL LOG
        log_line(f"preference_dist: {preference_distribution}")
        current_preference = preference_distribution

        path_start_pos = list(initial_translation)

        history_spatial = deque(maxlen=agent.steps_len)
        history_temporal = deque(maxlen=agent.steps_len)

        for _ in range(agent.steps_len):
            history_spatial.append([1.0] * N_LIDAR_SECTORS)
            history_temporal.append([0.0] * N_LIDAR_SECTORS)

        new_start, spawn_id = reset_robot_randomized(episode)
        path_start_pos = new_start

        debug_print(f"[RESET] Robot: {initial_translation}, Goal: {goal_pos}, Ped: {initial_ped_translation}", step_counter, episode)

        preference = current_preference
        total_rewards = torch.zeros(len(preference), dtype=torch.float32).to(device)

        robot.step(time_step)

        current_robot_pos = translation_field.getSFVec3f()
        robot_rotation = rotation_field.getSFRotation()
        theta = get_yaw_from_webots_rotation(robot_rotation)
        goal_distance, angle_error, dx_goal, dy_goal = compute_goal_metrics(
            current_robot_pos, theta, goal_pos
        )
        ped_pos = ped_translation_field.getSFVec3f()

        goal_tensor = torch.tensor([[
            dx_goal, dy_goal,
            goal_distance / MAX_DISTANCE,
            math.cos(angle_error), math.sin(angle_error),
            1.0 if ped_distance < 2.0 else 0.0,
        ]], dtype=torch.float32).to(device)

        spatial_seq = torch.from_numpy(np.array(history_spatial, dtype=np.float32)).unsqueeze(0).to(device)
        temporal_seq = torch.from_numpy(np.array(history_temporal, dtype=np.float32)).unsqueeze(0).to(device)
        spatial_seq = torch.clamp(spatial_seq, -1.0, 1.0)
        temporal_seq = torch.clamp(temporal_seq, -1.0, 1.0)

        human_pref = torch.tensor(preference, dtype=torch.float32).unsqueeze(0).to(device)
        if human_pref.dim() == 1:
            human_pref = human_pref.unsqueeze(0)

        B, T, _ = spatial_seq.shape
        state = encode_state(spatial_seq, temporal_seq, goal_tensor, human_pref)

        prev_ped_pos = ped_pos
        prev_goal_dist = goal_distance
        prev_theta = theta
        prev_robot_pos = current_robot_pos
        previous_ranges = None

        # --- Loop simulazione ---
        while True:

            robot_speak(preference, step_counter)
            stabilize_robot()
            step_counter += 1
            global_step += 1
            decision_step += 1

            epsilon = decay_epsilon(global_step, agent.epsilon_steps, agent.min_epsilon)
            # Floor adattivo: se il robot è bloccato, forza esplorazione
            if success_rate < 0.08 and episode > 200:
                epsilon = max(epsilon, 0.20)
            elif success_rate < 0.12 and episode > 200:
                epsilon = max(epsilon, 0.10)     
      
            robot_action = select_action(state, preference, episode, step_counter, epsilon, policy_dqn)
            action_counts[robot_action] += 1
            action_counts_diag[robot_action] += 1

            left_speed, right_speed = action_to_speeds(robot_action, factor, factor_frontal)

            delta_left = left_speed - prev_left_speed
            delta_right = right_speed - prev_right_speed
            delta_speed = [delta_left / MAX_SPEED, delta_right / MAX_SPEED]

            apply_action(check_speed(left_speed), check_speed(right_speed))
            prev_left_speed = left_speed
            prev_right_speed = right_speed

            for _ in range(MACRO_STEP):
                if robot.step(time_step) == -1:
                    return

                ranges_inner = read_lidar(lidar)

            current_robot_pos = translation_field.getSFVec3f()
            robot_rotation    = rotation_field.getSFRotation()
            theta             = get_yaw_from_webots_rotation(robot_rotation)
            ped_pos           = ped_translation_field.getSFVec3f()

            if previous_ranges is not None:
                spatial, temporal = extract_lidar_state(ranges_inner, previous_ranges)
            else:
                spatial  = [1.0] * N_LIDAR_SECTORS
                temporal = [0.0] * N_LIDAR_SECTORS
            previous_ranges = ranges_inner

            history_spatial.append(spatial)
            history_temporal.append(temporal)

            goal_distance, angle_error, dx_goal, dy_goal = compute_goal_metrics(
                current_robot_pos, theta, goal_pos
            )
            ped_pos = ped_translation_field.getSFVec3f()
            ranges  = read_lidar(lidar)

            progress = prev_goal_dist - goal_distance

            collision, goal_reached, ped_collision, ped_distance, near_obstacle, \
                dist, lateral, factor, factor_frontal, env_done = \
                analyze_environment(ranges, factor, factor_frontal, goal_distance,
                                    ped_pos, current_robot_pos, theta)

            timeout = step_counter >= agent.max_timestep
            env_done = env_done or timeout

            spatial_seq = torch.from_numpy(np.array(history_spatial, dtype=np.float32)).unsqueeze(0).to(device)
            temporal_seq = torch.from_numpy(np.array(history_temporal, dtype=np.float32)).unsqueeze(0).to(device)
            spatial_seq = torch.clamp(spatial_seq, -1.0, 1.0)
            temporal_seq = torch.clamp(temporal_seq, -1.0, 1.0)

            human_pref = torch.tensor(preference, dtype=torch.float32).unsqueeze(0).to(device)

            goal_tensor = torch.tensor([[
                dx_goal, dy_goal,
                goal_distance / MAX_DISTANCE,
                math.cos(angle_error), math.sin(angle_error),
                1.0 if ped_distance < 2.0 else 0.0,
            ]], dtype=torch.float32).to(device)

            overtake_bonus = 0.0
            if (prev_ped_dist is not None and 
                0.8 < ped_distance < 2.0 and 
                progress > 0.01 and          # robot si sta muovendo verso il goal
                robot_action != 5):           # robot non è fermo
                overtake_bonus = 0.1 * progress  # proporzionale al progresso fatto

            step_reward = get_reward_paper_like(
                progress, goal_reached, collision,
                dist, ped_distance, ped_collision,
                goal_distance, prev_goal_dist,
                current_robot_pos, path_start_pos, goal_pos,
                angle_error, episode, ped_pos, theta, robot_action,
                prev_ped_distance=prev_ped_dist    # ← aggiunto
            )

            if overtake_bonus > 0:
                step_reward = step_reward.clone()
                step_reward[0] = step_reward[0] + overtake_bonus

            # Sostituisci il blocco esistente con:
            if goal_distance < 1.5:
                path_start_pos = list(current_robot_pos)

            prev_robot_pos = current_robot_pos
            prev_ped_pos = ped_pos
            prev_goal_dist = goal_distance
            prev_theta = theta
            prev_ped_dist = ped_distance
            prev_dist = dist

            reward = step_reward

            next_goal_tensor = torch.tensor([[
                dx_goal, dy_goal,
                goal_distance / MAX_DISTANCE,
                math.cos(angle_error), math.sin(angle_error),
                1.0 if ped_distance < 2.0 else 0.0,
            ]], dtype=torch.float32).to(device)

            next_spatial_seq = torch.from_numpy(np.array(history_spatial, dtype=np.float32)).unsqueeze(0).to(device)
            next_temporal_seq = torch.from_numpy(np.array(history_temporal, dtype=np.float32)).unsqueeze(0).to(device)

            next_state = encode_state(next_spatial_seq, next_temporal_seq, next_goal_tensor, human_pref)

            # ✅ STATE + Q DIAGNOSTICS — ora scritti sul log
            if step_counter % 100 == 0:
                with torch.no_grad():
                    s_std  = state.std().item()
                    s_mean = state.mean().item()
                    q_debug = policy_dqn(state)
                    mean_q = q_debug.mean().item()
                    max_q  = q_debug.max().item()
                    min_q  = q_debug.min().item()

                    state_mean_last = s_mean
                    state_std_last  = s_std

                    log_line(f"[STATE] mean={s_mean:.4f} | std={s_std:.4f}")
                    log_line(f"[Q] mean={mean_q:.4f} | max={max_q:.4f} | min={min_q:.4f}")

            # ✅ Q EXPLOIT — ora scritto sul log
            if step_counter % 100 == 0 and epsilon < 0.1:
                with torch.no_grad():
                    q_raw  = policy_dqn(state)
                    pref_t = torch.tensor(preference, dtype=torch.float32).unsqueeze(0).to(device)
                    q_sc   = scalarize(q_raw, pref_t)
                    q_exploit_scalarized_last = q_sc.squeeze().cpu().numpy().tolist()
                    q_exploit_action_last = int(q_sc.argmax().item())
                    log_line(f"[EXPLOIT Q] scalarized: {q_exploit_scalarized_last}")
                    log_line(f"[EXPLOIT action chosen]: {q_exploit_action_last}")

            total_rewards = total_rewards + reward
            step_reward_np = reward.detach().cpu().numpy()
            reward = torch.tensor(step_reward_np, dtype=torch.float32, device=device)

            memory.append((
                spatial_seq.detach().cpu(),
                temporal_seq.detach().cpu(),
                goal_tensor.detach().cpu(),
                human_pref.detach().cpu(),
                next_spatial_seq.detach().cpu(),
                next_temporal_seq.detach().cpu(),
                next_goal_tensor.detach().cpu(),
                robot_action,
                reward.detach().cpu(),
                env_done,
                preference
            ))

            loss = train_step(
                memory, agent, policy_dqn, target_dqn,
                optimizer_value, optimizer_advantage, optimizer_rnn,
                device, global_step, step_counter, episode, debug_print
            )

            if loss is not None and step_counter % 50 == 0:
                debug_print(f"[LOSS] {loss:.4f}", step_counter, episode)

            if global_step % agent.target_update_freq == 0:
                target_dqn.load_state_dict(policy_dqn.state_dict())

            state = next_state
            spatial_seq = next_spatial_seq
            temporal_seq = next_temporal_seq
            goal_tensor = next_goal_tensor

            if env_done:
                if goal_reached:
                    episode_end_reason = "goal"
                    goal_reached_count += 1
                    log_line("Motivo: Goal raggiunto")
                elif ped_collision:
                    episode_end_reason = "ped_collision"
                    log_line("Motivo: Collisione con pedone")
                elif collision:
                    episode_end_reason = "collision"
                    log_line("Motivo: Collisione con ostacolo statico")
                    log_line(f"[COLLISION POS] robot={current_robot_pos[:2]}, goal={goal_pos}")
                elif timeout:
                    episode_end_reason = "timeout"
                    log_line("Motivo: Timeout")

                apply_action(0.0, 0.0)
                robot.step(time_step)
                break

        # fine while

        # ✅ ACTION DIST ogni 10 ep — ora scritta sul log
        if episode % 10 == 0:
            total_actions = sum(action_counts)
            if total_actions > 0:
                dist_str = " | ".join(
                    f"{action_names[i]}: {action_counts[i]/total_actions:.1%}"
                    for i in range(6)
                )
                log_line(f"[ACTION DIST ep.{episode}] {dist_str}")
            action_counts = [0] * 6

        recent_successes.append(1 if goal_reached else 0)
        episode_reward_mean = total_rewards.mean().item()
        recent_rewards.append(episode_reward_mean)
        success_rate = sum(recent_successes) / len(recent_successes)

        # --- BEST SUCCESS RATE ---
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            log_line(f"[BEST SUCCESS] {success_rate:.2%} (ep {episode})")
            torch.save({
                "policy_dqn": policy_dqn.state_dict(),
                "target_dqn": target_dqn.state_dict(),
                "rnn": rnn.state_dict(),
                "optimizer_value": optimizer_value.state_dict(),
                "optimizer_advantage": optimizer_advantage.state_dict(),
                "optimizer_rnn": optimizer_rnn.state_dict(),
                "episode": episode,
                "global_step": global_step,
                "best_success_rate": best_success_rate,
            }, os.path.join(models_dir, "best_success_model.pt"))

        # --- BEST REWARD (qualità del percorso) ---
        if episode_reward_mean > best_reward_mean and success_rate > 0.1:
            best_reward_mean = episode_reward_mean
            log_line(f"[BEST REWARD] {episode_reward_mean:.4f} (ep {episode})")
            torch.save({
                "policy_dqn": policy_dqn.state_dict(),
                "target_dqn": target_dqn.state_dict(),
                "rnn": rnn.state_dict(),
                "optimizer_value": optimizer_value.state_dict(),
                "optimizer_advantage": optimizer_advantage.state_dict(),
                "optimizer_rnn": optimizer_rnn.state_dict(),
                "episode": episode,
                "global_step": global_step,
                "best_reward_mean": best_reward_mean,
            }, os.path.join(models_dir, "best_reward_model.pt"))

        # --- BEST BALANCED (compromesso, solo dopo ep 200) ---
        if episode > 200:
            balanced_score = 0.6 * success_rate + 0.4 * np.clip(episode_reward_mean, -1.0, 1.0)
            if balanced_score > best_balanced_score:
                best_balanced_score = balanced_score
                log_line(f"[BEST BALANCED] score={balanced_score:.4f} (ep {episode})")
                torch.save({
                    "policy_dqn": policy_dqn.state_dict(),
                    "target_dqn": target_dqn.state_dict(),
                    "rnn": rnn.state_dict(),
                    "optimizer_value": optimizer_value.state_dict(),
                    "optimizer_advantage": optimizer_advantage.state_dict(),
                    "optimizer_rnn": optimizer_rnn.state_dict(),
                    "episode": episode,
                    "global_step": global_step,
                    "best_balanced_score": best_balanced_score,
                }, os.path.join(models_dir, "best_balanced_model.pt"))

        if goal_reached:
            end_reasons["goal"] += 1
        elif ped_collision:
            end_reasons["ped_collision"] += 1
        elif collision:
            end_reasons["collision"] += 1
        else:
            end_reasons["timeout"] += 1

        if episode % 50 == 0:
            with torch.no_grad():
                test_pref    = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32).to(device)
                test_spatial  = torch.zeros(1, agent.steps_len, N_LIDAR_SECTORS).to(device)
                test_temporal = torch.zeros(1, agent.steps_len, N_LIDAR_SECTORS).to(device)
                test_goal    = torch.tensor([[1.0, 0.0, 2.0, 1.0, 0.0, 1.0]], dtype=torch.float32).to(device)
                test_state   = encode_state(test_spatial, test_temporal, test_goal, test_pref)
                test_q       = policy_dqn(test_state)   # [1, A, K]

        if episode % 50 == 0:
            log_line(f"[REWARD COMPONENTS ep.{episode}] "
                    f"goal={total_rewards[0]:.2f} | "
                    f"static={total_rewards[1]:.2f} | "
                    f"ped={total_rewards[2]:.2f} | "
                    f"path={total_rewards[3]:.2f}")

        if episode % 100 == 0:
            total_ep = sum(end_reasons.values())
            goal_rate = end_reasons["goal"] / total_ep
            if total_ep > 0:
                # ✅ END REASONS — ora scritte sul log
                log_line(f"[END REASONS ep.{episode}]")
                for k, v in end_reasons.items():
                    log_line(f"  {k}: {v} ({v/total_ep:.1%})")

                q_per_action_std   = test_q.std(dim=2).mean().item()
                q_per_objective_std = test_q.std(dim=1).mean(dim=0)
                q_action_spread    = (test_q.max(dim=1).values - test_q.min(dim=1).values).mean().item()

                log_line("\n========== DIAG ==========")
                log_line(f"[Q] action spread (discriminazione): {q_action_spread:.6f}")
                log_line(f"[Q] std per azione: {q_per_action_std:.6f}")
                log_line(f"[Q] std per obiettivi: {q_per_objective_std.cpu().numpy()}")
                log_line("==========================\n")

                total_actions_diag = sum(action_counts_diag)
                if total_actions_diag > 0:
                    probs = np.array(action_counts_diag) / total_actions_diag
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                else:
                    entropy = 0.0
                action_counts_diag = [0] * 6

                collision_rate     = end_reasons["collision"] / total_ep
                ped_collision_rate = end_reasons["ped_collision"] / total_ep
                timeout_rate       = end_reasons["timeout"] / total_ep

                log_to_file(
                    models_dir=models_dir,
                    episode=episode,
                    epsilon=epsilon,
                    loss=loss if loss is not None else -1,
                    total_reward=total_rewards.detach().cpu().numpy(),
                    success_rate=success_rate,
                    goal_rate=goal_rate,
                    episode_steps=step_counter,
                    end_reasons=end_reasons,
                    collision_rate=collision_rate,
                    ped_collision_rate=ped_collision_rate,
                    goal_reached_count=goal_reached_count,
                    mean_q=mean_q,
                    max_q=max_q,
                    min_q=min_q,
                    q_action_spread=q_action_spread,
                    entropy=entropy,
                )
                flush_log() 
                end_reasons = {"goal": 0, "ped_collision": 0, "collision": 0, "timeout": 0}

        torch.save({
            "policy_dqn": policy_dqn.state_dict(),
            "target_dqn": target_dqn.state_dict(),
            "rnn": rnn.state_dict(),
            "optimizer_value": optimizer_value.state_dict(),
            "optimizer_advantage": optimizer_advantage.state_dict(),
            "optimizer_rnn": optimizer_rnn.state_dict(),
            "global_step": global_step,
            "episode": episode,
            "goal_reached_count": goal_reached_count,
            "best_success_rate": best_success_rate,
            "best_reward_mean": best_reward_mean,       # ← aggiunta
            "best_balanced_score": best_balanced_score, # ← aggiunta
        }, os.path.join(models_dir, "last_model.pt"))

        log_line(f"[EPISODE {episode}] Goal raggiunti: {goal_reached_count}/{episode}")
        log_line(f"[EPISODE {episode}] Success rate: {goal_reached_count / episode:.2%}")

    stop_all()

    torch.save({
        "policy_dqn": policy_dqn.state_dict(),
        "target_dqn": target_dqn.state_dict(),
        "rnn": rnn.state_dict(),
        "optimizer_value": optimizer_value.state_dict(),
        "optimizer_advantage": optimizer_advantage.state_dict(),
        "optimizer_rnn": optimizer_rnn.state_dict(),
        "global_step": global_step,
        "episode": episode,
        "goal_reached_count": goal_reached_count,
        "best_success_rate": best_success_rate,
        "best_reward_mean": best_reward_mean,       # ← aggiunta
        "best_balanced_score": best_balanced_score, # ← aggiunta
    }, os.path.join(models_dir, "last_model.pt"))

    log_line(f"\n[FINE TRAINING]")
    log_line(f"Episodi totali: {episode}")
    log_line(f"Goal raggiunti: {goal_reached_count}/{episode} ({goal_reached_count/episode:.2%})")
    log_line(f"Modello finale salvato in: {models_dir}/final_model.pt")

    while robot.step(time_step) != -1:
        apply_action(0.0, 0.0)


def generate_preferences_dynamic(episode):
    if episode < 500:
        alpha = [5, 5, 5, 5]
    elif episode < 1500:
        alpha = [1, 1, 1, 1]
    else:
        alpha = [0.3, 0.3, 0.3, 0.3]
    return np.random.dirichlet(alpha).tolist()


def generate_preferences(episode):
    if episode < 1500:
        return [0.4, 0.3, 0.2, 0.1]
    elif episode < 2500:
        # introduzione graduale: mix tra fisso e casuale
        alpha = np.random.dirichlet([2.0, 2.0, 2.0, 2.0])
        fixed = np.array([0.4, 0.3, 0.2, 0.1])
        t = (episode - 1500) / 1000.0
        return (fixed * (1 - t) + alpha * t).tolist()
    else:
        return np.random.dirichlet([1.0, 1.0, 1.0, 1.0]).tolist()


def set_pedestrian_speed(speed):
    controller_args_field = pedestrian_node.getField("controllerArgs")
    controller_args_field.setMFString(0, "--trajectory=-1 3, -1 0")
    controller_args_field.setMFString(1, f"--speed={speed}")
    ped_translation_field.setSFVec3f(initial_ped_translation)
    ped_rotation_field.setSFRotation(initial_ped_rotation)
    pedestrian_node.resetPhysics()
    pedestrian_node.restartController()


MISSIONS = {
    "supera_sinistra": {
        "preference":   [0.4, 0.2, 0.3, 0.1],
        "description":  "Supera il pedone passando alla sua sinistra",
        "ped_speed":    0.7,
        "action_bias":  0,
    },
    "supera_destra": {
        "preference":   [0.4, 0.2, 0.3, 0.1],
        "description":  "Supera il pedone passando alla sua destra",
        "ped_speed":    0.7,
        "action_bias":  4,
    },
    "arriva_veloce": {
        "preference":   [0.7, 0.1, 0.1, 0.1],
        "description":  "Raggiungi il goal il prima possibile",
        "ped_speed":    0.5,
        "action_bias":  None,
    },
    "massima_sicurezza": {
        "preference":   [0.1, 0.3, 0.5, 0.1],
        "description":  "Priorità assoluta: non collidere con nessuno",
        "ped_speed":    0.3,
        "action_bias":  None,
    },
    "segui_traiettoria": {
        "preference":   [0.2, 0.2, 0.1, 0.5],
        "description":  "Segui il path pianificato con precisione",
        "ped_speed":    0.5,
        "action_bias":  None,
    },
    "aspetta_poi_passa": {
        "preference":   [0.1, 0.2, 0.6, 0.1],
        "description":  "Aspetta che il pedone passi, poi vai",
        "ped_speed":    0.9,
        "action_bias":  5,
    },
}


def select_mission(mission_name: str) -> dict:
    if mission_name not in MISSIONS:
        print(f"Missione '{mission_name}' non trovata. Uso default.")
        return {"preference": [0.25, 0.25, 0.25, 0.25], "ped_speed": 0.5, "action_bias": None}

    mission = MISSIONS[mission_name]
    print(f"\n{'='*40}")
    print(f"MISSIONE: {mission_name}")
    print(f"Obiettivo: {mission['description']}")
    print(f"Preference: {mission['preference']}")
    print(f"{'='*40}\n")
    return mission


def run(mission_name: str, goal_pos: list):

    mission     = select_mission(mission_name)
    preference  = mission["preference"]
    ped_speed   = mission["ped_speed"]
    action_bias = mission["action_bias"]

    factor = 1.0
    factor_frontal = 1.0
    prev_ped_pos = None
    prev_robot_pos = None
    prev_theta = None
    step_counter = 0
    prev_left_speed = 0.0
    prev_right_speed = 0.0
    left_speed = 0.0
    right_speed = 0.0

    set_pedestrian_speed(ped_speed)

    history_spatial = deque(maxlen=agent.steps_len)
    history_temporal = deque(maxlen=agent.steps_len)
    for _ in range(agent.steps_len):
        history_spatial.append([1.0] * N_LIDAR_SECTORS)   # 1.0 = nessun ostacolo
        history_temporal.append([0.0] * N_LIDAR_SECTORS)

    while robot.step(time_step) != -1:
        step_counter += 1

        current_robot_pos = translation_field.getSFVec3f()
        robot_rotation = rotation_field.getSFRotation()
        theta = get_yaw_from_webots_rotation(robot_rotation)

        goal_distance, angle_error, dx_goal, dy_goal = compute_goal_metrics(
            current_robot_pos, theta, goal_pos
        )

        ped_pos = ped_translation_field.getSFVec3f()
        ranges  = read_lidar(lidar)

        collision, goal_reached, ped_collision, ped_distance, near_obstacle, \
            dist, lateral, factor, factor_frontal, done = \
            analyze_environment(ranges, factor, factor_frontal, goal_distance,
                                ped_pos, current_robot_pos, theta)

        delta_left  = left_speed - prev_left_speed
        delta_right = right_speed - prev_right_speed
        delta_speed = [delta_left / MAX_SPEED, delta_right / MAX_SPEED]

        spatial, temporal = extract_components(
            current_robot_pos, ped_pos, prev_ped_pos, prev_robot_pos,
            theta, prev_theta, time_step, delta_speed
        )
        history_spatial.append(spatial)
        history_temporal.append(temporal)

        spatial_seq = torch.from_numpy(np.array(history_spatial, dtype=np.float32)).unsqueeze(0).to(device)
        temporal_seq = torch.from_numpy(np.array(history_temporal, dtype=np.float32)).unsqueeze(0).to(device)

        goal_tensor = torch.tensor([[
            dx_goal, dy_goal,
            goal_distance / MAX_DISTANCE,
            math.cos(angle_error), math.sin(angle_error),
            1.0 if ped_distance < 2.0 else 0.0
        ]], dtype=torch.float32).to(device)

        human_pref = torch.tensor(preference, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            state    = rnn(spatial_seq, temporal_seq, goal_tensor, human_pref)
            q_values = policy_dqn(state)

        pref_t   = torch.tensor(preference, dtype=torch.float32).unsqueeze(0).to(device)
        q_scalar = scalarize(q_values, pref_t)

        if action_bias is not None:
            BIAS_STRENGTH = 0.3
            q_scalar[0, action_bias] += BIAS_STRENGTH

        robot_action = torch.argmax(q_scalar, dim=1).item()

        left_speed, right_speed = action_to_speeds(robot_action, factor, factor_frontal)
        delta_left  = left_speed - prev_left_speed
        delta_right = right_speed - prev_right_speed
        delta_speed = [delta_left / MAX_SPEED, delta_right / MAX_SPEED]
        prev_left_speed  = left_speed
        prev_right_speed = right_speed
        apply_action(left_speed, right_speed)

        prev_ped_pos   = ped_pos
        prev_robot_pos = current_robot_pos
        prev_theta     = theta

        if done:
            print("Episode finished")
            apply_action(0.0, 0.0)
            break


goal_pos = [1, 1]

path = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(path, "models")
best_path  = os.path.join(models_dir, "best_model.pt")
last_path  = os.path.join(models_dir, "last_model.pt")

empty = True
for _ in os.scandir(models_dir):
    empty = False
    break

if MODE == "train":
    print("MODE TRAIN")
    train(goal_pos, mode=MODE)

elif MODE == "finetune":
    print("MODE FINETUNE")
    train(goal_pos, mode=MODE)

elif MODE == "run":
    print("MODE RUN")

    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
    elif os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location=device)
    else:
        print("⚠ Nessun modello trovato")
        exit()

    policy_dqn.load_state_dict(checkpoint["policy_dqn"])
    rnn.load_state_dict(checkpoint["rnn"])

    policy_dqn.eval()
    rnn.eval()

    run(
        mission_name="aspetta_poi_passa",
        goal_pos=goal_pos
    )
