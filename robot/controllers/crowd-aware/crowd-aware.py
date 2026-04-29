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

DANGER_DISTANCE_PEDESTRIAN = 0.6
MAX_DISTANCE_PEDESTRIAN = 3.5

MAX_DISTANCE = 10
MAX_LATERAL_DISTANCE = 2.5  # distanza massima considerata per normalizzazione
CRUISING_SPEED = 4.5
TURN_SPEED = 3.0

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
# Recupero del nodo pedestrian
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
# print(f"[LIDAR] max_range={lidar.getMaxRange()}, resolution={lidar.getHorizontalResolution()}")

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


# =========================================================
# CREAZIONE RETI
# =========================================================

# Inizializza RNN
rnn = RNN.CrowdNavNet(
    spatial_dim=3,
    temporal_dim=2,
    goal_dim=7,
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

    # Lista di stringhe per azione
    # action_names = ["LEFT", "FRONT LEFT", "FRONT", "FRONT RIGHT", "RIGHT"]

    # print("Input - State:", state)
    # =========================
    # EXPLORATION (random robot action)
    # =========================
    if torch.rand(1).item() < epsilon:
        # print("Exploration: azione casuale")
        # time.sleep(2.0)

        robot_action = torch.randint(0, 6, (1,))
        # debug_print(f"Azione selezionata: {action_names[robot_action.item()]} (random)", step_counter, episode)
        return robot_action.item()

    # =========================
    # EXPLOITATION (best robot action)
    # =========================
    else:
        # print("Exploitation: azione migliore secondo il modello")
        with torch.no_grad():
            # print("\n--- DQN ---")
            # pref = torch.tensor(preference, dtype=torch.float32).unsqueeze(0).to(device)

            # print("State:", state)
            q_values = policy_dqn(state)
            debug_print(f"\n[DQN] Q-values (raw):\n{q_values.detach().cpu().numpy()}", step_counter, episode)
            # print(f"[DQN] Input state shape:    {state.shape}")
            # print(f"[DQN] Q-values shape:       {q_values.shape}")
            # print(f"[DQN] Q-values:\n{q_values.detach().numpy()}")
            # print(f"[DQN] Contiene NaN:         {torch.isnan(q_values).any().item()}")
            # print(f"[DQN] Q-values min/max:     {q_values.min().item():.4f} / {q_values.max().item():.4f}")

            pref = torch.tensor(preference, dtype=torch.float32).unsqueeze(0).to(device)
            # print("Preferenza q-network:", pref)

            # Calcola Q-values scalati per preferenza umana
            q_scalar = scalarize(q_values, pref)

            debug_print(f"[DQN] Q-values (scalarized):\n{q_scalar.detach().cpu().numpy()}", step_counter, episode)

            # q_scalar = torch.nan_to_num(q_scalar, nan=-1e6)

            # scegli migliore azione robot
            robot_action = torch.argmax(q_scalar, dim=1)
            # debug_print(f"Azione selezionata: {action_names[robot_action.item()]} (Q-value: {q_values[0, robot_action.item()]:.4f})", step_counter, episode)
            return robot_action.item()

# def select_action(state, preference, episode, step_counter, epsilon=0.1, policy_dqn=None):
#     return 0  # sempre FRONT


# Funzione per convertire l'azione discreta in velocità per le ruote
def action_to_speeds(action, factor, factor_frontal):

    if action == 0:      # LEFT — sterzata netta per aggirare
        return -TURN_SPEED, TURN_SPEED

    elif action == 1:    # FRONT LEFT — curva morbida a sinistra
        return CRUISING_SPEED * factor, CRUISING_SPEED

    elif action == 2:    # FRONT — dritto verso il goal
        return CRUISING_SPEED, CRUISING_SPEED

    elif action == 3:    # FRONT RIGHT — curva morbida a destra
        return CRUISING_SPEED, CRUISING_SPEED * factor

    elif action == 4:    # RIGHT — sterzata netta per aggirare
        return TURN_SPEED, -TURN_SPEED

    elif action == 5:    # STOP — aspetta
        return 0.0, 0.0


def stabilize_robot():
    pos = translation_field.getSFVec3f()
    # Se il robot si è inclinato troppo o è volato via
    if pos[2] > 0.3 or any(math.isnan(v) for v in pos):
        # print("Robot instabile - correzione posizione")
        translation_field.setSFVec3f([pos[0], pos[1], 0.095])
        robot_node.resetPhysics()


def stop_pedestrian():
    pos = ped_translation_field.getSFVec3f()
    rot = ped_rotation_field.getSFRotation()

    ped_translation_field.setSFVec3f(pos)
    ped_rotation_field.setSFRotation(rot)

    pedestrian_node.resetPhysics()


# Funzione per estrarre informazione spaziale e temporale dai dati LIDAR
def check_lidar(ranges, previous_ranges, spatial_info, temporal_info):

    # Scansione di ogni raggio del LIDAR, escludendo quelli inutilizzati
    for i in range(UNUSED_POINT, urg04lx_width - UNUSED_POINT - 1):

        # Filtro valori infiniti e oltre soglia
        if math.isinf(ranges[i]) or math.isinf(previous_ranges[i]):
            continue

        # Filtro ostacoli troppo lontani
        if ranges[i] >= range_threshold:
            continue

        # Calcolo valore normalizzato e velocità radiale (componente spaziale)
        spatial = 1.0 - ranges[i] / max_range

        # Calcolo velocità radiale (componente temporale)
        delta = previous_ranges[i] - ranges[i]
        radial_velocity = delta / (time_step / 1000.0)

        # Assegno ai rispettivi settori
        # Determina in quale zona del campo visivo cade il raggio
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

        # Accumulo informazione
        spatial_info[idx] += spatial
        temporal_info[idx] += radial_velocity

    # print(f"Spatial info: {spatial_info}")
    # print(f"Temporal info: {temporal_info}")

    return spatial_info, temporal_info


# Resetta il robot alla posizione iniziale all'inizio di ogni episodio
def reset_robot():
    translation_field.setSFVec3f(initial_translation)
    rotation_field.setSFRotation(initial_rotation)

    robot_node.resetPhysics()   # azzera velocità
    robot.simulationResetPhysics()   # reset fisica globale
    apply_action(0.0, 0.0)      # ferma ruote


# def reset_pedestrian(episode):
#     controller_args_field = pedestrian_node.getField("controllerArgs")

#     # =========================
#     # CURRICULUM A BLOCCHI (STABILE)
#     # =========================
#     if episode < 1500:
#         speed = 0.0

#     elif episode < 3000:
#         speed = 0.2

#     elif episode < 4500:
#         speed = 0.5

#     else:
#         speed = 0.8

#     # traiettoria sempre random ma coerente
#     x = np.random.uniform(-1.5, -0.5)
#     controller_args_field.setMFString(0, f"--trajectory={x:.2f} 3, {x:.2f} 0")
#     controller_args_field.setMFString(1, f"--speed={speed:.3f}")

#     ped_translation_field.setSFVec3f(initial_ped_translation)
#     ped_rotation_field.setSFRotation(initial_ped_rotation)

#     pedestrian_node.resetPhysics()
#     pedestrian_node.restartController()

#     print(f"[PEDESTRIAN RESET] episode={episode}, speed={speed}")

#     return speed


def reset_pedestrian(episode):
    speeds = [0.3, 0.6, 0.9]
    # print(f"[ROBOT] Nuova velocità pedestrian: {new_speed}")

    # Sceglie la velocità in speeds non random ma in ordine
    if not hasattr(reset_pedestrian, "counter"):
        reset_pedestrian.counter = 0

    new_speed = speeds[reset_pedestrian.counter % len(speeds)]
    reset_pedestrian.counter += 1

    # Cambia gli argomenti del controller prima del restart
    controller_args_field = pedestrian_node.getField("controllerArgs")
    x = np.random.uniform(-1.5, -0.5)
    controller_args_field.setMFString(0, f"--trajectory={x:.2f} 3, {x:.2f} 0")
    controller_args_field.setMFString(1, f"--speed={new_speed}")

    # Reset posizione
    ped_translation_field.setSFVec3f(initial_ped_translation)
    ped_rotation_field.setSFRotation(initial_ped_rotation)

    pedestrian_node.resetPhysics()
    pedestrian_node.restartController()
    return new_speed


# def reset_pedestrian(episode):
#     controller_args_field = pedestrian_node.getField("controllerArgs")
#     controller_args_field.setMFString(0, "--trajectory=-2 3, -2 0")

#     if episode < 800:
#         new_speed = 0.0
#     elif episode < 1500:
#         new_speed = 0.2
#     elif episode < 2500:
#         new_speed = 0.4
#     elif episode < 4000:
#         new_speed = 0.6
#     else:
#         new_speed = 0.9

#     controller_args_field.setMFString(1, f"--speed={new_speed}")
#     ped_translation_field.setSFVec3f(initial_ped_translation)
#     ped_rotation_field.setSFRotation(initial_ped_rotation)
#     pedestrian_node.resetPhysics()
#     pedestrian_node.restartController()
#     print(f"[PEDESTRIAN RESET] episode={episode}, speed={new_speed}")
#     return new_speed


# Rileva collisioni utilizzando un comando di movimento
def detect_collision_lidar(ranges, factor, factor_frontal, ped_pos=None, robot_pos=None):
    n = len(ranges)
    if n == 0:
        return False, float("inf"), float("inf"), factor, factor_frontal, False

    # --- settori ---
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

    # 🔥 soglie più coerenti
    CRITICAL = 0.6
    WARNING = 1.5

    collision = False
    near_obstacle = False

    if min_dist < CRITICAL:
        collision = True
        near_obstacle = True
        factor = 0.0
        factor_frontal = 0.0

    elif min_dist < WARNING:
        near_obstacle = True
        factor_frontal = np.clip(min_dist / WARNING, 0.3, 1.0)

    else:
        factor_frontal = 1.0

    if ped_pos is not None and robot_pos is not None:
        ped_dx = ped_pos[0] - robot_pos[0]
        ped_dy = ped_pos[1] - robot_pos[1]
        ped_dist = math.sqrt(ped_dx**2 + ped_dy**2)
        if ped_dist < 2.0 and abs(min_dist - ped_dist) < 0.5:
            # la lettura LIDAR frontale corrisponde al pedone → ignora per collision statica
            collision = False
            near_obstacle = False
            factor = 1.0
            factor_frontal = 1.0

    return collision, min_dist, min_lateral, factor, factor_frontal, near_obstacle


def detect_goal(goal_distance):
    # print(f"goal raggiunto: {goal_distance < GOAL_THRESHOLD}")
    goal_reached = goal_distance < GOAL_THRESHOLD

    return goal_reached, goal_distance


def preference_function(reward_vector, base_reward, preference, n_pref):

    for i in range(n_pref):
        reward_vector[i] = base_reward * preference[i]

    return reward_vector


def normalize(value, min_val, max_val):
    return 2.0 * (value - min_val) / (max_val - min_val) - 1.0


def get_reward(
    progress, goal_reached, collision, near_obstacle,
    dist, lateral,
    ped_distance, goal_distance,
    robot_pos, ped_pos, theta, goal_pos,
    episode, done, step_counter, path_start_pos,
    angle_error,
    prev_dist,
    prev_ped_dist,
    ped_collision, robot_action
):

    # =========================
    # 1. GOAL / PROGRESS (PRINCIPALE)
    # =========================
    reward_goal = 8.0 * progress
    reward_goal += 1.5 * np.cos(angle_error)

    if goal_reached:
        reward_goal += 10.0

    if robot_action == 5:  # STOP
        reward_goal -= 0.1

    # =========================
    # 2. OSTACOLI STATICI
    # =========================
    effective_dist = dist

    ped_dx = ped_pos[0] - robot_pos[0]
    ped_dy = ped_pos[1] - robot_pos[1]
    ped_dist_actual = math.sqrt(ped_dx**2 + ped_dy**2)

    if ped_dist_actual < 1.5 and abs(dist - ped_dist_actual) < 0.5:
        effective_dist = 10.0  # ignora lettura se è il pedone

    if effective_dist > 1.5:
        reward_safety_object = 0.0          # distanza sicura → nessuna penalità
    elif effective_dist > 0.6:
        reward_safety_object = -1.0 / (effective_dist + 0.2)   # zona warning
    else:
        reward_safety_object = -5.0         # zona critica → penalità fissa, non divergente

    if prev_dist is not None:
        reward_safety_object += 0.5 * (effective_dist - prev_dist)

    if collision:
        reward_safety_object -= 5.0

    # =========================
    # 3. PEDONE (IMPORTANTE)
    # =========================
    if ped_distance is None:
        reward_safety_pedestrian = 0.0
    else:
        reward_safety_pedestrian = -2.0 / (ped_distance + 0.3)
        if prev_ped_dist is not None:
            delta_ped = ped_distance - prev_ped_dist
            if delta_ped > 0:
                reward_safety_pedestrian += 2.0 * delta_ped
            else:
                reward_safety_pedestrian += 0.5 * delta_ped
        if ped_distance > 2.5:
            reward_safety_pedestrian += 0.1

    if ped_collision:                          # ← aggiungi questo
        reward_safety_pedestrian -= 8.0

    # =========================
    # 4. PATH TRACKING
    # =========================
    cross_track_error = path_tracking(robot_pos, path_start_pos, goal_pos)
    reward_path = -0.5 * cross_track_error

    # =========================
    # VECTOR REWARD (NO TANH, NO CLIP)
    # =========================
    # Penalità temporale su reward_goal per incentivare velocità
    reward_goal -= 0.05

    reward = np.array([
        normalize(reward_goal, -10.0, 15.0),           # goal + progress
        normalize(reward_safety_object, -10.0, 0.5),   # ostacoli statici
        normalize(reward_safety_pedestrian, -10.0, 1.0),  # pedone
        normalize(reward_path, -5.0, 0.0)              # path tracking
    ], dtype=np.float32)

    reward = np.clip(reward, -1.0, 1.0)  # safety clip per valori fuori range

    return torch.tensor(reward, dtype=torch.float32, device=device), done, cross_track_error


def get_yaw_from_webots_rotation(rotation):
    x, y, z, angle = rotation
    # Se l’asse è Z (robot mobile piano XY)
    if z < 0:
        angle = -angle
    return angle  # questo è theta (yaw)


# =========================================================
# MODELLO CINEMATICO DIFFERENTIAL DRIVE
# =========================================================
def differential_drive_kinematics(
        x,
        y,
        theta,
        wl,
        wr,
        wheel_radius,
        wheel_base,
        dt
):
    """
    Aggiorna la posa del robot usando modello cinematico.

    x, y, theta  -> posa attuale
    wl, wr       -> velocità angolari ruote (rad/s)
    wheel_radius -> raggio ruote
    wheel_base   -> distanza tra ruote
    dt           -> passo temporale (secondi)
    """

    # velocità lineare
    v = wheel_radius * (wr + wl) / 2.0

    # velocità angolare
    omega = wheel_radius * (wr - wl) / wheel_base

    # integrazione
    x_new = x + v * math.cos(theta) * dt
    y_new = y + v * math.sin(theta) * dt
    theta_new = theta + omega * dt

    return x_new, y_new, theta_new, v, omega


def compute_goal_metrics(robot_pos, theta, goal_pos):

    x, y = robot_pos[0], robot_pos[1]

    # vettore goal in frame globale
    dx = goal_pos[0] - x
    dy = goal_pos[1] - y

    # distanza
    goal_distance = math.sqrt(dx**2 + dy**2)

    # =========================
    # FRAME DEL ROBOT (ROTATED)
    # =========================
    dx_goal = math.cos(theta) * dx + math.sin(theta) * dy
    dy_goal = -math.sin(theta) * dx + math.cos(theta) * dy

    # =========================
    # DIREZIONE NORMALIZZATA
    # =========================
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

    return num / den  # cross-track error in metri, grezzo


def decay_epsilon(global_step, epsilon_steps, min_epsilon=0.05):
    if global_step <= epsilon_steps:
        epsilon = max(min_epsilon, np.exp(-global_step / epsilon_steps))
    else:
        epsilon = min_epsilon

    return epsilon


def read_lidar(lidar):
    return lidar.getRangeImage()[::-1]


def detect_pedestrian_collision(robot_pos, ped_pos, theta):
    # Distanza tra robot e pedone
    dx = ped_pos[0] - robot_pos[0]
    dy = ped_pos[1] - robot_pos[1]
    ped_distance = math.sqrt(dx**2 + dy**2)
    # print(f"Distanza pedone: {ped_distance:.3f} m")

    # Direzione robot
    # heading_x = math.cos(theta)
    # heading_y = math.sin(theta)

    # Dot product per capire se il pedone è davanti o dietro
    # dot = dx * heading_x + dy * heading_y

    # il pedone è dietro se dot product è negativo, ignora collisione
    # if dot < 0:
    #     return False, ped_distance

    # il pedone è avanti, normale controllo
    ped_collision = ped_distance < DANGER_DISTANCE_PEDESTRIAN

    return ped_collision, ped_distance


def analyze_environment(ranges, factor, factor_frontal, goal_distance, ped_pos, robot_pos, theta):

    # Funzione che calcola la distanza tra il robot e l'oggetto più vicino:
    #     - Calcola factor utile a rallentare robot in presenza di ostacoli
    #     - Restituisce valori booleani per capire la distanza degli ostacoli
    collision, min_dist, min_lateral, factor, factor_frontal, near_obstacle = detect_collision_lidar(
        ranges,
        factor,
        factor_frontal,
        ped_pos,
        robot_pos
    )

    # Funzione che restituisce un valore booleano se si è arrivati al goal
    goal_reached, _ = detect_goal(goal_distance)

    # Funzione che restituisce un valore booleano se si è troppo vicini al pedone
    ped_collision, ped_distance = detect_pedestrian_collision(robot_pos, ped_pos, theta)

    # if ped_collision:
    #     collision = False  # annulla collisione lidar

    done = collision or ped_collision or goal_reached

    return collision, goal_reached, ped_collision, ped_distance, near_obstacle, min_dist, min_lateral, factor, factor_frontal, done


def extract_components(robot_pos, ped_pos, prev_ped_pos, prev_robot_pos,
                       theta, prev_theta, dist=1.0, lateral=1.0):

    # =========================
    # SPAZIALE (STABILE E NON RIDONDANTE)
    # =========================
    dx = ped_pos[0] - robot_pos[0]
    dy = ped_pos[1] - robot_pos[1]

    distance = math.sqrt(dx**2 + dy**2)

    # angolo relativo normalizzato
    angle = math.atan2(dy, dx) - theta

    spatial = [
        distance / MAX_DISTANCE,
        math.cos(angle),
        math.sin(angle)
    ]

    # =========================
    # TEMPORALE (VELOCITÀ RELATIVA ROBUSTA)
    # =========================
    if prev_ped_pos is not None and prev_robot_pos is not None:

        prev_dx = prev_ped_pos[0] - prev_robot_pos[0]
        prev_dy = prev_ped_pos[1] - prev_robot_pos[1]

        prev_distance = math.sqrt(prev_dx**2 + prev_dy**2)
        prev_angle = math.atan2(prev_dy, prev_dx) - prev_theta

        # dt implicito (approssimato stabile)
        d_distance = (distance - prev_distance) / MAX_DISTANCE

        # differenza angolare stabilizzata (wrap-safe)
        d_angle = math.atan2(
            math.sin(angle - prev_angle),
            math.cos(angle - prev_angle)
        ) / math.pi

        # velocità “utile” verso il pedone (più informativa)
        temporal = [
            d_distance,
            d_angle
        ]

    else:
        temporal = [0.0, 0.0]

    spatial = np.clip(spatial, -1, 1)
    temporal = np.clip(temporal, -1, 1)

    return spatial, temporal

# def pedestrian_right_overcome(robot_pos, ped_pos, reward):
#     # Il robot deve superare il pedestrian a sinistra
#     print(f"Posizione robot: x={robot_pos[0]:.3f}, y={robot_pos[1]:.3f}")
#     print(f"Posizione pedone: x={ped_pos[0]:.3f}, y={ped_pos[1]:.3f}")

#     if abs(ped_pos[0] - robot_pos[0]) < 0.5 and abs(ped_pos[1] - robot_pos[1]) < 2:

#         if ped_pos[0] > robot_pos[0] and ped_pos[1] > robot_pos[1]:
#             # for _ in range(int(30000 / time_step)):
#             #     pass
#             reward += OVERCOME_REWARD
#             print("Superamento a destra!")

#     return reward


# def pedestrian_left_overcome(robot_pos, ped_pos, reward):
#     # Il robot deve superare il pedestrian a sinistra
#     print(f"Posizione robot: x={robot_pos[0]:.3f}, y={robot_pos[1]:.3f}")
#     print(f"Posizione pedone: x={ped_pos[0]:.3f}, y={ped_pos[1]:.3f}")

#     if abs(ped_pos[0] - robot_pos[0]) < 0.5 and abs(ped_pos[1] - robot_pos[1]) < 2:

#         if ped_pos[0] < robot_pos[0] and ped_pos[1] < robot_pos[1]:
#             # for _ in range(int(30000 / time_step)):
#             #     pass
#             reward += OVERCOME_REWARD
#             print("Superamento a sinistra!")

#     return reward


def memorize_checkpoint(models_dir, optimizer_value, optimizer_advantage, optimizer_rnn,
                        policy_dqn, target_dqn, rnn, global_step, episode):
    checkpoint_path = os.path.join(models_dir, "last_model.pt")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        policy_dqn.load_state_dict(checkpoint["policy_dqn"])
        target_dqn.load_state_dict(checkpoint["target_dqn"])
        rnn.load_state_dict(checkpoint["rnn"])

        optimizer_value.load_state_dict(checkpoint["optimizer_value"])
        optimizer_advantage.load_state_dict(checkpoint["optimizer_advantage"])
        optimizer_rnn.load_state_dict(checkpoint["optimizer_rnn"])

        global_step = checkpoint["global_step"]
        start_episode = checkpoint["episode"] + 1
        goal_reached_count = checkpoint.get("goal_reached_count", 0)
        print(f"Riprendo da episodio {start_episode}, global_step={global_step}, goal_reached_count={goal_reached_count}")
    else:
        start_episode = 1
        global_step = 0
        goal_reached_count = 0  # ← mancava questo

    return global_step, start_episode, goal_reached_count


def check_finetune(mode, optimizer_value, optimizer_advantage, optimizer_rnn,
                   policy_dqn, target_dqn, rnn, global_step, episode, models_dir):
    if mode == "finetune":
        global_step, start_episode, goal_reached_count = memorize_checkpoint(
            models_dir,
            optimizer_value,
            optimizer_advantage,
            optimizer_rnn,
            policy_dqn,
            target_dqn,
            rnn,
            global_step,
            episode
        )
    else:
        start_episode = 1
        global_step = 0
        goal_reached_count = 0

    return global_step, start_episode, goal_reached_count


def debug_print(msg, step_counter=None, episode=None):
    if step_counter is None:
        print(msg)
    else:
        if step_counter % DEBUG_PRINT_EVERY == 0:
            prefix = ""
            if episode is not None:
                prefix += f"[EP {episode}] "
            prefix += f"[STEP {step_counter}] "
            print(prefix + str(msg))  # stampa informazioni di debug ogni N step


def scalarize(q, pref):
    return (q * pref.unsqueeze(1)).sum(dim=2)


def chebyshev_scalarize(q, pref, tau=0.1):
    weights = pref.unsqueeze(1)
    weighted = weights * q

    return tau * torch.logsumexp(weighted / tau, dim=2)


def sample_preferences(k=5):
    # Dirichlet = standard in MORL
    alpha = np.ones(4)  # oppure skewed se vuoi bias
    prefs = np.random.dirichlet(alpha, size=k)
    return torch.tensor(prefs, dtype=torch.float32, device=device)


def train_step(
    memory,
    agent,
    policy_dqn,
    target_dqn,
    optimizer_value,
    optimizer_advantage,
    optimizer_rnn,
    device,
    global_step,
    step_counter,
    episode,
    debug_print=None
):
    if len(memory) < agent.batch_size:
        return None

    batch = memory.sample(agent.batch_size)
    (b_spatial, b_temporal, b_goal, b_pref_rnn,
     b_next_spatial, b_next_temporal, b_next_goal,
     b_actions, b_rewards, b_dones, b_pref) = zip(*batch)

    b_spatial = torch.cat(b_spatial).to(device)
    b_temporal = torch.cat(b_temporal).to(device)
    b_goal = torch.cat(b_goal).to(device)
    b_pref_rnn = torch.cat(b_pref_rnn).to(device)

    b_next_spatial = torch.cat(b_next_spatial).to(device)
    b_next_temporal = torch.cat(b_next_temporal).to(device)
    b_next_goal = torch.cat(b_next_goal).to(device)

    b_actions = torch.tensor(b_actions, dtype=torch.long, device=device).unsqueeze(1)
    b_rewards = torch.stack([r.to(device) for r in b_rewards])
    b_dones = torch.tensor(b_dones, dtype=torch.float32, device=device)
    b_pref = torch.stack([
        torch.tensor(p, dtype=torch.float32) if not isinstance(p, torch.Tensor) else p
        for p in b_pref
    ]).to(device)

    B = agent.batch_size

    # RNN eseguita con gradiente: aggiorna i pesi della RNN
    b_state = rnn(b_spatial, b_temporal, b_goal, b_pref_rnn)

    q_values = policy_dqn(b_state)                                    # [B, A, K]
    current_q = q_values[torch.arange(B), b_actions.squeeze(1)]        # [B, K]

    with torch.no_grad():

        b_next_state = rnn(b_next_spatial, b_next_temporal, b_next_goal, b_pref_rnn)

        q_next = target_dqn(b_next_state)  # [B, A, K]

        # =========================
        # DEBUG 1: stato RNN
        # =========================
        if step_counter % 5000 == 0:
            print("\n[DEBUG RNN NEXT STATE]")
            print("mean:", b_next_state.mean().item())
            print("std :", b_next_state.std().item())
            print("min :", b_next_state.min().item())
            print("max :", b_next_state.max().item())

        # =========================
        # DEBUG 2: Q raw
        # =========================
        if step_counter % 5000 == 0:
            print("\n[DEBUG Q NEXT RAW]")
            print("q_next shape:", q_next.shape)
            print("q_next mean:", q_next.mean().item())
            print("q_next std :", q_next.std().item())
            print("q_next min :", q_next.min().item())
            print("q_next max :", q_next.max().item())

        q_next_scalar = scalarize(q_next, b_pref)

        # =========================
        # DEBUG 3: scalarization
        # =========================
        if step_counter % 5000 == 0:
            print("\n[DEBUG SCALARIZATION]")
            print("pref sample:", b_pref[0].detach().cpu().numpy())
            print("q_scalar shape:", q_next_scalar.shape)
            print("q_scalar mean:", q_next_scalar.mean().item())
            print("q_scalar std :", q_next_scalar.std().item())
            print("q_scalar min :", q_next_scalar.min().item())
            print("q_scalar max :", q_next_scalar.max().item())

        next_actions = q_next_scalar.argmax(dim=1)

        # =========================
        # DEBUG 4: azioni scelte
        # =========================
        if step_counter % 5000 == 0:
            unique_actions = torch.bincount(next_actions, minlength=q_next.shape[1])
            print("\n[DEBUG ACTION DISTRIBUTION]")
            print("actions:", next_actions[:10].detach().cpu().numpy())
            print("distribution:", unique_actions.detach().cpu().numpy())

        q_next_selected = q_next[torch.arange(B), next_actions]

        # =========================
        # DEBUG 5: target Q
        # =========================
        target_q = b_rewards + agent.discount_factor * q_next_selected * (1 - b_dones.unsqueeze(1))

        if step_counter % 5000 == 0:
            print("\n[DEBUG TARGET Q]")
            print("reward mean:", b_rewards.mean().item())
            print("target_q mean:", target_q.mean().item())
            print("target_q std :", target_q.std().item())
            print("target_q min :", target_q.min().item())
            print("target_q max :", target_q.max().item())

    # loss
    loss = F.smooth_l1_loss(current_q, target_q)

    if step_counter % 50 == 0:
        print("\n[DEBUG LOSS]")
        print(loss.item())

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
    """
    Stampa il comportamento del robot ogni N step.
    """

    if step_counter % print_every != 0:
        return  # non stampa

    pref = np.array(preference)
    idx = np.argmax(pref)

    messages = [
        "MI CONCENTRO SUL GOAL",
        "PRIORITÀ: EVITARE OSTACOLI",
        "ATTENZIONE AI PEDONI",
        "SEGUO LA TRAIETTORIA"
    ]

    strength = pref[idx]

    if strength > 0.6:
        tone = "!!!"
    elif strength > 0.4:
        tone = "!!"
    else:
        tone = "!"

    msg = messages[idx] + " " + tone

    print(f"[STEP {step_counter}] 🤖 {msg}")


agent = Agent("hyperparameters.yml", "thiago")


# =========================================================
# TRAINING PARAMETRI
# =========================================================
def train(goal_pos, mode):

    global global_step
    global_step = 0
    episode = 0
    goal_reached_count = 0
    start_episode = 1
    # DOPO (aggiungi queste due righe sotto)
    # max_reward = -float('inf')
    best_success_rate = 0.0
    recent_successes = deque(maxlen=20)
    prev_theta = None
    decision_step = 0
    end_reasons = {"goal": 0, "ped_collision": 0, "collision": 0, "timeout": 0}
    action_counts = [0] * 6
    action_names = ["LEFT", "FRONT_LEFT", "FRONT", "FRONT_RIGHT", "RIGHT", "STOP"]

    # DQN replay buffer (memoria per training)
    memory = DQN.ReplayMemory(agent.memory_maxlen)

    # copia pesi iniziali randomici da policy a target
    target_dqn.load_state_dict(policy_dqn.state_dict())

    # Ottimizzatore per aggiornare i pesi della policy network
    # learning rate più alto per DQN, più basso per RNN che deve essere più stabile
    optimizer_value = torch.optim.Adam(
        list(policy_dqn.trunk.parameters()) +
        list(policy_dqn.value_stream.parameters()),
        lr=agent.learning_rate_dqn
    )
    optimizer_advantage = torch.optim.Adam(
        policy_dqn.advantage_stream.parameters(),
        lr=agent.learning_rate_dqn * 5.0  # advantage impara 5x più veloce
    )

    optimizer_rnn = torch.optim.Adam(rnn.parameters(), lr=agent.learning_rate_rnn)

    # Creazione directory per salvare i modelli
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # models_dir = "/home/luigi/webots_ws/src/webots_ros2/webots_ros2_tiago/webots_ros2_tiago/models"
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    global_step, start_episode, goal_reached_count = check_finetune(
        mode, optimizer_value, optimizer_advantage, optimizer_rnn,
        policy_dqn, target_dqn, rnn, global_step, start_episode, models_dir
    )

    epsilon_check = decay_epsilon(global_step, agent.epsilon_steps, agent.min_epsilon)
    print(f"[RESUME] global_step={global_step}, epsilon iniziale={epsilon_check:.4f}")

    # =========================================================
    # LOOP PRINCIPALE
    # =========================================================
    for episode in range(start_episode, agent.n_episodes + 1):

        prev_dist = None
        prev_ped_pos = None
        prev_robot_pos = None
        step_counter = 0  # Reset counter
        factor = 1.0  # Fattori di scala per la velocità del robot
        factor_frontal = 1.0  # Fattore per la velocità frontale
        total_rewards = 0
        prev_ped_dist = None

        debug_print("\n============================", step_counter, episode)
        debug_print(f"EPISODIO: {episode}", step_counter, episode)
        debug_print("============================\n", step_counter, episode)

        # current_preference = generate_preferences_dynamic(episode)
        preference_distribution = [
            [0.5, 0.2, 0.2, 0.1],    # goal molto prioritario
            [0.45, 0.2, 0.25, 0.1],
            [0.4, 0.2, 0.3, 0.1],
            [0.35, 0.2, 0.3, 0.15],
            [0.3, 0.25, 0.25, 0.2],
        ]
        # current_preference = random.choice(preference_distribution)
        # Reset posizione iniziale del robot e del pedone
        path_start_pos = list(initial_translation)

        # Inizializza le code per la storia spaziale e temporale da dare alla RNN
        history_spatial = deque(maxlen=agent.steps_len)
        history_temporal = deque(maxlen=agent.steps_len)

        for _ in range(agent.steps_len):
            history_spatial.append([0.0] * 3)
            history_temporal.append([0.0] * 2)

        # Riposizionamento del robot nella posizione iniziale
        reset_robot()
        debug_print(f"[RESET] Robot: {initial_translation}, Goal: {goal_pos}, Ped: {initial_ped_translation}", step_counter, episode)
        current_ped_speed = reset_pedestrian(episode)

        # # Campiono randomicamente una preferenza lineare
        # if episode < 500:
        #     current_preference = [0.4, 0.3, 0.2, 0.1]  # fisso, goal-oriented
        # elif episode % 20 == 0:
        #     current_preference = random.choice(preference_distribution)

        current_preference = random.choice(preference_distribution)
        debug_print(f"Current_preference: {current_preference}", step_counter, episode)

        preference = current_preference
        # debug_print(f"Preferenza: {preference}", step_counter, episode)

        total_rewards = torch.zeros(len(preference), dtype=torch.float32).to(device)

        # Riposiziona il robot nelle coordinate iniziali
        robot.step(time_step)  # time_step è 32 ms

        current_robot_pos = translation_field.getSFVec3f()
        robot_rotation = rotation_field.getSFRotation()
        theta = get_yaw_from_webots_rotation(robot_rotation)
        goal_distance, angle_error, dx_goal, dy_goal = compute_goal_metrics(
            current_robot_pos, theta, goal_pos
        )
        ped_pos = ped_translation_field.getSFVec3f()

        # Estrae le caratteristiche iniziali del pedone e le aggiunge alla storia
        # Le caratteristiche spaziali e temporali del pedone sono utili per la RNN
        # spatial_0, temporal_0 = extract_components(
        #     current_robot_pos, ped_pos, prev_ped_pos, prev_robot_pos, theta, prev_theta
        #     # min_front e min_lateral omessi → usano default 1.0
        # )
        # history_spatial.append(spatial_0)
        # history_temporal.append(temporal_0)

        # print(f"history_spatial: {history_spatial}")
        # print(f"history_temporal: {history_temporal}")goal_

        goal_tensor = torch.tensor([[
            dx_goal,
            dy_goal,
            goal_distance / MAX_DISTANCE,
            math.cos(angle_error),
            math.sin(angle_error),
            factor,           # ← aggiungi
            factor_frontal    # ← aggiungi
        ]], dtype=torch.float32).to(device)

        spatial_seq = torch.from_numpy(
            np.array(history_spatial, dtype=np.float32)
        ).unsqueeze(0).to(device)

        temporal_seq = torch.from_numpy(
            np.array(history_temporal, dtype=np.float32)
        ).unsqueeze(0).to(device)

        spatial_seq = torch.clamp(spatial_seq, -1.0, 1.0)
        temporal_seq = torch.clamp(temporal_seq, -1.0, 1.0)

        human_pref = torch.tensor(preference, dtype=torch.float32).unsqueeze(0).to(device)

        # forza shape corretta: [1, 4]
        if human_pref.dim() == 1:
            human_pref = human_pref.unsqueeze(0)

        T = spatial_seq.shape[1]

        # espandi SOLO internamente nel forward
        B, T, _ = spatial_seq.shape

        # goal_exp = goal_tensor.unsqueeze(1).repeat(1, T, 1)
        # pref_exp = human_pref.unsqueeze(1).repeat(1, T, 1)

        # fused = torch.cat([spatial_seq, temporal_seq, goal_exp, pref_exp], dim=-1)
        # print(f"Goal tensor: {goal_tensor}")
        # print(f"spatial_seq: {spatial_seq}")
        # print(f"temporal_seq: {temporal_seq}")
        # print(f"human_pref: {human_pref}")

        # Costruisce lo stato iniziale utilizzando la RNN
        # state = rnn(spatial_seq, temporal_seq, goal_tensor, human_pref)
        # print(f"[RNN] State (primi 5 valori): {state[0, :5].detach().cpu().numpy()}")
        state = rnn(spatial_seq, temporal_seq, goal_tensor, human_pref)

        prev_ped_pos = ped_pos
        prev_goal_dist = goal_distance
        prev_theta = theta
        prev_robot_pos = current_robot_pos

        # --- Loop simulazione ---
        while True:   # avanza simulazione → frame T+1

            # debug_print("\n============================", step_counter, episode)
            # debug_print(f"Step: {step_counter}", step_counter, episode)
            # debug_print("============================", step_counter, episode)
            robot_speak(preference, step_counter)

            stabilize_robot()
            step_counter += 1
            global_step += 1
            decision_step += 1

            # Funzione che implementa la strategia epsilon-greedy per selezionare l'azione da eseguire

            epsilon = decay_epsilon(global_step, agent.epsilon_steps, agent.min_epsilon)
            # ① Seleziona ed esegui azione
            robot_action = select_action(state, preference, episode, step_counter, epsilon, policy_dqn)
            action_counts[robot_action] += 1
            # total_macro_reward = torch.zeros(4, dtype=torch.float32, device=device)
            # gamma_k = 1.0

            # Esegui UNA sola azione per step
            left_speed, right_speed = action_to_speeds(robot_action, factor, factor_frontal)

            apply_action(check_speed(left_speed), check_speed(right_speed))

            macro_done = False
            for _ in range(MACRO_STEP):
                if robot.step(time_step) == -1:
                    return
                # controlla collisioni intermedie per sicurezza
                mid_pos = translation_field.getSFVec3f()
                mid_ranges = read_lidar(lidar)
                mid_col, _, mid_ped_col, _, _, _, _, _, _, mid_done = analyze_environment(
                    mid_ranges, factor, factor_frontal, goal_distance,
                    ped_translation_field.getSFVec3f(), mid_pos, theta
                )
                if mid_done:
                    macro_done = True
                    break

            # if robot.step(time_step) == -1:
            #     return

            for _ in range(MACRO_STEP):
                if robot.step(time_step) == -1:
                    return

            # ② Osservazioni
            current_robot_pos = translation_field.getSFVec3f()
            robot_rotation = rotation_field.getSFRotation()
            theta = get_yaw_from_webots_rotation(robot_rotation)

            goal_distance, angle_error, dx_goal, dy_goal = compute_goal_metrics(current_robot_pos, theta, goal_pos)
            progress = prev_goal_dist - goal_distance
            ped_pos = ped_translation_field.getSFVec3f()

            ranges = read_lidar(lidar)

            collision, goal_reached, ped_collision, ped_distance, near_obstacle, dist, lateral, factor, factor_frontal, done = \
                analyze_environment(ranges, factor, factor_frontal, goal_distance, ped_pos, current_robot_pos, theta)

            spatial, temporal = extract_components(
                current_robot_pos, ped_pos, prev_ped_pos, prev_robot_pos, theta, prev_theta, dist, lateral
            )

            history_spatial.append(spatial)
            history_temporal.append(temporal)

            spatial_seq = torch.from_numpy(
                np.array(history_spatial, dtype=np.float32)
            ).unsqueeze(0).to(device)

            temporal_seq = torch.from_numpy(
                np.array(history_temporal, dtype=np.float32)
            ).unsqueeze(0).to(device)

            spatial_seq = torch.clamp(spatial_seq, -1.0, 1.0)
            temporal_seq = torch.clamp(temporal_seq, -1.0, 1.0)

            human_pref = torch.tensor(preference, dtype=torch.float32).unsqueeze(0).to(device)

            goal_tensor = torch.tensor([[
                dx_goal,
                dy_goal,
                goal_distance / MAX_DISTANCE,
                math.cos(angle_error),
                math.sin(angle_error),
                factor,           # ← aggiungi
                factor_frontal    # ← aggiungi
            ]], dtype=torch.float32).to(device)

            step_reward, done, cross_track_error = get_reward(
                progress, goal_reached, collision, near_obstacle,
                dist, lateral, ped_distance, goal_distance,
                current_robot_pos, ped_pos, theta, goal_pos, episode, done, step_counter, path_start_pos,
                angle_error, prev_dist, prev_ped_dist, ped_collision, robot_action
            )

            # Se il robot è vicino alla traiettoria ideale e ha fatto un certo numero di passi, aggiorna il punto di partenza del path tracking
            if goal_distance < 1.5:
                path_start_pos = list(current_robot_pos)

            prev_robot_pos = current_robot_pos
            prev_ped_pos = ped_pos
            prev_goal_dist = goal_distance
            prev_theta = theta
            prev_ped_dist = ped_distance
            prev_dist = dist

            if done:
                apply_action(0.0, 0.0)

            # print("\n\n\n Fuori for MACRO_STEP: ")
            # Funzione che calcola la reward in base all'analisi dell'ambiente e al progresso verso il goal
            # reward = get_reward(progress, goal_reached, collision, near_obstacle, dist, lateral, ped_distance, goal_distance, current_robot_pos, ped_pos, theta)
            reward = step_reward
            # debug_print(f"total_macro_reward: {total_macro_reward}", step_counter, episode)

            next_goal_tensor = torch.tensor([[
                dx_goal,
                dy_goal,
                goal_distance / MAX_DISTANCE,
                math.cos(angle_error),
                math.sin(angle_error),
                factor,           # ← aggiungi
                factor_frontal    # ← aggiungi
            ]], dtype=torch.float32).to(device)

            # ③ Costruisci next_state con history aggiornata
            next_spatial_seq = torch.from_numpy(
                np.array(history_spatial, dtype=np.float32)
            ).unsqueeze(0).to(device)

            next_temporal_seq = torch.from_numpy(
                np.array(history_temporal, dtype=np.float32)
            ).unsqueeze(0).to(device)
            # next_human_pref = torch.tensor(preference, dtype=torch.float32).unsqueeze(0).to(device)

            # print(f"Goal tensor: {next_goal_tensor}")
            # print(f"Next_spatial_seq: {next_spatial_seq}")
            # print(f"Next_temporal_seq: {next_temporal_seq}")
            # print(f"Next_human_pref: {next_human_pref}")
            # next_goal_tensor = goal_tensor
            # Costruisce lo stato iniziale utilizzando la RNN
            next_state = rnn(next_spatial_seq, next_temporal_seq, next_goal_tensor, human_pref)
            # print next-state
            # print(f"Next state: {next_state}")
            # print(f"[RNN] State (primi 5 valori): {next_state[0, :5].detach().cpu().numpy()}")

            if step_counter % 100 == 0:
                with torch.no_grad():
                    s_std = state.std().item()
                    s_mean = state.mean().item()
                    print(f"[STATE] mean={s_mean:.4f} | std={s_std:.4f}")

            total_rewards = total_rewards + reward

            # ④ Memorizza transizione
            # Salva gli INPUT della RNN invece dello stato già codificato
            memory.append((
                # input RNN
                spatial_seq.detach().cpu(),
                temporal_seq.detach().cpu(),
                goal_tensor.detach().cpu(),
                human_pref.detach().cpu(),
                # input RNN next
                next_spatial_seq.detach().cpu(),
                next_temporal_seq.detach().cpu(),
                next_goal_tensor.detach().cpu(),
                # resto
                robot_action,
                reward.detach().cpu(),
                done,
                preference
            ))
            # =========================
            # TRAIN STEP (DQN + RNN)
            # =========================
            loss = train_step(
                memory,
                agent,
                policy_dqn,
                target_dqn,
                optimizer_value,
                optimizer_advantage,
                optimizer_rnn,
                device,
                global_step,
                step_counter,
                episode,
                debug_print
            )

            if loss is not None and step_counter % 50 == 0:
                debug_print(f"[LOSS] {loss:.4f}", step_counter, episode)

            # =================================================
            # 14. SALVATAGGIO MODELLO
            # =================================================

            # Aggiornamento dei pesi della target network ogni B step
            if global_step % agent.target_update_freq == 0:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                print(f"[TARGET UPDATE] aggiornamento a step {global_step}")

            # scalar_total = total_rewards.sum().item()

            # debug_print(f"total_rewards: {scalar_total}", step_counter, episode)
            # debug_print(f"max_reward: {max_reward}", step_counter, episode)

            # ⑧ Avanza: state diventa next_state
            state = next_state
            spatial_seq = next_spatial_seq
            temporal_seq = next_temporal_seq
            goal_tensor = next_goal_tensor

            # =================================================
            # 15. FINE EPISODIO
            # =================================================
            if done:
                if goal_reached:
                    print("Motivo: Goal raggiunto")
                    goal_reached_count += 1
                elif ped_collision:
                    print("Motivo: Collisione con pedone")
                elif collision:
                    print("Motivo: Collisione con ostacolo statico")
                    print(f"[COLLISION POS] robot={current_robot_pos[:2]}, goal={goal_pos}")

                apply_action(0.0, 0.0)
                robot.step(time_step)
                break

                # for _ in range(int(10000 / time_step)):
                #     robot.step(time_step)
                # break

            if step_counter >= agent.max_timestep:
                # debug_print("Motivo: Goal raggiunto", step_counter, episode)
                # debug_print("Motivo: Collisione con pedone", step_counter, episode)
                # debug_print("Motivo: Collisione con ostacolo statico", step_counter, episode)
                # debug_print("Episodio terminato", step_counter, episode)
                break
        # fine while simulazione

        if episode % 10 == 0:
            total_actions = sum(action_counts)
            dist_str = " | ".join(
                f"{action_names[i]}: {action_counts[i]/total_actions:.1%}"
                for i in range(6)
            )
            print(f"[ACTION DIST ep.{episode}] {dist_str}")
            action_counts = [0] * 6  # reset

        # Salva il miglior modello solo se ha raggiunto il goal
        recent_successes.append(1 if goal_reached else 0)
        success_rate = sum(recent_successes) / len(recent_successes)

        if success_rate >= best_success_rate and goal_reached:
            best_success_rate = success_rate
            print(f"[BEST MODEL] Nuovo best success rate: {success_rate:.2%} (ep {episode})")
            torch.save({
                "policy_dqn": policy_dqn.state_dict(),
                "target_dqn": target_dqn.state_dict(),
                "rnn": rnn.state_dict(),
                "optimizer_value": optimizer_value.state_dict(),
                "optimizer_advantage": optimizer_advantage.state_dict(),
                "optimizer_rnn": optimizer_rnn.state_dict(),
                "episode": episode,
                "global_step": global_step,
                "best_preference": preference,
                "best_ped_speed": current_ped_speed,
                "best_success_rate": best_success_rate,
            }, os.path.join(models_dir, "best_model.pt"))
            # debug_print(f"Nuovo max reward: {max_reward:.2f} all'episodio {episode}", step_counter, episode)
            # debug_print("MODELLO SALVATO!!!", step_counter, episode)

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
                test_pref = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32).to(device)
                test_spatial = torch.zeros(1, agent.steps_len, 3).to(device)
                test_temporal = torch.zeros(1, agent.steps_len, 2).to(device)
                test_goal = torch.tensor([[1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0]], dtype=torch.float32).to(device)
                test_state = rnn(test_spatial, test_temporal, test_goal, test_pref)
                test_q = policy_dqn(test_state)   # [1, A, K]

                # =========================
                # 1. Q per azione (quanto sono separabili le azioni)
                # =========================
                q_per_action_std = test_q.std(dim=2).mean().item()

                # =========================
                # 2. Q per preferenza (quanto la rete usa davvero multi-objective)
                # =========================
                q_per_objective_std = test_q.std(dim=1).mean(dim=0)

                # =========================
                # 3. discriminazione tra azioni (exploration vs collapse)
                # =========================
                q_action_spread = (test_q.max(dim=1).values - test_q.min(dim=1).values).mean().item()

                print("\n========== DIAG ==========")
                print(f"[Q] action spread (discriminazione): {q_action_spread:.6f}")
                print(f"[Q] std per azione: {q_per_action_std:.6f}")
                print(f"[Q] std per obiettivi: {q_per_objective_std.cpu().numpy()}")
                print("==========================\n")
        # =========================
        # SAVE "LAST MODEL"
        # =========================
        torch.save({
            "policy_dqn": policy_dqn.state_dict(),
            "target_dqn": target_dqn.state_dict(),
            "rnn": rnn.state_dict(),
            "optimizer_value": optimizer_value.state_dict(),
            "optimizer_advantage": optimizer_advantage.state_dict(),
            "optimizer_rnn": optimizer_rnn.state_dict(),
            "global_step": global_step,
            "episode": episode,
            "goal_reached_count": goal_reached_count,  # ← aggiungi
        }, os.path.join(models_dir, "last_model.pt"))

        print(f"[EPISODE {episode}] Goal raggiunti: {goal_reached_count}/{episode}")
        print(f"[EPISODE {episode}] Success rate: {goal_reached_count / episode:.2%}")


def generate_preferences_dynamic(episode):
    if episode < 500:
        alpha = [5, 5, 5, 5]
    elif episode < 1500:
        alpha = [1, 1, 1, 1]
    else:
        alpha = [0.3, 0.3, 0.3, 0.3]

    return np.random.dirichlet(alpha).tolist()


def set_pedestrian_speed(speed):
    controller_args_field = pedestrian_node.getField("controllerArgs")
    controller_args_field.setMFString(0, "--trajectory=-1 3, -1 0")
    controller_args_field.setMFString(1, f"--speed={speed}")
    ped_translation_field.setSFVec3f(initial_ped_translation)
    ped_rotation_field.setSFRotation(initial_ped_rotation)
    pedestrian_node.resetPhysics()
    pedestrian_node.restartController()


def run(preference, goal_pos, ped_speed):

    factor = 1.0
    factor_frontal = 1.0
    prev_ped_pos = None
    prev_robot_pos = None
    prev_theta = None

    steps_rnn = agent.steps_len
    step_counter = 0  # ✅ FIX

    set_pedestrian_speed(ped_speed)

    # history coerente col training
    history_spatial = deque(maxlen=steps_rnn)
    history_temporal = deque(maxlen=steps_rnn)

    for _ in range(steps_rnn):
        history_spatial.append([0.0] * 3)
        history_temporal.append([0.0] * 2)

    while robot.step(time_step) != -1:

        step_counter += 1

        current_robot_pos = translation_field.getSFVec3f()
        robot_rotation = rotation_field.getSFRotation()
        theta = get_yaw_from_webots_rotation(robot_rotation)

        goal_distance, angle_error, dx_goal, dy_goal = compute_goal_metrics(
            current_robot_pos, theta, goal_pos
        )

        ped_pos = ped_translation_field.getSFVec3f()
        ranges = read_lidar(lidar)

        collision, goal_reached, ped_collision, ped_distance, near_obstacle, dist, lateral, env_factor, env_factor_frontal, done = \
            analyze_environment(ranges, factor, factor_frontal, goal_distance, ped_pos, current_robot_pos, theta)

        spatial, temporal = extract_components(
            current_robot_pos, ped_pos, prev_ped_pos, prev_robot_pos, theta, prev_theta, dist, lateral
        )

        # aggiorna SEMPRE (come training)
        history_spatial.append(spatial)
        history_temporal.append(temporal)

        # costruzione tensor coerente
        spatial_seq = torch.from_numpy(
            np.array(history_spatial, dtype=np.float32)
        ).unsqueeze(0).to(device)

        temporal_seq = torch.from_numpy(
            np.array(history_temporal, dtype=np.float32)
        ).unsqueeze(0).to(device)

        goal_tensor = torch.tensor([[
            dx_goal,
            dy_goal,
            goal_distance / MAX_DISTANCE,
            math.cos(angle_error),
            math.sin(angle_error)
        ]], dtype=torch.float32).to(device)

        human_pref = torch.tensor(preference, dtype=torch.float32).unsqueeze(0).to(device)

        # forward RNN
        with torch.no_grad():
            state = rnn(spatial_seq, temporal_seq, goal_tensor, human_pref)

        # azione greedy
        robot_action = select_action(
            state,
            preference,
            episode=0,
            step_counter=step_counter,
            epsilon=0.0,
            policy_dqn=policy_dqn
        )

        left_speed, right_speed = action_to_speeds(
            robot_action,
            env_factor,
            env_factor_frontal
        )

        apply_action(left_speed, right_speed)

        # update variabili
        prev_ped_pos = ped_pos
        prev_robot_pos = current_robot_pos
        prev_theta = theta

        if done:
            print("Episode finished")
            apply_action(0.0, 0.0)
            break


# preference_distribution = generate_preferences(50)
# preference_distribution = [
#     [0.5, 0.3, 0.1, 0.1],
#     [0.3, 0.4, 0.2, 0.1],
#     [0.25, 0.25, 0.25, 0.25],
#     [0.2, 0.2, 0.4, 0.2],
#     [0.1, 0.2, 0.1, 0.6],
# ]
# preference_distribution = [
#     [0.5, 0.3, 0.1, 0.1],
#     [0.4, 0.3, 0.2, 0.1],
#     [0.4, 0.2, 0.2, 0.2],
#     [0.35, 0.25, 0.2, 0.2],
#     [0.25, 0.25, 0.25, 0.25],
# ]

goal_pos = [1, 1]

path = os.path.dirname(os.path.abspath(__file__))
# models_dir = "/home/luigi/webots_ws/src/webots_ros2/webots_ros2_tiago/webots_ros2_tiago/models"
models_dir = os.path.join(path, "models")
best_path = os.path.join(models_dir, "best_model.pt")
last_path = os.path.join(models_dir, "last_model.pt")

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
        checkpoint.get("best_preference", [0.25] * 4),
        goal_pos,
        ped_speed=checkpoint.get("best_ped_speed", 0.7)
    )
