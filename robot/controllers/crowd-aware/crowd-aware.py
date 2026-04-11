#!/usr/bin/env python3

import os
import math
import torch
import numpy as np
import random

from controller import Supervisor
from collections import deque

import TwoStream_RNN as RNN
import DeepQNetwork as DQN

# =========================================================
# MODALITÀ
# =========================================================
MODE = "train"   # "train" | "finetune" | "run"
MODEL_PATH = "crowd_model.pt"


# =========================================================
# DEBUG
# =========================================================
DEBUG_PRINT_EVERY = 10


# =========================================================
# COSTANTI
# =========================================================
MACRO_STEP = 6  # numero di step Webots per ogni azione del robot
MAX_SPEED = 6
UNUSED_POINT = 83
N_SECTOR = 5
ROBOT_RADIUS = 0.35

DANGER_DISTANCE = 0.9  # distanza minima per un ostacolo pericoloso
DANGER_LATERAL_DISTANCE = 0.3  # distanza per left e right
MIN_LATERAL_DISTANCE = 0.5

DANGER_DISTANCE_PEDESTRIAN = 1.1
MAX_DISTANCE_PEDESTRIAN = 3.5

MAX_DISTANCE = 2.5
MAX_LATERAL_DISTANCE = 2.5  # distanza massima considerata per normalizzazione
CRUISING_SPEED = 6.0
TURN_SPEED = 2.0

# crowd parameters
NEAR_OBSTACLE_THRESHOLD = 0.3
FAR_OBSTACLE_THRESHOLD = 0.3

# robot parameters
WHEEL_RADIUS = 0.0985      # metri
WHEEL_BASE = 0.404         # metri distanza tra ruote

# Distanza minima dal goal
GOAL_THRESHOLD = 0.5


# REWARDS
NEUTRAL_PENALTY = -0.05

# TIme
TIME_PENALTY = -0.02

# Goal
PROGRESS_GAIN = -0.02
GOAL_REWARD = 100
TRACKING_GOAL_REWARD = -0.05
TRACKING_PROGRESS = -0.02

# Collision
COLLISION_PEDESTRIAN_PENALTY = -10
COLLISION_PENALTY = -10
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
# ROBOT
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
    spatial_dim=5,
    temporal_dim=5,
    goal_dim=5,
    human_pref_dim=4,
    n_robot_actions=5,
    n_human_actions=4
).to(device)


# Inizializza DQN
policy_dqn = DQN.DQN(
    input_dim=196,
    hidden_dim=64,
    n_robot_actions=5,
    n_human_actions=4
).to(device)

target_dqn = DQN.DQN(
    input_dim=196,
    hidden_dim=64,
    n_robot_actions=5,
    n_human_actions=4
).to(device)


# =========================================================
# SCELTA AZIONE
# =========================================================
def select_action(state, preference, epsilon=0.1, policy_dqn=None):
    print(f"Selezione azione con epsilon={epsilon:.2f}")
    # Lista di stringhe per azione
    action_names = ["LEFT", "FRONT LEFT", "FRONT", "FRONT RIGHT", "RIGHT"]

    # print("Input - State:", state)
    # =========================
    # EXPLORATION (random robot action)
    # =========================
    if torch.rand(1).item() < epsilon:
        # print("Exploration: azione casuale")
        # time.sleep(2.0)

        robot_action = torch.randint(0, 5, (1,))
        print(f"Azione selezionata: {action_names[robot_action.item()]} (random)")
        return robot_action.item()

    # =========================
    # EXPLOITATION (best robot action)
    # =========================
    else:
        # print("Exploitation: azione migliore secondo il modello")
        with torch.no_grad():
            q_values = policy_dqn(state)
            print("\nQ-values:", q_values)
            # print(f"[DQN] Input state shape:    {state.shape}")
            # print(f"[DQN] Q-values shape:       {q_values.shape}")
            # print(f"[DQN] Q-values:\n{q_values.detach().numpy()}")
            # print(f"[DQN] Contiene NaN:         {torch.isnan(q_values).any().item()}")
            # print(f"[DQN] Q-values min/max:     {q_values.min().item():.4f} / {q_values.max().item():.4f}")

            pref = torch.tensor(preference, dtype=torch.float32).to(device)
            print("Preferenza q-network:", pref)

            # → per ogni robot action prende il miglior outcome umano
            robot_q = (q_values * pref).sum(dim=-1)

            print("Q-values con preferenza:", robot_q)

            # scegli migliore azione robot
            robot_action = torch.argmax(robot_q, dim=1)
            print(f"Azione selezionata: {action_names[robot_action.item()]} con Q-value: {robot_q[0, robot_action.item()]:.4f}")

            return robot_action.item()


# Funzione per convertire l'azione discreta in velocità per le ruote
def action_to_speeds(action, factor, factor_frontal):

    if action == 0:      # LEFT
        # print("Azione: LEFT")
        return -TURN_SPEED, TURN_SPEED

    elif action == 1:    # FRONT LEFT
        # print("Azione: FRONT LEFT")
        return CRUISING_SPEED * factor, CRUISING_SPEED

    elif action == 2:    # FRONT
        # print("Azione: FRONT")
        return CRUISING_SPEED * factor_frontal, CRUISING_SPEED * factor_frontal

    elif action == 3:    # FRONT RIGHT
        # print("Azione: FRONT RIGHT")
        return CRUISING_SPEED, CRUISING_SPEED * factor

    elif action == 4:    # RIGHT
        # print("Azione: RIGHT")
        return TURN_SPEED, -TURN_SPEED

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

    print(f"Spatial info: {spatial_info}")
    print(f"Temporal info: {temporal_info}")

    return spatial_info, temporal_info


# Resetta il robot alla posizione iniziale all'inizio di ogni episodio
def reset_robot():
    translation_field.setSFVec3f(initial_translation)
    rotation_field.setSFRotation(initial_rotation)

    robot_node.resetPhysics()   # azzera velocità
    robot.simulationResetPhysics()   # reset fisica globale
    apply_action(0.0, 0.0)      # ferma ruote


def reset_pedestrian():
    new_speed = round(random.uniform(0.3, 1.2), 2)
    # print(f"[ROBOT] Nuova velocità pedestrian: {new_speed}")

    # Cambia gli argomenti del controller prima del restart
    controller_args_field = pedestrian_node.getField("controllerArgs")
    controller_args_field.setMFString(0, "--trajectory=-2 3, -2 0")
    controller_args_field.setMFString(1, f"--speed={new_speed}")

    # Reset posizione
    ped_translation_field.setSFVec3f(initial_ped_translation)
    ped_rotation_field.setSFRotation(initial_ped_rotation)

    pedestrian_node.resetPhysics()
    pedestrian_node.restartController()
    return new_speed


# Rileva collisioni utilizzando un comando di movimento
def detect_collision_lidar(ranges, factor, factor_frontal):

    collision = False
    near_obstacle = False
    # print("min lidar:", min(ranges))
    # print("max lidar:", max(ranges))
    # print("ranges lidar:", ranges)

    n = len(ranges)
    if n == 0:
        return False

    # --- suddivisione in 5 settori ---
    sector_size = n // 5

    left = ranges[0: sector_size]
    front_left = ranges[sector_size: 2 * sector_size]
    front = ranges[2 * sector_size: 3 * sector_size]
    front_right = ranges[3 * sector_size: 4 * sector_size]
    right = ranges[4 * sector_size: n]

    # print("Left:", min(left))
    # print("Front:", min(front))
    # print("Right:", min(right))

    def mean_valid(values):
        valid = [v for v in values if not math.isinf(v) and v > 0.01]
        return sum(valid) / len(valid) if valid else float("inf")

    def median_valid(values):

        valid = [v for v in values if ROBOT_RADIUS < v < MAX_DISTANCE]
        if not valid:
            return MAX_DISTANCE

        return np.median(valid)

    d_left = median_valid(left)
    d_front_left = mean_valid(front_left)
    d_front = mean_valid(front)
    d_front_right = mean_valid(front_right)
    d_right = median_valid(right)

    # print("Distanze settori:")
    # print(f"Left: {d_left:.3f}")
    # print(f"Front-left: {d_front_left:.3f}")
    # print(f"Front: {d_front:.3f}")
    # print(f"Front-right: {d_front_right:.3f}")
    # print(f"Right: {d_right:.3f}")

    min_dist = min(d_front_left, d_front, d_front_right)
    min_lateral = min(d_left, d_right)
    # min_dist = min(d_left, d_front_left, d_front, d_front_right, d_right)
    # max_dist = max(d_front_left, d_front, d_front_right)

    # limita aggressività
    if (min_dist < DANGER_DISTANCE):
        factor_frontal = max(0.2, min_dist / DANGER_DISTANCE)
        collision = True
    elif (min_lateral < DANGER_LATERAL_DISTANCE):
        factor = max(0.2, min_lateral / DANGER_LATERAL_DISTANCE)
        collision = True
    elif DANGER_DISTANCE < min_dist < MAX_DISTANCE:
        near_obstacle = True

    return collision, min_dist, min_lateral, factor, factor_frontal, near_obstacle


def detect_goal(goal_distance):
    # print(f"goal raggiunto: {goal_distance < GOAL_THRESHOLD}")
    goal_reached = goal_distance < GOAL_THRESHOLD

    return goal_reached, goal_distance


def preference_function(reward_vector, base_reward, preference, n_pref):

    for i in range(n_pref):
        reward_vector[i] = base_reward * preference[i]

    return reward_vector


def get_reward(progress, goal_reached, collision, near_obstacle, dist, lateral, ped_distance, goal_distance, robot_pos, ped_pos, theta):

    # print("\n--- CALCOLO REWARD ---")

    # =========================
    # GOAL OBJECTIVE
    # =========================
    reward_goal = 0.0

    progress = max(min(progress, 0.5), -0.5)

    if goal_reached:
        reward_goal = GOAL_REWARD
        # print("Reward reason: GOAL RAGGIUNTO")
    else:
        reward_goal = -1.0 * goal_distance / 10.0

    # if progress > 0:
    #     reward_goal += PROGRESS_GAIN * progress
    #     # print("Reward reason: PROGRESSO VERSO IL GOAL")
    # elif progress < 0:
    #     reward_goal += REGRESS_PENALTY * progress  # penalizza regressione

    # =========================
    # TIME
    # =========================

    # reward_goal = -0.2 * goal_distance - 0.1 * math.tanh(goal_distance)
    # reward_goal += TIME_PENALTY

    # =========================
    # SAFETY OBJECTIVE — PEDESTRIAN
    # =========================
    reward_safety_pedestrian = 0.0

    # Stessa logica di detect_pedestrian_collision:
    # penalizza solo se il pedone è nel semicerchio frontale
    dx = ped_pos[0] - robot_pos[0]
    dy = ped_pos[1] - robot_pos[1]
    dot = dx * math.cos(theta) + dy * math.sin(theta)

    # if dot > 0:  # pedone davanti → penalità attiva
    #     if ped_distance < DANGER_DISTANCE_PEDESTRIAN:
    #         # print("Reward reason: COLLISIONE CON PEDONE!!!")
    #         reward_safety_pedestrian += COLLISION_PEDESTRIAN_PENALTY
    #     elif ped_distance < MAX_DISTANCE_PEDESTRIAN:
    #         # print("Reward reason: PEDONE VICINO - REWARD NEGATIVO")
    #         proximity = (MAX_DISTANCE_PEDESTRIAN - ped_distance) / MAX_DISTANCE_PEDESTRIAN
    #         reward_safety_pedestrian += -0.5 * proximity
    #     # elif ped_distance > MAX_DISTANCE_PEDESTRIAN:
    #     #     print("Reward reason: PEDONE LONTANO - REWARD POSITIVO")
    #     #     reward_safety_pedestrian += 0.05

    if dot > 0:
        if ped_distance is not None:
            if ped_distance < 0.1:
                reward_safety_pedestrian = COLLISION_PEDESTRIAN_PENALTY  # collisione
            elif ped_distance < 0.3:
                reward_safety_pedestrian = NEAR_PENALTY    # zona rischio
            else:
                reward_safety_pedestrian = 0.0
        else:
            reward_safety_pedestrian = 0.0

    # =========================
    # SAFETY OBJECTIVE
    # =========================
    reward_safety_object = 0.0

    # proximity_penalty = (MAX_DISTANCE - dist) / MAX_DISTANCE
    # shaping = -0.5 * proximity_penalty
    # shaping = -1.0 / (dist + 0.1)

    # # collisione con ostacolo o se si è troppo vicini a un ostacolo
    # if collision:
    #     reward_safety_object += COLLISION_PENALTY
    #     # print("Reward reason: COLLISIONE!!!")

    # elif near_obstacle:
    #     reward_safety_object += NEAR_PENALTY
    #     # print("Reward reason: OSTACOLO VICINO!!!")

    #     if dist < MAX_DISTANCE:
    #         reward_safety_object += shaping
    #         # print("Reward reason: OSTACOLO NELLA VICINANZA!!!")
    # else:
    #     if dist < MAX_DISTANCE:
    #         reward_safety_object += shaping
    #         # print("Reward reason: OSTACOLO NELLA VICINANZA!!!")

    if collision:
        reward_safety_object = COLLISION_PENALTY
    else:
        reward_safety_object = 0.0

    # penalità crescente man mano che si avvicina a un ostacolo

    # Sbagliato...
    # Se il robot sorpassa il pedone, assegna un piccolo reward extra
    # if (robot_pos[1] > ped_pos[1] and ped_distance > DANGER_DISTANCE_PEDESTRIAN and abs(robot_pos[0] - (-2.0)) < 1.5):  # robot nella zona del pedone
    #     reward_goal += 0.1
    #     print("Reward reason: PEDONE SUPERATO")

    # =========================
    # PATH OBJECTIVE
    # =========================
    # start_pos = initial_translation
    # reward_path = path_tracking(robot_pos, start_pos, goal_pos)

    start_pos = initial_translation
    cross_track_error = path_tracking(robot_pos, start_pos, goal_pos)
    sigma = 0.3
    c2f = 1.0
    reward_path = -c2f * (1 - math.exp(-(cross_track_error ** 2) / (2 * sigma ** 2)))

    # =========================
    # REWARD VECTOR
    # =========================
    reward = torch.tensor([
        reward_goal,
        reward_safety_object,
        reward_safety_pedestrian,
        reward_path
    ], dtype=torch.float32)

    # print(f"Final reward vector: {reward}\n")

    return reward


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

    # estrazione x, y da posizione
    x, y = robot_pos[0], robot_pos[1]
    # print(f"Posizione robot: x={x:.3f}, y={y:.3f}, theta={theta:.3f} rad")

    # Creazione del vettore della direzione del robot
    robot_heading = np.array([math.cos(theta), math.sin(theta)])

    # Creazione vettore della dirazione verso il goal
    goal_vector = np.array([
        goal_pos[0] - x,
        goal_pos[1] - y
    ])

    # print(f"Vettore verso il goal: {goal_vector}")

    # Normalizzazione distanza goal
    goal_distance = np.linalg.norm(goal_vector)

    # print(f"Distanza al goal normalizzata: {goal_distance:.3f} m")

    if goal_distance > 1e-6:
        goal_direction = goal_vector / goal_distance
    else:
        goal_direction = np.array([0.0, 0.0])

    # Misura dell'allineamento tra i vettori precedenti
    dot = np.clip(np.dot(robot_heading, goal_direction), -1.0, 1.0)

    # Estrapola il verso della rotazione del robot
    cross = robot_heading[0] * goal_direction[1] - robot_heading[1] * goal_direction[0]

    # Calcolo dell'angolo tra i vettori
    angle_error = math.atan2(cross, dot)
    # print(f"Errore angolare: {math.degrees(angle_error):.1f}°")

    # turning_radius = goal_direction/(2*math.sin(angle_error))

    # print(f"Distanza al goal: {goal_distance:.3f}")
    # print(f"Errore angolare: {math.degrees(angle_error):.1f}°")

    return goal_distance, angle_error


def path_tracking(robot_pos, start_pos, goal_pos):
    x, y = robot_pos[0], robot_pos[1]
    x1, y1 = start_pos[0], start_pos[1]
    x2, y2 = goal_pos[0], goal_pos[1]

    num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    den = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    if den < 1e-6:
        return 0.0

    return num / den  # cross-track error in metri, grezzo


def decay_epsilon(global_step, epsilon_steps):
    if global_step <= epsilon_steps:
        epsilon = max(0.05, 1.0 - global_step / epsilon_steps)
    else:
        epsilon = 0.05

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
    heading_x = math.cos(theta)
    heading_y = math.sin(theta)

    # Dot product per capire se il pedone è davanti o dietro
    dot = dx * heading_x + dy * heading_y

    # il pedone è dietro se dot product è negativo, ignora collisione
    if dot < 0:
        return False, ped_distance

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
        factor_frontal
    )

    # Funzione che restituisce un valore booleano se si è arrivati al goal
    goal_reached, _ = detect_goal(goal_distance)

    # Funzione che restituisce un valore booleano se si è troppo vicini al pedone
    ped_collision, ped_distance = detect_pedestrian_collision(robot_pos, ped_pos, theta)

    done = collision or goal_reached or ped_collision

    return collision, goal_reached, ped_collision, ped_distance, near_obstacle, min_dist, min_lateral, factor, factor_frontal, done


# # Estrae componente spaziale e temporale dal lidar
# def extract_lidar_features(ranges, previous_ranges):

#     if previous_ranges is None:
#         previous_ranges = ranges.copy()

#     spatial_info = [0.0] * 5
#     temporal_info = [0.0] * 5

#     spatial_info, temporal_info = check_lidar(
#         ranges,
#         previous_ranges,
#         spatial_info,
#         temporal_info
#     )

#     spatial = [v / sector_size for v in spatial_info]
#     temporal = [v / sector_size for v in temporal_info]

#     return spatial, temporal


def extract_components(robot_pos, ped_pos, prev_ped_pos, theta, min_front=1.0, min_lateral=1.0):

    # Componenti spaziali
    # Calcolo distanza e angolo tra robot e pedone
    dx = ped_pos[0] - robot_pos[0]
    dy = ped_pos[1] - robot_pos[1]
    distance = math.sqrt(dx**2 + dy**2)
    angle = math.atan2(dy, dx) - theta
    # print(f"Posizione robot: x={robot_pos[0]:.3f}, y={robot_pos[1]:.3f}, theta={math.degrees(theta):.1f}°")
    # print(f"Posizione pedone: x={ped_pos[0]:.3f}, y={ped_pos[1]:.3f}")

    if prev_ped_pos is not None:
        # Velocità del pedone (componente temporale)
        vx = (ped_pos[0] - prev_ped_pos[0]) / (time_step / 1000.0)
        vy = (ped_pos[1] - prev_ped_pos[1]) / (time_step / 1000.0)
        speed = math.sqrt(vx**2 + vy**2)
    else:
        vx, vy, speed = 0.0, 0.0, 0.0

    # Normalizzazione
    dx_n = dx / MAX_DISTANCE
    dy_n = dy / MAX_DISTANCE
    distance_n = distance / MAX_DISTANCE
    angle_n = angle / math.pi  # normalizza a [-1, 1]
    speed_n = speed / CRUISING_SPEED
    vx_n = vx / CRUISING_SPEED
    vy_n = vy / CRUISING_SPEED
    min_front_n = min(min_front / MAX_DISTANCE, 1.0)
    min_lateral_n = min(min_lateral / MAX_DISTANCE, 1.0)

    # componente spaziale — dove è il pedone
    spatial = [dx_n, dy_n, distance_n, angle_n, min_front_n]
    print(f"Componente spaziale: {spatial}")

    # componente temporale — come si sta muovendo
    temporal = [vx_n, vy_n, speed_n, min_front_n, min_lateral_n]
    print(f"Componente temporale: {temporal}")

    return spatial, temporal


def build_goal_tensor(goal_pos, theta, angle_error):

    return torch.tensor([[
        goal_pos[0],
        goal_pos[1],
        math.cos(theta),
        math.sin(theta),
        angle_error
    ]], dtype=torch.float32).to(device)


def build_state(rnn, spatial, temporal, goal_tensor, preference):
    human_pref = torch.tensor(preference, dtype=torch.float32).unsqueeze(0).to(device)
    state = rnn(spatial, temporal, goal_tensor, human_pref)

    # print(f"[RNN] Input spatial shape:  {spatial.shape}")
    # print(f"[RNN] Input temporal shape: {temporal.shape}")
    # print(f"[RNN] Input goal shape:     {goal_tensor.shape}")
    # print(f"[RNN] Input pref shape:     {human_pref.shape}")
    # print(f"[RNN] Output state shape:   {state.shape}")
    print(f"[RNN] State (primi 5 valori): {state[0, :5].detach().cpu().numpy()}")
    # print(f"[RNN] Contiene NaN: {torch.isnan(state).any().item()}")

    return state, human_pref


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


def memorize_checkpoint(models_dir, optimizer_dqn, optimizer_rnn, policy_dqn, target_dqn, rnn, global_step, episode):
    # carico checkpoint se esiste, altrimenti inizio da zero
    checkpoint_path = os.path.join(models_dir, "best_model.pt")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        policy_dqn.load_state_dict(checkpoint["policy_dqn"])
        target_dqn.load_state_dict(checkpoint["target_dqn"])
        rnn.load_state_dict(checkpoint["rnn"])

        optimizer_dqn.load_state_dict(checkpoint["optimizer_dqn"])
        optimizer_rnn.load_state_dict(checkpoint["optimizer_rnn"])

        global_step = checkpoint["global_step"]
        start_episode = checkpoint["episode"] + 1

        print(f"Riprendo da episodio {start_episode}")
    else:
        start_episode = 1

    return global_step, start_episode


def check_finetune(mode, optimizer_dqn, optimizer_rnn, policy_dqn, target_dqn, rnn, global_step, episode, models_dir):
    if mode == "finetune":
        global_step, start_episode = memorize_checkpoint(
            models_dir,
            optimizer_dqn,
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

    return global_step, start_episode


DEBUG_PRINT_EVERY = 10  # stampa informazioni di debug ogni N step


# =========================================================
# TRAINING PARAMETRI
# =========================================================
def train(preference_distribution, goal_pos, mode):

    global global_step
    global_step = 0
    episode = 0
    goal_reached_count = 0
    # Iperparametri
    discount_factor = 0.99
    memory_maxlen = 50000
    batch_size = 64  # numero di transizioni in un batch
    M = 32  # preferenze per transizione
    B = 100  # ogni numero di step in cui aggiornare gli Shadow Parameter
    epsilon_steps = 30000  # Numero di step in cui la epsilon si riduce da 1 a 0.05
    epsilon = 1.0
    MaxEpisode = 3000
    start_episode = 1

    # Nel paper ci sono massimo 300 step poiché ogni step dura 200 ms (timestep)
    # Poichè ogni nostro step dura 32 ms ci saranno massimo 1875 step
    # MaxTimestep = 300
    MaxTimestep = 300

    max_reward = -float('inf')

    # DQN replay buffer (memoria per training)
    memory = DQN.ReplayMemory(memory_maxlen)

    # copia pesi iniziali
    target_dqn.load_state_dict(policy_dqn.state_dict())

    # Ottimizzatore per aggiornare i pesi della policy network
    optimizer_dqn = torch.optim.Adam(policy_dqn.parameters(), lr=1e-4)
    optimizer_rnn = torch.optim.Adam(rnn.parameters(), lr=5e-5)

    # Creazione directory per salvare i modelli
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # models_dir = "/home/luigi/webots_ws/src/webots_ros2/webots_ros2_tiago/webots_ros2_tiago/models"
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    global_step, start_episode = check_finetune(
        mode,
        optimizer_dqn,
        optimizer_rnn,
        policy_dqn,
        target_dqn,
        rnn,
        global_step,
        start_episode,
        models_dir
    )

    steps_rnn = 10  # Numero di steps per RNN

    # =========================================================
    # LOOP PRINCIPALE
    # =========================================================
    for episode in range(start_episode, MaxEpisode + 1):

        print("\n============================")
        print("EPISODIO:", episode)
        print("============================")

        prev_ped_pos = None

        # Reset counter
        step_counter = 0

        factor = 1.0
        factor_frontal = 1.0

        total_rewards = 0

        # Inizializza le code per la storia spaziale e temporale da dare alla RNN
        history_spatial = deque(maxlen=steps_rnn)
        history_temporal = deque(maxlen=steps_rnn)

        for _ in range(steps_rnn):
            history_spatial.append([0.0] * 5)
            history_temporal.append([0.0] * 5)

        # Riposizionamento del robot nella posizione iniziale
        reset_robot()
        current_ped_speed = reset_pedestrian()

        # Campiono randomicamente una preferenza lineare
        preference = random.choice(preference_distribution)
        print("Preferenza: ", preference)
        total_rewards = torch.zeros(len(preference), dtype=torch.float32)

        # Riposiziona il robot nelle coordinate iniziali
        # time_Step è 32 ms
        robot.step(time_step)

        current_robot_pos = translation_field.getSFVec3f()
        robot_rotation = rotation_field.getSFRotation()
        theta = get_yaw_from_webots_rotation(robot_rotation)
        goal_distance, angle_error = compute_goal_metrics(current_robot_pos, theta, goal_pos)

        ped_pos = ped_translation_field.getSFVec3f()

        # Estrae le caratteristiche iniziali del pedone e le aggiunge alla storia
        spatial_0, temporal_0 = extract_components(
            current_robot_pos, ped_pos, None, theta
            # min_front e min_lateral omessi → usano default 1.0
        )
        history_spatial.append(spatial_0)
        history_temporal.append(temporal_0)

        # Costruisce il goal tensor e lo stato iniziale da dare alla RNN
        goal_tensor = build_goal_tensor(goal_pos, theta, angle_error)
        spatial_seq = torch.tensor([list(history_spatial)], dtype=torch.float32).to(device)
        temporal_seq = torch.tensor([list(history_temporal)], dtype=torch.float32).to(device)
        state, human_pref = build_state(rnn, spatial_seq, temporal_seq, goal_tensor, preference)

        prev_ped_pos = ped_pos
        # previous_ranges = None
        prev_goal_dist = goal_distance

        # --- Loop simulazione ---
        while True:   # avanza simulazione → frame T+1

            print("\n============================")
            print("Step:", step_counter)
            print("============================")

            stabilize_robot()
            step_counter += 1
            global_step += 1

            # Funzione che implementa la strategia epsilon-greedy per selezionare l'azione da eseguire
            epsilon = decay_epsilon(global_step, epsilon_steps)

            # ① Seleziona ed esegui azione
            robot_action = select_action(state, preference, epsilon, policy_dqn)

            for _ in range(MACRO_STEP):

                left_speed, right_speed = action_to_speeds(robot_action, factor, factor_frontal)
                apply_action(check_speed(left_speed), check_speed(right_speed))

                if robot.step(time_step) == -1:
                    break

                # ② Osservazioni
                current_robot_pos = translation_field.getSFVec3f()
                robot_rotation = rotation_field.getSFRotation()
                theta = get_yaw_from_webots_rotation(robot_rotation)

                goal_distance, angle_error = compute_goal_metrics(current_robot_pos, theta, goal_pos)

                progress = prev_goal_dist - goal_distance if prev_goal_dist else 0.0
                prev_goal_dist = goal_distance

                ped_pos = ped_translation_field.getSFVec3f()

                # Funzione che estrae le informazioni dell'ambiente, calcola la reward e costruisce next_state
                # Legge dati LIDAR e calcola componenti spaziale e temporale
                ranges = read_lidar(lidar)

                # Funzione che analizza l'ambiente con i dati LIDAR e restituisce informazioni utili per reward e done
                collision, goal_reached, ped_collision, ped_distance, near_obstacle, dist, lateral, factor, factor_frontal, done = \
                    analyze_environment(ranges, factor, factor_frontal, goal_distance, ped_pos, current_robot_pos, theta)

                # Funzione che estrae le componenti spaziale e temporale del pedone in base alla sua posizione e movimento
                # Spatial e temporal utili per la RNN a capire dove si trova il pedone e come si sta muovendo
                spatial, temporal = extract_components(current_robot_pos, ped_pos, prev_ped_pos, theta, dist, lateral)

                history_spatial.append(spatial)
                history_temporal.append(temporal)

                spatial_seq = torch.tensor([list(history_spatial)], dtype=torch.float32).to(device)
                temporal_seq = torch.tensor([list(history_temporal)], dtype=torch.float32).to(device)

                # Funzione che applica un reward extra se il robot riesce a superare il pedone
                # reward = pedestrian_left_overcome(robot_pos, ped_pos, reward)

                # previous_ranges = ranges.copy()
                prev_ped_pos = ped_pos

                last_done = done
                last_goal_reached = goal_reached
                last_collision = collision
                last_ped_collision = ped_collision

                if done:
                    apply_action(0.0, 0.0)  # ferma il robot
                    break

            # Funzione che calcola la reward in base all'analisi dell'ambiente e al progresso verso il goal
            reward = get_reward(progress, goal_reached, collision, near_obstacle, dist, lateral, ped_distance, goal_distance, current_robot_pos, ped_pos, theta)

            done = last_done
            goal_reached = last_goal_reached
            collision = last_collision
            ped_collision = last_ped_collision

            # Costruisce il goal tensor e lo stato da dare alla RNN
            goal_tensor = build_goal_tensor(goal_pos, theta, angle_error)

            # ③ Costruisci next_state con history aggiornata
            next_spatial_seq = torch.tensor([list(history_spatial)], dtype=torch.float32).to(device)
            next_temporal_seq = torch.tensor([list(history_temporal)], dtype=torch.float32).to(device)
            next_state, _ = build_state(rnn, next_spatial_seq, next_temporal_seq,
                                        goal_tensor, preference)

            total_rewards += reward

            # ④ Memorizza transizione
            memory.append((
                spatial_seq,  # Stato attuale (componente spaziale)
                temporal_seq,  # Stato attuale (componente temporale)
                goal_tensor,  # Stato attuale (goal)
                human_pref,  # Stato attuale (preferenza umana)
                robot_action,  # Azione eseguita
                reward,  # Reward ottenuta
                next_spatial_seq,  # Prossimo stato (componente spaziale)
                next_temporal_seq,  # Prossimo stato (componente temporale)
                goal_tensor,  # Prossimo stato (goal)
                done  # Episodio terminato
            ))

            # Debug: stampa ultime transizioni
            # print("Ultime 5 transizioni:")
            # for transition in list(memory)[-5:]:
            #     s, a, r, s_next, d = transition
            #     print("Action:", a, "Reward:", r, "Done:", d)

            # =================================================
            # 13. TRAINING
            # =================================================
            if len(memory) >= batch_size:
                # Campiona un batch di transizioni dalla memoria
                batch = memory.sample(batch_size)

                # Spacchettamento delle transizioni nelle sue componenti
                (
                    b_spatial, b_temporal, b_goal, b_human, b_actions, b_rewards,
                    b_next_spatial, b_next_temporal, b_next_goal, b_dones
                ) = zip(*batch)

                # Prima converti le tuple in tensori (come facevi prima)
                b_spatial = torch.cat(b_spatial).to(device)
                b_temporal = torch.cat(b_temporal).to(device)
                b_goal = torch.cat(b_goal).squeeze(1).to(device)
                b_next_spatial = torch.cat(b_next_spatial).to(device)
                b_next_temporal = torch.cat(b_next_temporal).to(device)
                b_next_goal = torch.cat(b_next_goal).squeeze(1).to(device)
                b_actions = torch.tensor(b_actions).unsqueeze(1).to(device)
                b_rewards = torch.stack(b_rewards).to(device)
                b_dones = torch.tensor(b_dones, dtype=torch.float32).to(device)
                b_human = torch.cat(b_human).squeeze(1).to(device)

                # Campiona M preferenze
                m_prefs = torch.tensor(
                    random.choices(preference_distribution, k=M),
                    dtype=torch.float32
                ).to(device)  # [M, 4]

                # Replica ogni transizione M volte
                # [batch_size, ...] → [batch_size*M, ...]
                # b_spatial_rep = b_spatial.repeat_interleave(M, dim=0)
                # b_temporal_rep = b_temporal.repeat_interleave(M, dim=0)
                # b_goal_rep = b_goal.repeat_interleave(M, dim=0)
                # b_next_spatial_rep = b_next_spatial.repeat_interleave(M, dim=0)
                # b_next_temporal_rep = b_next_temporal.repeat_interleave(M, dim=0)
                # b_next_goal_rep = b_next_goal.repeat_interleave(M, dim=0)
                b_actions_rep = b_actions.repeat_interleave(M, dim=0)
                b_rewards_rep = b_rewards.repeat_interleave(M, dim=0)
                b_dones_rep = b_dones.repeat_interleave(M, dim=0)

                # RNN calcolata su 64 transizioni con preferenza reale del batch
                batch_state_base = rnn(b_spatial, b_temporal, b_goal, b_human)
                batch_next_state_base = rnn(b_next_spatial, b_next_temporal, b_next_goal, b_human)

                # Replica l'output della RNN M volte → [64*32, state_dim]
                batch_state = batch_state_base.repeat_interleave(M, dim=0)
                batch_next_state = batch_next_state_base.repeat_interleave(M, dim=0)

                # Replica le preferenze per ogni transizione
                # [M, 4] → [batch_size*M, 4]
                b_human_rep = m_prefs.repeat(batch_size, 1)

                with torch.no_grad():
                    q_next = target_dqn(batch_next_state)
                    scalar_q_next = (q_next * b_human_rep.unsqueeze(1)).sum(dim=2)
                    next_actions = torch.argmax(scalar_q_next, dim=1)
                    q_next_selected = q_next[range(batch_size * M), next_actions]
                    target_q = b_rewards_rep + discount_factor * q_next_selected * (1 - b_dones_rep.unsqueeze(1))

                q_values = policy_dqn(batch_state)
                current_q = q_values[range(batch_size * M), b_actions_rep.squeeze()]

                loss = torch.nn.functional.mse_loss(current_q, target_q)

                optimizer_dqn.zero_grad()
                optimizer_rnn.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=0.5)
                optimizer_dqn.step()
                optimizer_rnn.step()

                print(f"[TRAIN] Loss: {loss.item():.6f}")

                # Controlla che i gradienti siano non nulli (RNN + DQN si stanno aggiornando)
                # for name, param in policy_dqn.named_parameters():
                #     if param.grad is not None:
                #         print(f"[DQN grad] {name}: norm={param.grad.norm().item():.6f}")
                #     else:
                #         print(f"[DQN grad] {name}: NESSUN GRADIENTE")

                # for name, param in rnn.named_parameters():
                #     if param.grad is not None:
                #         print(f"[RNN grad] {name}: norm={param.grad.norm().item():.6f}")
                #     else:
                #         print(f"[RNN grad] {name}: NESSUN GRADIENTE")

            # =================================================
            # 14. SALVATAGGIO MODELLO
            # =================================================

            # Aggiornamento dei pesi della target network ogni B step
            if global_step % B == 0:
                target_dqn.load_state_dict(policy_dqn.state_dict())

                print(f"[TARGET DQN] Pesi copiati al global_step {global_step}")

            scalar_total = total_rewards.sum().item()
            print("total_rewards: ", scalar_total)
            print("max_reward: ", max_reward)

            # Salva il miglior modello solo se ha raggiunto il goal
            if goal_reached and scalar_total >= max_reward:
                max_reward = scalar_total
                torch.save({
                    "policy_dqn": policy_dqn.state_dict(),
                    "target_dqn": target_dqn.state_dict(),
                    "rnn": rnn.state_dict(),

                    "optimizer_dqn": optimizer_dqn.state_dict(),
                    "optimizer_rnn": optimizer_rnn.state_dict(),

                    "episode": episode,
                    "global_step": global_step,

                    "best_preference": preference,
                    "best_ped_speed": current_ped_speed

                    # opzionale:
                    # "memory": memory
                }, os.path.join(models_dir, "best_model.pt"))
                print(f"Nuovo max reward: {max_reward:.2f} all'episodio {episode}")
                print("MODELLO SALVATO!!!")

            # ⑧ Avanza: state diventa next_state
            state = next_state
            spatial_seq = next_spatial_seq
            temporal_seq = next_temporal_seq
            goal_tensor = goal_tensor

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

                apply_action(0.0, 0.0)
                robot.step(time_step)
                break

                # for _ in range(int(10000 / time_step)):
                #     robot.step(time_step)
                # break

            if step_counter >= MaxTimestep:
                print("Episodio terminato")
                break

        # =========================
        # SAVE "LAST MODEL"
        # =========================
        torch.save({
            "policy_dqn": policy_dqn.state_dict(),
            "target_dqn": target_dqn.state_dict(),
            "rnn": rnn.state_dict(),
            "optimizer_dqn": optimizer_dqn.state_dict(),
            "optimizer_rnn": optimizer_rnn.state_dict(),
            "global_step": global_step,
            "episode": episode,
        }, os.path.join(models_dir, "last_model.pt"))

        print(f"[EPISODE {episode}] Goal raggiunti: {goal_reached_count}/{episode}")
        print(f"[EPISODE {episode}] Success rate: {goal_reached_count / episode:.2%}")


def generate_preferences(n_samples=1000, alpha=None):
    """
    Genera n_samples vettori di preferenza (somma = 1)

    alpha controlla la forma:
    - [1,1,1] → uniforme
    - >1 → più centrato
    - <1 → più estremi
    """
    if alpha is None:
        alpha = [1.0, 1.0, 1.0, 1.0]

    preferences = np.random.dirichlet(alpha, size=n_samples)

    return preferences.tolist()


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
    steps_rnn = 10

    set_pedestrian_speed(ped_speed)

    # Inizializza history come nel training
    history_spatial = deque(maxlen=steps_rnn)
    history_temporal = deque(maxlen=steps_rnn)
    for _ in range(steps_rnn):
        history_spatial.append([0.0] * 5)
        history_temporal.append([0.0] * 5)

    while robot.step(time_step) != -1:

        current_robot_pos = translation_field.getSFVec3f()
        robot_rotation = rotation_field.getSFRotation()
        theta = get_yaw_from_webots_rotation(robot_rotation)
        goal_distance, angle_error = compute_goal_metrics(current_robot_pos, theta, goal_pos)
        ped_pos = ped_translation_field.getSFVec3f()

        ranges = read_lidar(lidar)

        collision, goal_reached, ped_collision, ped_distance, near_obstacle, dist, lateral, factor, factor_frontal, done = \
            analyze_environment(ranges, factor, factor_frontal, goal_distance, ped_pos, current_robot_pos, theta)

        spatial, temporal = extract_components(
            current_robot_pos, ped_pos, prev_ped_pos, theta, dist, lateral
        )

        goal_tensor = build_goal_tensor(goal_pos, theta, angle_error)

        # Aggiorna history come nel training
        history_spatial.append(spatial)
        history_temporal.append(temporal)

        spatial_seq = torch.tensor([list(history_spatial)], dtype=torch.float32).to(device)
        temporal_seq = torch.tensor([list(history_temporal)], dtype=torch.float32).to(device)

        state, _ = build_state(rnn, spatial_seq, temporal_seq, goal_tensor, preference)

        prev_ped_pos = ped_pos  # ← anche questo mancava

        robot_action = select_action(state, preference, epsilon=0.0, policy_dqn=policy_dqn)
        left_speed, right_speed = action_to_speeds(robot_action, factor, factor_frontal)
        apply_action(left_speed, right_speed)

        if done:
            print("Episode finished")
            apply_action(0.0, 0.0)
            break


preference_distribution = generate_preferences(500)

goal_pos = [0, 1]

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
    train(preference_distribution, goal_pos, mode=MODE)

elif MODE == "finetune":
    print("MODE FINETUNE")
    train(preference_distribution, goal_pos, mode=MODE)

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
