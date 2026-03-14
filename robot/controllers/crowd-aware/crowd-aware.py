#!/usr/bin/env python3

import os
import math
import torch
import numpy as np
import random
from controller import Supervisor

import TwoStream_RNN as RNN
import DeepQNetwork as DQN


# =========================================================
# MODALITÀ
# =========================================================
MODE = "run"        # "train" oppure "run"
MODEL_PATH = "crowd_model.pt"


# =========================================================
# DEBUG
# =========================================================
DEBUG_PRINT_EVERY = 10


# =========================================================
# COSTANTI
# =========================================================
MAX_SPEED = 6
UNUSED_POINT = 83
N_SECTOR = 5
ROBOT_RADIUS = 0.35

DANGER_DISTANCE = 0.9  # distanza minima per un ostacolo pericoloso
DANGER_LATERAL_DISTANCE = 0.3 # distanza per left e right
MIN_LATERAL_DISTANCE = 0.5

MAX_DISTANCE = 2.5      # distanza massima considerata per normalizzazione
CRUISING_SPEED = 6.0
TURN_SPEED = 2.0

# crowd parameters
NEAR_OBSTACLE_THRESHOLD = 0.3
FAR_OBSTACLE_THRESHOLD = 0.3

# robot parameters
WHEEL_RADIUS = 0.0985      # metri
WHEEL_BASE = 0.404         # metri distanza tra ruote

# reward shaping
GOAL_THRESHOLD = 0.5

TIME_PENALTY = -0.02

# Goal
PROGRESS_GAIN = 5.0
GOAL_REWARD = 30
TRACKING_GOAL_REWARD = -0.05
TRACKING_PROGRESS = -0.02

# Collision
COLLISION_PENALTY = -20
NEAR_PENALTY = -0.01


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
device = torch.device("cpu")

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
            max(min(arm_positions[i-1], motor.getMaxPosition()),
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
    human_pref_dim=3,
    n_robot_actions=5,
    n_human_actions=3
).to(device)


# Inizializza DQN
policy_dqn = DQN.DQN(
    input_dim=195,
    hidden_dim=64,
    n_robot_actions=5,
    n_human_actions=3
)

target_dqn = DQN.DQN(
    input_dim=195,
    hidden_dim=64,
    n_robot_actions=5,
    n_human_actions=3
)


# =========================================================
# SCELTA AZIONE
# =========================================================
def select_action(state, preference, epsilon=0.1, policy_dqn=None):
    print(f"Selezione azione con epsilon={epsilon:.2f}")
    # print("Input - State:", state)
    # =========================
    # EXPLORATION (random robot action)
    # =========================
    if torch.rand(1).item() < epsilon:
        print("Exploration: azione casuale")
        robot_action = torch.randint(0, 5, (1,))
        return robot_action.item()

    # =========================
    # EXPLOITATION (best robot action)
    # =========================
    else:
        print("Exploitation: azione migliore secondo il modello")
        with torch.no_grad():
            q_values = policy_dqn(state)
            print("Q-values (robot x human):", q_values)

            pref = torch.tensor(preference, dtype=torch.float32)

            # → per ogni robot action prende il miglior outcome umano
            robot_q = (q_values * pref).sum(dim=2)
            # shape = [1, 5]

            print("Q-values massimi per ogni azione robot:", robot_q)

            # scegli migliore azione robot
            robot_action = torch.argmax(robot_q, dim=1)

            return robot_action.item()


# Funzione per convertire l'azione discreta in velocità per le ruote
def action_to_speeds(action, factor):

    if action == 0:      # LEFT
        # print("Azione: LEFT")
        return -TURN_SPEED, TURN_SPEED

    elif action == 1:    # FRONT LEFT
        # print("Azione: FRONT LEFT")
        return CRUISING_SPEED * factor, CRUISING_SPEED

    elif action == 2:    # FRONT
        # print("Azione: FRONT")
        return CRUISING_SPEED, CRUISING_SPEED

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
        print("Robot instabile - correzione posizione")
        translation_field.setSFVec3f([pos[0], pos[1], 0.095])
        robot_node.resetPhysics()


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
        value = 1.0 - ranges[i] / max_range

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
        spatial_info[idx] += value
        temporal_info[idx] += radial_velocity

    return spatial_info, temporal_info


# Resetta il robot alla posizione iniziale all'inizio di ogni episodio
def reset_robot():
    translation_field.setSFVec3f(initial_translation)
    rotation_field.setSFRotation(initial_rotation)

    robot_node.resetPhysics()   # azzera velocità
    robot.simulationResetPhysics()   # reset fisica globale
    apply_action(0.0, 0.0)      # ferma ruote


def reset_pedestrian():

    ped_translation_field.setSFVec3f(initial_ped_translation)
    ped_rotation_field.setSFRotation(initial_ped_rotation)

    pedestrian_node.resetPhysics()
    pedestrian_node.restartController()


# Rileva collisioni utilizzando un comando di movimento
def detect_collision_lidar(ranges, factor):

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
        factor = max(0.2, min_dist / DANGER_DISTANCE)
        collision = True
    elif (min_lateral < DANGER_LATERAL_DISTANCE):
        factor = max(0.2, min_lateral / DANGER_LATERAL_DISTANCE)
        collision = True
    elif DANGER_DISTANCE < min_dist < MAX_DISTANCE:
        near_obstacle = True

    return collision, min_dist, factor, near_obstacle


def detect_goal(goal_distance):
    print(f"goal raggiunto: {goal_distance < GOAL_THRESHOLD}")
    goal_reached = goal_distance < GOAL_THRESHOLD

    return goal_reached, goal_distance


def preference_function(reward_vector, base_reward, preference, n_pref):

    for i in range(n_pref):
        reward_vector[i] = base_reward * preference[i]

    return reward_vector


def get_reward(progress, goal_reached, collision, near_obstacle, dist):

    # print("\n--- CALCOLO REWARD ---")

    # =========================
    # GOAL OBJECTIVE
    # =========================
    reward_goal = PROGRESS_GAIN * progress
    reward_goal += TIME_PENALTY

    if goal_reached:
        reward_goal += GOAL_REWARD

    # =========================
    # SAFETY OBJECTIVE
    # =========================
    reward_safety = 0.0

    if collision:
        reward_safety += COLLISION_PENALTY

    elif near_obstacle:
        reward_safety += NEAR_PENALTY

    if dist < MAX_DISTANCE:
        proximity_penalty = (MAX_DISTANCE - dist) / MAX_DISTANCE
        shaping = -0.5 * proximity_penalty
        reward_safety += shaping

    # ??
    if dist > 1.5:
        reward_safety += 0.05

    # =========================
    # PATH OBJECTIVE
    # =========================
    reward_path = path_tracking(dist)

    # ??
    if progress > 0:
        reward_path += 0.02

    if progress < 0:
        reward_path -= 0.05

    # =========================
    # REWARD VECTOR
    # =========================
    reward = torch.tensor([
        reward_goal,
        reward_safety,
        reward_path
    ], dtype=torch.float32)

    print(f"Final reward vector: {reward}\n")

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
    print(f"Posizione robot: x={x:.3f}, y={y:.3f}, theta={theta:.3f} rad")

    # =================================================
    # 2. DISTANZA GOAL + ORIENTAMENTO
    # =================================================

    # Creazione del vettore della direzione del robot
    robot_heading = np.array([math.cos(theta), math.sin(theta)])

    # Creazione vettore della dirazione verso il goal
    goal_vector = np.array([
        goal_pos[0] - x,
        goal_pos[1] - y
    ])

    # Normalizzazione distanza goal
    goal_distance = np.linalg.norm(goal_vector)

    if goal_distance > 1e-6:
        goal_direction = goal_vector / goal_distance
    else:
        goal_direction = np.array([0.0, 0.0])

    # Misura dell'allineamento tra i vettori precedenti
    dot = np.clip(np.dot(robot_heading, goal_direction), -1.0, 1.0)

    # Estrapola il verso della rotazione del robot
    cross = robot_heading[0]*goal_direction[1] - robot_heading[1]*goal_direction[0]

    # Calcolo dell'angolo tra i vettori
    angle_error = math.atan2(cross, dot)

    # turning_radius = goal_direction/(2*math.sin(angle_error))

    print(f"Distanza al goal: {goal_distance:.3f}")
    print(f"Errore angolare: {math.degrees(angle_error):.1f}°")

    return goal_distance, angle_error


def path_tracking(min_dist):

    if min_dist < MAX_DISTANCE:
        tracking_reward = TRACKING_GOAL_REWARD * (min_dist/MAX_DISTANCE)
        print("Reward reason: TRACKING_GOAL")
    else:
        tracking_reward = TRACKING_PROGRESS
        print("Reward reason: TRACKING_PROGRESS")

    return tracking_reward


def decay_epsilon(global_step, epsilon_steps):
    if global_step <= epsilon_steps:
        epsilon = max(0.05, 1.0 - global_step / epsilon_steps)
    else:
        epsilon = 0.05

    return epsilon


def read_lidar(lidar):
    return lidar.getRangeImage()[::-1]


def analyze_environment(ranges, factor, goal_distance):

    collision, min_dist, factor, near_obstacle = detect_collision_lidar(
        ranges,
        factor
    )

    goal_reached, _ = detect_goal(goal_distance)

    done = collision or goal_reached

    return collision, goal_reached, near_obstacle, min_dist, factor, done


# Estrae componente spaziale e temporale dal lidar
def extract_lidar_features(ranges, previous_ranges):

    if previous_ranges is None:
        previous_ranges = ranges.copy()

    spatial_info = [0.0]*5
    temporal_info = [0.0]*5

    spatial_info, temporal_info = check_lidar(
        ranges,
        previous_ranges,
        spatial_info,
        temporal_info
    )

    spatial_info = [v / sector_size for v in spatial_info]
    temporal_info = [v / sector_size for v in temporal_info]

    spatial = torch.tensor([[spatial_info]], dtype=torch.float32)
    temporal = torch.tensor([[temporal_info]], dtype=torch.float32)

    return spatial, temporal


def build_goal_tensor(goal_pos, theta, angle_error):

    return torch.tensor([[
        goal_pos[0],
        goal_pos[1],
        math.cos(theta),
        math.sin(theta),
        angle_error
    ]], dtype=torch.float32)


def build_state(rnn, spatial, temporal, goal_tensor, preference):

    human_pref = torch.tensor([preference], dtype=torch.float32)

    state = rnn(
        spatial,
        temporal,
        goal_tensor,
        human_pref
    )

    return state, human_pref


def planAndTrackPath(
        lidar,
        previous_ranges,
        goal_pos,
        goal_distance,
        theta,
        angle_error,
        progress,
        rnn,
        preference,
        factor):

    ranges = read_lidar(lidar)

    collision, goal_reached, near_obstacle, dist, factor, done = \
        analyze_environment(ranges, factor, goal_distance)

    # Calcola il reward per ogni caso
    reward = get_reward(
        progress,
        goal_reached,
        collision,
        near_obstacle,
        dist
    )

    # Calcolo di componente spaziale e temporale (resi tensori per la RNN)
    spatial, temporal = extract_lidar_features(
        ranges,
        previous_ranges
    )

    # ?? Costruzione del tensore per il goal
    goal_tensor = build_goal_tensor(
        goal_pos,
        theta,
        angle_error
    )

    # Costruzione dello stato utilizzando la RNN
    state, human_pref = build_state(
        rnn,
        spatial,
        temporal,
        goal_tensor,
        preference
    )

    return (
        state,
        reward,
        done,
        collision,
        goal_reached,
        dist,
        ranges,
        factor,
        spatial,
        temporal,
        goal_tensor,
        human_pref
    )


# =========================================================
# TRAINING PARAMETRI
# =========================================================
def train(preference_distribution, goal_pos):
    episode = 0
    factor = 1.0

    # Iperparametri
    discount_factor = 0.99
    memory_maxlen = 50000
    batch_size = 64  # numero di transizioni in un batch
    B = 100  # ogni numero di step in cui aggiornare gli Shadow Parameter
    epsilon_steps = 120000  # Numer odi step in cui la epsilon si riduce da 1 a 0.05
    epsilon = 1.0
    MaxEpisode = 3000

    # Nel paper ci sono massimo 300 step poiché ogni step dura 200 ms (timestep)
    # POichè ongi nostro step dura 32 ms ci saranno massimo 1875 step
    # MaxTimestep = 300
    MaxTimestep = 1875

    max_reward = -float('inf')
    global_step = 0

    # DQN replay buffer (memoria per training)
    memory = DQN.ReplayMemory(memory_maxlen)

    # copia pesi iniziali
    target_dqn.load_state_dict(policy_dqn.state_dict())

    # Ottimizzatore per aggiornare i pesi della policy network
    optimizer = torch.optim.Adam(list(policy_dqn.parameters()) + list(rnn.parameters()),lr=1.25e-4)

    # Creazione di un array di rewards in base al numero di episodi
    rewards = np.zeros(MaxEpisode)

    # Creazione directory per salvare i modelli
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # =========================================================
    # LOOP PRINCIPALE
    # =========================================================
    for episode in range(1, MaxEpisode + 1):

        print("\n============================")
        print("EPISODIO:", episode)
        print("============================")

        total_rewards = 0

        # RIposizionamento del robot nella posizione iniziale
        reset_robot()
        reset_pedestrian()

        # Campiono randomicamente una preferenza lineare
        preference = random.choice(preference_distribution)
        total_rewards = torch.zeros(len(preference), dtype=torch.float32)

        # Riposiziona il robot nelle coordinate iniziali
        # time_Step è 32 ms
        robot.step(time_step)   # stabilizza fisica
        prev_goal_dist = None

        # inizializzazione velocità ruote
        left_speed = 0.0
        right_speed = 0.0

        previous_ranges = None
        step_counter = 0

        # ===============================
        # LOOP SIMULAZIONE EPISODIO
        # ===============================
        while robot.step(time_step) != -1:

            stabilize_robot()

            print("# ===============================")
            print(f"Step: {step_counter}")
            print("# ===============================")

            # Step counter per ogni episodio
            step_counter += 1

            # Step counter utile per ridurre epsilon progressivamente
            global_step += 1

            # Azzeramento del reward per ogni step
            reward = 0.0
            done = False

            # Decay epsilon lineare da 1.0 a 0.05 nei primi 30k step
            epsilon = decay_epsilon(global_step, epsilon_steps)

            # =================================================
            # 1. OSSERVA STATO ATTUALE (posizione + orientamento)
            # =================================================
            current_robot_pos = translation_field.getSFVec3f()
            robot_rotation = rotation_field.getSFRotation()

            # estrazione theta (yaw) da rotazione Webots
            theta = get_yaw_from_webots_rotation(robot_rotation)

            # Funzione per orientare il robot verso il goal
            goal_distance, angle_error = compute_goal_metrics(current_robot_pos, theta, goal_pos)

            # =================================================
            # 3. PROGRESS REALE
            # =================================================
            if prev_goal_dist is None:
                progress = 0.0
            else:
                progress = prev_goal_dist - goal_distance

            prev_goal_dist = goal_distance

            # =================================================
            # 4. PATH PLANNING
            # =================================================
            (
                state, reward, done, collision, goal_reached, min_dist,
                ranges, factor, spatial, temporal, goal_tensor, human_pref
            ) = planAndTrackPath(
                lidar,
                previous_ranges,
                goal_pos,
                goal_distance,
                theta,
                angle_error,
                progress,
                rnn,
                preference,
                factor
            )

            previous_ranges = ranges.copy()
            # print(f"State (RNN output): {state.detach().numpy()}")

            # =================================================
            # 9. ACTION
            # =================================================
            robot_action = select_action(state, preference, epsilon, policy_dqn)

            left_speed, right_speed = action_to_speeds(robot_action, factor)
            left_speed = check_speed(left_speed)
            right_speed = check_speed(right_speed)

            # =================================================
            # 10. ACT (esegui movimento)
            # =================================================
            apply_action(left_speed, right_speed)

            total_rewards += reward

            # =================================================
            # 11. OSSERVA NEXT STATE (dopo movimento)
            # =================================================

            current_robot_pos = translation_field.getSFVec3f()
            robot_rotation = rotation_field.getSFRotation()

            theta = get_yaw_from_webots_rotation(robot_rotation)

            goal_distance, angle_error = compute_goal_metrics(current_robot_pos, theta, goal_pos)

            (
                next_state, _, done, collision, goal_reached, min_dist, ranges,
                factor, next_spatial, next_temporal, next_goal_tensor,
                next_human_pref
            ) = planAndTrackPath(
                lidar,
                previous_ranges,
                goal_pos,
                goal_distance,
                theta,
                angle_error,
                progress,
                rnn,
                preference,
                factor
            )

            # =================================================
            # 12. STORE TRANSITION
            # =================================================
            memory.append((
                spatial.detach(),
                temporal.detach(),
                goal_tensor.detach(),
                human_pref.detach(),
                robot_action,
                reward.detach(),
                next_spatial.detach(),
                next_temporal.detach(),
                next_goal_tensor.detach(),
                done
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
                batch = memory.sample(batch_size)

                # Spacchettamento delle transizioni nelle sue componenti
                (
                    spatial, temporal, goal, human, actions, rewards,
                    next_spatial, next_temporal, next_goal, dones
                ) = zip(*batch)

                # -----------------------------
                # Conversione in Tensor
                # -----------------------------

                # Unisce i campioni del batch
                spatial = torch.cat(spatial).to(device)
                temporal = torch.cat(temporal).to(device)
                goal = torch.cat(goal).to(device)
                human = torch.cat(human).to(device)
                next_spatial = torch.cat(next_spatial).to(device)
                next_temporal = torch.cat(next_temporal).to(device)
                next_goal = torch.cat(next_goal).to(device)

                # Preparazione
                actions = torch.tensor(actions).unsqueeze(1).to(device)
                rewards = torch.stack(rewards).to(device)

                # Annulla bootstrap se episodio finito
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                # Encoding dello stato (RNN)
                state = rnn(spatial, temporal, goal, human)
                next_state = rnn(next_spatial, next_temporal, next_goal, human)

                # -----------------------------
                # TARGET
                # -----------------------------
                # Disabilita il gradiente per velocizzare le operazioni
                with torch.no_grad():

                    q_next = target_dqn(next_state)  # [batch,5,3]

                    pref = human.squeeze(1)  # [batch,3]

                    scalar_q_next = (q_next * pref.unsqueeze(1)).sum(dim=2)

                    next_actions = torch.argmax(scalar_q_next, dim=1)

                    q_next_selected = q_next[range(batch_size), next_actions]

                    # Calcolo equazione di Bellman
                    target_q = rewards + discount_factor * q_next_selected * (1 - dones.unsqueeze(1))

                # -----------------------------
                # CURRENT Q
                # -----------------------------
                q_values = policy_dqn(state)  # [batch,5,3]

                current_q = q_values[range(batch_size), actions.squeeze()]

                # Calcolo della loss
                loss = torch.nn.functional.mse_loss(current_q, target_q)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # =================================================
            # 14. SALVATAGGIO MODELLO
            # =================================================

            # Aggiornamento dei pesi della target network ogni B step
            if global_step % B == 0:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                print(f"Step {global_step}: target network aggiornata")

            scalar_total = total_rewards.sum().item()
            print("total_rewards: ", scalar_total)
            print("max_reward: ", max_reward)

            # Salva il modello migliore eliminando quelli subottimali
            if goal_reached and scalar_total >= max_reward:
                max_reward = scalar_total

                model_path = os.path.join(models_dir, "best_model.pt")

                torch.save(policy_dqn.state_dict(), model_path)
                policy_dqn.to(device)
                print("MODELLO SALVATO!!!")

            # =================================================
            # 15. FINE EPISODIO
            # =================================================
            if done:
                # print("Episodio terminato")

                if collision:
                    print("Motivo: Collisione rilevata")

                elif goal_reached:
                    print("Motivo: Goal raggiunto")

                apply_action(0.0, 0.0)

                for _ in range(int(2000 / time_step)):
                    robot.step(time_step)

                break

            if step_counter > MaxTimestep:
                print("Episodio terminato")
                break


# PER LA FASE DI RUN

def observe_state(lidar, previous_ranges, goal_pos, goal_distance, theta, angle_error, rnn, preference, factor):

    ranges = read_lidar(lidar)

    collision, goal_reached, near_obstacle, dist, factor, done = \
        analyze_environment(ranges, factor, goal_distance)

    spatial, temporal = extract_lidar_features(
        ranges,
        previous_ranges
    )

    goal_tensor = build_goal_tensor(
        goal_pos,
        theta,
        angle_error
    )

    state, human_pref = build_state(
        rnn,
        spatial,
        temporal,
        goal_tensor,
        preference
    )

    return state, collision, goal_reached, near_obstacle, dist, ranges, factor, done


def run(preference, goal_pos):

    previous_ranges = None
    factor = 1.0

    while robot.step(time_step) != -1:

        current_robot_pos = translation_field.getSFVec3f()
        robot_rotation = rotation_field.getSFRotation()
        theta = get_yaw_from_webots_rotation(robot_rotation)

        goal_distance, angle_error = compute_goal_metrics(
            current_robot_pos, theta, goal_pos
        )

        (
            state, collision, goal_reached, near_obstacle, min_dist, ranges,
            factor, done
        ) = observe_state(
            lidar,
            previous_ranges,
            goal_pos,
            goal_distance,
            theta,
            angle_error,
            rnn,
            preference,
            factor
        )

        previous_ranges = ranges.copy()

        robot_action = select_action(state, preference, epsilon=0.0, policy_dqn=policy_dqn)

        left_speed, right_speed = action_to_speeds(robot_action, factor)

        apply_action(left_speed, right_speed)

        if done:
            print("Episode finished")
            apply_action(0.0, 0.0)

            break


# Definizione dell'insieme delle preferenze Ω
# Ogni elemento ω è un vettore di pesi di lunghezza 3 la cui somma è 1 (distribuzione)
# Il valore più alto ωi denota l'obiettivo con importanza maggiore
# Il valore più basso ωi denota l'obiettivo con importanza minore

preference_distribution = [
    [0.6, 0.3, 0.1],
    [0.2, 0.5, 0.3],
    [0.1, 0.2, 0.7],
    [0.4, 0.4, 0.2],
    [0.7, 0.2, 0.1],
    [0.3, 0.6, 0.1],
    [0.25, 0.25, 0.5]
]

goal_pos = [2, 1]  # posizione obiettivo

path = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(path, "models")
model_path = os.path.join(models_dir, "best_model.pt")


empty = True
for _ in os.scandir(models_dir):
    empty = False
    break

if empty:
    print("MODE TRAIN")
    train(preference_distribution, goal_pos)

else:
    print("MODE RUN")

    if os.path.exists(model_path):
        print("Carico modello salvato")
        policy_dqn.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("⚠ Modello non trovato → uso pesi random")

    policy_dqn.eval()

    preference = [0.4, 0.4, 0.2]

    run(preference, goal_pos)
