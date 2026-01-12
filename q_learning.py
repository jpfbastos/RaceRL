import numpy as np
import gymnasium as gym
import radar_wrapper as radar
import cv2
import matplotlib.pyplot as plt

N_RAYS = 5
LENGTH_RAY = 70
N_BUCKETS = 4
N_STATES = N_BUCKETS ** N_RAYS
N_ACTIONS = 5 # nothing, right, left, gas, brake
LEARNING_RATE = 0.6
DISCOUNT_RATE = 0.95
EPSILON = 0.2
EPOCHS = 1_000_000

Q_table = np.zeros((N_STATES, N_ACTIONS))

env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)

def analog_to_idx(distances):
    # first is least significant digit
    return sum([min((reading // (LENGTH_RAY // N_BUCKETS)), N_BUCKETS-1) * N_BUCKETS ** i for i, reading in enumerate(distances)])


def decode_state(idx, n_rays=5):
    readings = []
    print("\n--- WHAT THE CAR SAW ---")
    print("Left-to-Right")
    print("(0 = Very Close/Wall, 3 = Far Away/Open)")

    for i in range(n_rays):
        bucket = (idx // (4 ** i)) % 4

        # Helper text to visualize
        desc = "UNKNOWN"
        if bucket == 0:
            desc = "WALL (Danger!)"
        elif bucket == 1:
            desc = "Close"
        elif bucket == 2:
            desc = "Medium"
        elif bucket == 3:
            desc = "Open Road"

        print(f"Ray {i + 1}: Bucket {bucket} -> {desc}")
        readings.append(bucket)

    return readings

def train():
    for epoch in range(EPOCHS):
        obs, info = env.reset()
        readings = radar.get_radar_readings(obs, N_RAYS, LENGTH_RAY)
        current_state = analog_to_idx(readings)
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            if np.random.rand() < EPSILON:
                action = np.random.randint(0, N_ACTIONS)
            else:
                action = np.argmax(Q_table[current_state])

            obs, reward, terminated, truncated, info = env.step(action)

            readings = radar.get_radar_readings(obs, N_RAYS, LENGTH_RAY)
            next_state = analog_to_idx(readings)

            if epoch % 100 == 0:
                cv2.imshow("Game", obs)
                cv2.waitKey(1)

            if terminated:
                target_q = reward
            else:
                target_q = reward + DISCOUNT_RATE * np.max(Q_table[next_state])

            Q_table[current_state, action] += LEARNING_RATE * (
                    target_q - Q_table[current_state, action]
            )

            current_state = next_state
            total_reward += reward

        print(f"Epoch: {epoch} | Reward: {total_reward:.2f}")
        if epoch % 100 == 0:
            np.save("q_table_car_racing.npy", Q_table)
    np.save("q_table_car_racing_final.npy", Q_table)

def play():
    Q_table = np.load("q_table_car_racing.npy")
    terminated = False
    truncated = False
    obs, info = env.reset()
    readings = radar.get_radar_readings(obs, N_RAYS, LENGTH_RAY)
    current_state = analog_to_idx(readings)
    while not (terminated or truncated):
        action = np.argmax(Q_table[current_state])
        obs, reward, terminated, truncated, info = env.step(action)
        readings = radar.get_radar_readings(obs, N_RAYS, LENGTH_RAY)
        current_state = analog_to_idx(readings)
        cv2.imshow("Game", obs)
        cv2.waitKey(1)

def display():
    Q_table = np.load("q_table_car_racing.npy")
    print(decode_state(np.argmin(Q_table[:, 1])))
    print(f"Zero values: {Q_table.shape[0]*Q_table.shape[1]-np.count_nonzero(Q_table)} out of {Q_table.shape[0]*Q_table.shape[1]} ({(1-np.count_nonzero(Q_table)/(Q_table.shape[0]*Q_table.shape[1]))*100:.2f}%)")
    plt.figure(figsize=(10, 20))
    plt.imshow(Q_table, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Q-Value')
    plt.xlabel('Actions (0:Wait, 1:Right, 2:Left, 3:Gas, 4:Brake)')
    plt.ylabel('States (0 to 1023)')
    plt.grid(axis='x', color='black', linestyle='-', linewidth=1, alpha=0.3)
    plt.title('Agent Brain (Q-Table)')
    plt.show()

display()