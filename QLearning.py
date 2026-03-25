import numpy as np
import gymnasium as gym
import radar_wrapper as radar
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from copy import deepcopy

class QLearning:
    def __init__(self, n_rays=5, len_ray=70, n_ray_buckets=4, n_speed_buckets=4, lr=0.2, gamma=0.95, epsilon_decay=0.99, min_epsilon=0.05):
        self.n_rays = n_rays
        self.len_ray = len_ray
        self.n_ray_buckets = n_ray_buckets
        self.n_speed_buckets = n_speed_buckets
        self.n_states = self.n_ray_buckets ** (self.n_rays+1) # rays + velocity
        self.learning_rate = lr
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.MAX_SPEED = 70
        self.N_ACTIONS = 5 # nothing, right, left, gas, brake
        self.q_table = defaultdict(lambda: np.zeros(self.N_ACTIONS))

        self.env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)

    def analog_to_idx(self, distances, speed):
        # first is least significant digit
        distance_buckets = tuple(
            min(int(d // (self.len_ray / self.n_ray_buckets)), self.n_ray_buckets - 1)
            for d in distances
        )

        speed_bucket = min(int(min(speed, self.MAX_SPEED) // (self.MAX_SPEED / self.n_speed_buckets)), self.n_speed_buckets - 1)

        return *distance_buckets, speed_bucket


    def decode_state(self, idx, n_rays=5):
        readings = []
        print("\n--- WHAT THE CAR SAW ---")
        print("Left-to-Right")
        print("(0 = Very Close/Wall, 3 = Far Away/Open)")

        for i in range(n_rays):
            bucket = (idx // (self.n_ray_buckets ** i)) % self.n_ray_buckets

            # Helper text to visualize
            desc = "UNKNOWN"
            if bucket == 0:
                desc = "WALL (Danger!)"
            elif bucket <= 1:
                desc = "Close"
            elif bucket == 2:
                desc = "Medium"
            elif bucket == 3:
                desc = "Open Road"

            print(f"Ray {i + 1}: Bucket {bucket} -> {desc}")
            readings.append(bucket)

        return readings

    def train(self, n_epochs=100, train_seeds=20, val_seeds=5):
        best_val_reward = float('inf')
        best_table = self.q_table
        for epoch in range(n_epochs):
            train_reward = 0
            epsilon = max(self.min_epsilon, self.epsilon_decay ** epoch)
            for seed in range(train_seeds):
                obs, info = self.env.reset(seed=seed)
                readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray)
                speed = self.env.unwrapped.car.hull.linearVelocity.length
                current_state = self.analog_to_idx(readings, speed)
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    if np.random.rand() < epsilon:
                        action = np.random.randint(0, self.N_ACTIONS)
                    else:
                        action = np.argmax(self.q_table[current_state])

                    obs, reward, terminated, truncated, info = self.env.step(action)

                    readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray)
                    speed = self.env.unwrapped.car.hull.linearVelocity.length
                    next_state = self.analog_to_idx(readings, speed)

                    if terminated:
                        target_q = reward
                    else:
                        target_q = reward + self.gamma * np.max(self.q_table[next_state])

                    self.q_table[current_state][action] += self.learning_rate * (
                            target_q - self.q_table[current_state][action]
                    )

                    current_state = next_state
                    train_reward += reward

            train_reward /= train_seeds

            val_reward = 0
            for seed in range(val_seeds):
                obs, info = self.env.reset(seed=seed)
                readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
                speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
                readings = np.append(readings, speed)
                terminated = False
                truncated = False

                obs, info = self.env.reset(seed=seed)
                readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray)
                speed = self.env.unwrapped.car.hull.linearVelocity.length
                current_state = self.analog_to_idx(readings, speed)
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    action = np.argmax(self.q_table[current_state])

                    obs, reward, terminated, truncated, info = self.env.step(action)

                    readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray)
                    speed = self.env.unwrapped.car.hull.linearVelocity.length
                    next_state = self.analog_to_idx(readings, speed)

                    current_state = next_state
                    val_reward += reward

                    if epoch % 20 == 0:
                        cv2.imshow("Game", obs)
                        cv2.waitKey(1)

            val_reward /= val_seeds

            print(f"Epoch: {epoch} | Train Reward: {train_reward:.2f}, Val Reward: {val_reward:.2f}")
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                best_table = deepcopy(self.q_table)

            if epoch % 20 == 0:
                with open("q_table.pkl", "wb") as f:
                    pickle.dump(dict(best_table), f)

        with open("q_table_final.pkl", "wb") as f:
            pickle.dump(dict(best_table), f)

    def play(self):
        with open("q_table_final.pkl", "rb") as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.N_ACTIONS), data)
        terminated = False
        truncated = False
        obs, info = self.env.reset()
        readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray)
        speed = self.env.unwrapped.car.hull.linearVelocity.length
        current_state = self.analog_to_idx(readings, speed)
        while not (terminated or truncated):
            action = np.argmax(self.q_table[current_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray)
            speed = self.env.unwrapped.car.hull.linearVelocity.length
            current_state = self.analog_to_idx(readings, speed)
            cv2.imshow("Game", obs)
            cv2.waitKey(1)

    def display(self):
        with open("q_table_final.pkl", "rb") as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.N_ACTIONS), data)

        q_array = np.array(list(self.q_table.values()))  # shape: (visited_states, N_ACTIONS)
        total = q_array.size
        nonzero = np.count_nonzero(q_array)
        print(f"Visited states: {len(self.q_table)}")
        print(f"Zero values: {total - nonzero} out of {total} ({(1 - nonzero / total) * 100:.2f}%)")

        plt.figure(figsize=(10, 20))
        plt.imshow(q_array, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Q-Value')
        plt.xlabel('Actions (0:Wait, 1:Right, 2:Left, 3:Gas, 4:Brake)')
        plt.ylabel('Visited States')
        plt.title('Agent Brain (Q-Table)')
        plt.show()


racing = QLearning()
racing.train(n_epochs=10000)
racing.play()
racing.display()

#TODO Fix 20% number as i want to calculate how many keys the dict has / how many it could have
"""
Having too few buckets reduces granularity of data, whereas having too many buckets makes the data sparse (already at 
20%). The discrete actions, and especially random ones, cause jerky behaviour which goes against the smooth optimal policy
Since entries on the table are independent, it is hard to extrapolate any trends between each entry. This results in
the model not being able to learn general behaviour (e.g. if next to left wall turn right) because it will have to reach
that conclusion in all other ray/speed combinations until it becomes a general rule. 
"""