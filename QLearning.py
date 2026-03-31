import numpy as np
import gymnasium as gym
import radar_wrapper as radar
import cv2
from collections import defaultdict
import pickle
from copy import deepcopy

class QLearning:
    def __init__(self, n_rays=5, len_ray=70, n_ray_buckets=4, n_speed_buckets=4, lr=0.2, gamma=0.95, epsilon_decay=0.99, min_epsilon=0.05):
        self.n_rays = n_rays
        self.len_ray = len_ray
        self.n_ray_buckets = n_ray_buckets
        self.n_speed_buckets = n_speed_buckets
        self.n_states = self.n_ray_buckets ** self.n_rays * n_speed_buckets # all rays and velocity measurement

        # how much the current episode should shift the q-value by
        self.learning_rate = lr
        # how much we should value future rewards
        self.gamma = gamma
        # epsilon represents the probability of choosing a random action vs the one with the highest q-value
        # given the state. We decay this over time because we initially want to let the model explore different options
        # but as it improves, it is better for the model to finetune on the best actions (exploit rather than explore)
        # so we want a low epsilon
        self.epsilon_decay = epsilon_decay
        # but still have some degree of exploration so we can keep improving
        self.min_epsilon = min_epsilon

        # determined by me testing by hand. Speed can in truth be higher, but this is quite fast, so as far as buckets
        # are concerned, any speed above 70 can just be clipped to 70 to make sure we fit in the bucket
        self.MAX_SPEED = 70
        self.N_ACTIONS = 5 # nothing, right, left, gas, brake
        self.q_table = defaultdict(lambda: np.zeros(self.N_ACTIONS))

        self.env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)

    def analog_to_idx(self, distances, speed):
        """
        Converts raw distances and speed into a tuple with each value corresponding to the bucket of the rays and speed

        :param distances:
        :param speed:
        :return:
        """
        distance_buckets = tuple(
            min(int(d // (self.len_ray / self.n_ray_buckets)), self.n_ray_buckets - 1)
            for d in distances
        )

        speed_bucket = min(int(min(speed, self.MAX_SPEED) // (self.MAX_SPEED / self.n_speed_buckets)), self.n_speed_buckets - 1)

        return *distance_buckets, speed_bucket

    def train(self, n_epochs=100, train_seeds=20, val_seeds=5):
        """
        Trains the Q-learning agent on fixed random seeds, and validates on other fixed random seeds. We save the agent
        with the lowest validation loss every 20 epochs in "q_table.pkl", and then again at the end, under a
        "q_table_final.pkl".

        :param n_epochs:
        :param train_seeds:
        :param val_seeds:
        :return:
        """
        best_val_reward = float('-inf')
        best_table = deepcopy(self.q_table)
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

                    # if the episode is terminated, then there is no future reward, so that becomes our aim, but if
                    # there is, then we also want to weigh what the next state would be to ensure we not only consider
                    # immediate rewards, but also if by chasing that immediate reward we can put ourselves in a bad
                    # position for the future
                    if terminated:
                        target_q = reward
                    else:
                        target_q = reward + self.gamma * np.max(self.q_table[next_state])

                    # Q(s,a) = Q(s,a) + α[r + γ * max_a'(Q(s',a')) - Q(s,a)]
                    self.q_table[current_state][action] += self.learning_rate * (
                            target_q - self.q_table[current_state][action]
                    )

                    current_state = next_state
                    train_reward += reward

            train_reward /= train_seeds

            val_reward = 0
            for seed in range(train_seeds, train_seeds+val_seeds):
                obs, info = self.env.reset(seed=seed)
                readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray)
                speed = self.env.unwrapped.car.hull.linearVelocity.length
                current_state = self.analog_to_idx(readings, speed)
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    # here we don't use epsilon to test how our agent is building up its policy
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

    def play(self, filename="q_table_final.pkl"):
        """
        Plays a game of Q-learning on a saved Q-table.

        :param filename:
        :return:
        """
        with open(filename, "rb") as f:
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

    def frac_used(self):
        with open("q_table_final.pkl", "rb") as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.N_ACTIONS), data)

        states = list(self.q_table.keys())

        # assume all tuples have same length
        dim = len(states[0])

        max_vals = [0] * dim

        for s in states:
            for i in range(dim):
                if s[i] > max_vals[i]:
                    max_vals[i] = s[i]

        print("Max per dimension:", max_vals)

        total_states = np.prod([m + 1 for m in max_vals])
        print("Estimated total possible states:", total_states)
        print(f"Actual number of visited states {len(self.q_table)},"
              f" {np.round(len(self.q_table)/total_states*100, 2)}% of possible states")


if __name__ == "__main__":
    racing = QLearning()
    racing.train(n_epochs=300)
    for _ in range(10):
        racing.play("q_table_final.pkl")
    racing.frac_used()

"""
Having too few buckets reduces granularity of data, whereas having too many buckets makes the data sparse, meaning the 
q-values are dictated by only a few sample points (curse of dimensionality). We save values in a dictionary, but if we 
were to save them in a table, we'd only use 29.63% of entries (where the size of each dimension is zero to the largest 
numbered bucket observed - buckets not in the data are not counted, so if using all buckets in truth this percentage
would be even lower). 

Since entries on the table are independent, it is hard to extrapolate any trends between each entry. This results in
the model not being able to generalise behaviour across the policy (e.g. if next to left wall turn right) because it
would have to reach that conclusion in all other ray/speed combinations until it becomes a general rule. Therefore, 
we see a highly oscillating pattern as readings transition from one set of bucket to the next while moving.
"""
