import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends.mps import is_available
import gymnasium as gym
import radar_wrapper as radar
import numpy as np
import cv2
from copy import deepcopy

class QNetwork(nn.Module):

    def __init__(self, device=torch.device('cpu'), n_rays=5, len_ray=70, lr=0.0003, gamma=0.95, epsilon_decay=0.95, min_epsilon=0.05):
        super(QNetwork, self).__init__()

        self.device = device
        self.n_rays = n_rays
        self.len_ray = len_ray
        self.lr = lr
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.MAX_SPEED = 70
        self.N_ACTIONS = 5
        self.fc1 = nn.Linear(self.n_rays+1, 64) # rays+sped
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.N_ACTIONS)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.to(device)
        self.env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)

    def forward(self, x):
        # Pass input through layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def training_step(self, state, action, reward, next_state, terminated):
        state_t = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)  # Shape: [1, 5+1]
        action_t = torch.tensor([[action]], dtype=torch.long).to(self.device)  # Shape: [1, 1]
        reward_t = torch.tensor([[reward]], dtype=torch.float32).to(self.device)  # Shape: [1, 1]
        next_state_t = torch.tensor(np.array([next_state]), dtype=torch.float32).to(self.device)  # Shape: [1, 5+1]

        all_q_values = self.forward(state_t)
        chosen_q = all_q_values.gather(1, action_t)

        with torch.no_grad():
            next_q_values = self.forward(next_state_t)
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)

        if terminated:
            target_q = reward_t
        else:
            target_q = reward_t + self.gamma * max_next_q_values

        loss = self.loss_fn(chosen_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def train_agent(self, n_epochs=100, train_seeds=20, val_seeds=5):
        best_val_reward = float('-inf')
        best_model_state = deepcopy(self.state_dict())
        for epoch in range(n_epochs):
            self.train()
            epsilon = max(self.min_epsilon, self.epsilon_decay ** epoch)
            train_reward = 0
            for seed in range(train_seeds):
                obs, info = self.env.reset(seed=seed)
                readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
                speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
                readings = np.append(readings, speed)
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    if np.random.rand() < epsilon:
                        action = np.random.randint(0, self.N_ACTIONS)
                    else:
                        state_tensor = torch.tensor(np.array([readings]), dtype=torch.float32).to(self.device)
                        with torch.no_grad():
                            q_values = self.forward(state_tensor)
                            action = q_values.argmax().item()  # Use torch argmax

                    obs, reward, terminated, truncated, info = self.env.step(action)

                    next_readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
                    next_speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
                    next_readings = np.append(next_readings, next_speed)

                    loss = self.training_step(readings, action, reward, next_readings, terminated)

                    readings = next_readings
                    train_reward += reward

            train_reward /= train_seeds

            self.eval()
            val_reward = 0
            for seed in range(train_seeds, train_seeds+val_seeds):
                obs, info = self.env.reset(seed=seed)
                readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
                speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
                readings = np.append(readings, speed)
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    state_tensor = torch.tensor(np.array([readings]), dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        q_values = self.forward(state_tensor)
                        action = q_values.argmax().item()

                    obs, reward, terminated, truncated, info = self.env.step(action)
                    readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
                    speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
                    readings = np.append(readings, speed)
                    val_reward += reward

                    if epoch % 20 == 0:
                        cv2.imshow("Game", obs)
                        cv2.waitKey(1)

            val_reward /= val_seeds

            print(f"Epoch: {epoch} | Train Reward: {train_reward:.2f}, Val Reward: {val_reward:.2f}")

            if val_reward > best_val_reward:
                best_val_reward = val_reward
                best_model_state = deepcopy(self.state_dict())

            # save game
            if epoch % 20 == 0:
                torch.save(best_model_state, 'qnetwork.pt')

        torch.save(best_model_state, 'qnetwork_final.pt')

    def play(self):
        self.load_state_dict(torch.load("qnetwork.pt", map_location=self.device))
        self.eval()
        self.env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)
        obs, info = self.env.reset()
        readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
        speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
        readings = np.append(readings, speed)
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            state_tensor = torch.tensor(np.array([readings]), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                q_values = self.forward(state_tensor)
                action = q_values.argmax().item()  # Use torch argmax


            obs, reward, terminated, truncated, info = self.env.step(action)
            readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
            speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
            readings = np.append(readings, speed)
            total_reward += reward

            cv2.imshow("Game", obs)
            cv2.waitKey(1)

if is_available():
    device = torch.device("mps")
    print("Success: Using Apple M4 GPU (Metal)")
else:
    device = torch.device("cpu")
    print("Warning: Using CPU")

racing = QNetwork()
racing.train_agent(n_epochs=200)
racing.play()