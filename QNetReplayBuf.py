import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import radar_wrapper as radar
import numpy as np
import cv2
from ReplayBuffer import ReplayBuffer
import os
from QNetwork import QNetwork


class QNetReplayBuf(nn.Module):
    LENGTH_RAY = 70
    N_RAYS = 5
    N_ACTIONS = 5
    EPSILON_MIN = 0.1  # Minimum exploration (10% random)
    EPSILON_DECAY = 0.995
    GAMMA = 0.999
    rb = ReplayBuffer(1000)
    target_net = QNetwork()

    def __init__(self, device=torch.device('cpu')):
        super(QNetReplayBuf, self).__init__()
        self.fc1 = nn.Linear(self.N_RAYS, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.N_ACTIONS)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.to(device)
        self.device = device
        self.target_net.to(device)
        self.target_net.load_state_dict(self.state_dict())
        self.target_net.eval()

    def forward(self, x):
        # Pass input through layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def training_step(self, batch_size):

        if len(self.rb) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.rb.sample(batch_size)

        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)  # Shape: [B, 5]
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)  # Shape: [B, 1]
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)  # Shape: [B, 1]
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)  # Shape: [B, 5]
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)  # Shape: [B, 1]

        all_q_values = self.forward(states_t)
        chosen_q = all_q_values.gather(1, actions_t.unsqueeze(1))
        chosen_q = chosen_q.squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.GAMMA * next_q * (1 - dones_t)

        loss = self.loss_fn(chosen_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_agent(self, epochs, batch_size=64):
        if os.path.exists("mlp_final.pt"):
            print("Loading existing smart brain...")
            self.load_state_dict(torch.load("mlp_final.pt", map_location=self.device))
            epsilon = 0.4
        else:
            print("Starting from scratch...")
            epsilon = 1.0
        env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)
        for epoch in range(epochs):
            obs, info = env.reset()
            readings = radar.get_radar_readings(obs, self.N_RAYS, self.LENGTH_RAY) / self.LENGTH_RAY
            terminated = False
            truncated = False
            total_reward = 0
            offroad_count = 0
            while not (terminated or truncated):
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, self.N_ACTIONS)
                else:
                    state_tensor = torch.tensor(np.array([readings]), dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        q_values = self.forward(state_tensor)
                        action = q_values.argmax().item()  # Use torch argmax

                obs, reward, terminated, truncated, info = env.step(action)
                if all(readings < 0.1):
                    offroad_count += 1
                    reward = -5
                    if offroad_count > 20:
                        terminated = True
                else:
                    offroad_count = 0
                next_readings = radar.get_radar_readings(obs, self.N_RAYS, self.LENGTH_RAY) / self.LENGTH_RAY

                self.rb.push(readings, action, reward, next_readings, terminated)

                loss = self.training_step(batch_size)

                readings = next_readings
                total_reward += reward

                if epoch % 20 == 0:
                    cv2.imshow("Game", obs)
                    cv2.waitKey(1)

            self.target_net.load_state_dict(self.state_dict())

            if epsilon > self.EPSILON_MIN:
                epsilon *= self.EPSILON_DECAY

            print(f"Epoch: {epoch} | Reward: {total_reward:.2f}")

            # save game
            if epoch % 20 == 0 and epoch > 0:
                torch.save(self.state_dict(), 'mlp.pt')

        torch.save(self.state_dict(), 'mlp_final.pt')

    def play(self):
        self.load_state_dict(torch.load("mlp_final.pt", map_location=self.device))
        self.eval()
        env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)
        obs, info = env.reset()
        readings = radar.get_radar_readings(obs, self.N_RAYS, self.LENGTH_RAY)
        readings = readings / self.LENGTH_RAY
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            state_tensor = torch.tensor(np.array([readings]), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                q_values = self.forward(state_tensor)
                action = q_values.argmax().item()  # Use torch argmax
            print(action)
            obs, reward, terminated, truncated, info = env.step(action)
            next_readings = radar.get_radar_readings(obs, self.N_RAYS, self.LENGTH_RAY)

            readings = next_readings
            readings = readings / self.LENGTH_RAY
            total_reward += reward

            cv2.imshow("Game", obs)
            cv2.waitKey(1)
