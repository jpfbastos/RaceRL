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


class REINFORCE(nn.Module):
    LENGTH_RAY = 70
    N_RAYS = 5
    N_ACTIONS = 5
    GAMMA = 0.999
    BETA = 0.01

    def __init__(self, device=torch.device('cpu')):
        super(REINFORCE, self).__init__()
        self.fc1 = nn.Linear(self.N_RAYS, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.N_ACTIONS)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.to(device)
        self.device = device

    def forward(self, x):
        # Pass input through layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

    def training_step(self, states, actions, rewards):

        discounted_reward = np.zeros_like(rewards)
        R = 0
        for i, r in enumerate(reversed(rewards)):
            R = r + self.GAMMA * R
            discounted_reward[len(rewards)-1-i] = R

        returns = torch.tensor(discounted_reward, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        states_tensor = torch.cat(states)
        actions_tensor = torch.stack(actions)

        probs = self.forward(states_tensor)

        chosen_probs = probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        log_probs = torch.log(chosen_probs)

        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        loss = -(log_probs * returns).sum() - self.BETA * entropy.sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_agent(self, epochs):
        if os.path.exists("mlp_final.pt"):
            print("Loading existing smart brain...")
            self.load_state_dict(torch.load("mlp_final.pt", map_location=self.device))
        else:
            print("Starting from scratch...")

        env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)
        for epoch in range(epochs):
            obs, info = env.reset()
            readings = radar.get_radar_readings(obs, self.N_RAYS, self.LENGTH_RAY) / self.LENGTH_RAY
            terminated = False
            truncated = False
            total_reward = 0
            offroad_count = 0
            states, actions, rewards = [], [], []
            while not (terminated or truncated):
                state_tensor = torch.tensor(np.array([readings]), dtype=torch.float32).to(self.device)
                probs = self.forward(state_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                obs, reward, terminated, truncated, info = env.step(action)
                if all(readings < 0.1):
                    offroad_count += 1
                    reward = -5
                    if offroad_count > 20:
                        terminated = True
                else:
                    offroad_count = 0
                next_readings = radar.get_radar_readings(obs, self.N_RAYS, self.LENGTH_RAY) / self.LENGTH_RAY

                states.append(state_tensor)
                actions.append(torch.tensor(action, dtype=torch.int64).to(self.device))
                rewards.append(reward)

                readings = next_readings
                total_reward += reward

                if epoch % 20 == 0:
                    cv2.imshow("Game", obs)
                    cv2.waitKey(1)

            self.training_step(states, actions, rewards)

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
            probs = self.forward(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            print(action)

            obs, reward, terminated, truncated, info = env.step(action)
            next_readings = radar.get_radar_readings(obs, self.N_RAYS, self.LENGTH_RAY)

            readings = next_readings
            readings = readings / self.LENGTH_RAY
            total_reward += reward

            cv2.imshow("Game", obs)
            cv2.waitKey(1)
