import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import radar_wrapper as radar
import numpy as np
import cv2
from copy import deepcopy


class REINFORCE(nn.Module):

    def __init__(self, device=torch.device('cpu'), n_rays=5, len_ray=70, lr=0.0003, gamma=0.95):
        super(REINFORCE, self).__init__()

        self.device = device
        self.n_rays = n_rays
        self.len_ray = len_ray
        self.lr = lr
        self.gamma = gamma

        self.MAX_SPEED = 70
        self.N_ACTIONS = 5
        self.fc1 = nn.Linear(self.N_RAYS+1, 64) # rays+sped
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.N_ACTIONS)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.to(self.device)
        self.env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)

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

        discounted_rewards = torch.tensor(discounted_reward, dtype=torch.float32).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        states_tensor = torch.cat(states)
        actions_tensor = torch.stack(actions)

        probs = self.forward(states_tensor)

        chosen_probs = probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        log_probs = torch.log(chosen_probs)

        loss = -(log_probs * discounted_rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_agent(self, epochs, train_seeds=20, val_seeds=5):
        best_val_reward = float('-inf')
        best_model_state = deepcopy(self.state_dict())

        for epoch in range(epochs):
            self.train()
            train_reward = 0
            for seed in range(train_seeds):
                obs, info = self.env.reset(seed=seed)
                readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
                speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
                readings = np.append(readings, speed)
                terminated = False
                truncated = False
                states, actions, rewards = [], [], []

                while not (terminated or truncated):
                    state_tensor = torch.tensor(np.array([readings]), dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        probs = self.forward(state_tensor)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample().item()

                    obs, reward, terminated, truncated, info = self.env.step(action)

                    readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
                    speed = self.env.unwrapped.car.hull.linearVelocity.linearSpeed / self.MAX_SPEED
                    readings = np.append(readings, speed)

                    states.append(state_tensor)
                    actions.append(torch.tensor(action, dtype=torch.int64).to(self.device))
                    rewards.append(reward)

                    train_reward += reward

                    if epoch % 20 == 0:
                        cv2.imshow("Game", obs)
                        cv2.waitKey(1)

                loss = self.training_step(states, actions, rewards)

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
                        probs = self.forward(state_tensor)
                        action = probs.argmax().item()

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
                torch.save(best_model_state, 'mlp.pt')

        torch.save(best_model_state, 'mlp_final.pt')

    def play(self):
        self.load_state_dict(torch.load("mlp_final.pt", map_location=self.device))
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
                probs = self.forward(state_tensor)
                action = probs.argmax().item()

            obs, reward, terminated, truncated, info = self.env.step(action)
            readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
            speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
            readings = np.append(readings, speed)
            total_reward += reward

            cv2.imshow("Game", obs)
            cv2.waitKey(1)
