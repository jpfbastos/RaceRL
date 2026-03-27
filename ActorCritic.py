import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import radar_wrapper as radar
import numpy as np
import cv2
from copy import deepcopy

class ActorCritic(nn.Module):

    def __init__(self, device=torch.device('cpu'), n_rays=5, len_ray=70, lr=0.0003, gamma=0.95):
        super(ActorCritic, self).__init__()

        self.device = device
        self.n_rays = n_rays
        self.len_ray = len_ray
        self.lr = lr
        self.gamma = gamma

        self.MAX_SPEED = 70
        self.N_ACTIONS = 5
        self.fc1 = nn.Linear(self.n_rays+1, 64) # n_rays+speed
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, self.N_ACTIONS)
        self.critic = nn.Linear(64, 1)
        self.loss_fn = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.to(self.device)
        self.env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        probs = F.softmax(self.actor(x), dim=1)
        value = self.critic(x).squeeze(1)
        return probs, value

    def training_step(self, state, action, reward, next_state, terminated):
        state_t = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)  # Shape: [1, 5+1]
        next_state_t = torch.tensor(np.array([next_state]), dtype=torch.float32).to(self.device)  # Shape: [1, 5+1]

        probs, value = self.forward(state_t)
        dist = torch.distributions.Categorical(probs)
        _, next_value = self.forward(next_state_t)

        reward = reward + self.gamma * next_value * (1-terminated)
        advantage = reward - value

        actor_loss = -dist.log_prob(action)*advantage
        critic_loss = self.loss_fn(value, reward)

        self.optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item(),

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

                while not (terminated or truncated):
                    state_tensor = torch.tensor(np.array([readings]), dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        action_probs, _ = self.forward(state_tensor)
                        dist = torch.distributions.Categorical(action_probs)
                        action = dist.sample().item()

                    obs, reward, terminated, truncated, info = self.env.step(action)

                    next_readings = radar.get_radar_readings(obs, self.N_RAYS, self.LENGTH_RAY) / self.LENGTH_RAY
                    speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
                    next_readings = np.append(next_readings, speed)

                    metrics = self.training_step(readings, action, reward, next_readings, terminated)

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
                        probs, _ = self.forward(state_tensor)
                        action = probs.argmax().item()

                    obs, reward, terminated, truncated, info = env.step(action)
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

            if epoch % 20 == 0:
                torch.save(best_model_state, 'a2c.pt')

        torch.save(best_model_state, 'a2c_final.pt')

    def play(self):
        self.load_state_dict(torch.load("mlp_final.pt", map_location=self.device))
        self.eval()
        env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)
        obs, info = env.reset()
        readings = radar.get_radar_readings(obs, self.N_RAYS, self.LENGTH_RAY) / self.LENGTH_RAY
        readings = np.append(readings, env.unwrapped.car.hull.linearVelocity.length / 10.0)
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            state_tensor = torch.tensor(np.array([readings]), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                probs, _ = self.forward(state_tensor)
                action = probs.argmax().item()

            obs, reward, terminated, truncated, info = self.env.step(action)
            readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
            speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
            readings = np.append(readings, speed)
            total_reward += reward

            cv2.imshow("Game", obs)
            cv2.waitKey(1)
