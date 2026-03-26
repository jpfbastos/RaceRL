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
from copy import deepcopy

class QNetworkWithReplay(QNetwork):

    def __init__(self, buffer_size=1000, target_update_freq=10, **kwargs):
        super().__init__(**kwargs)
        self.rb = ReplayBuffer(buffer_size)
        self.target_update_freq = target_update_freq
        self.target_net = deepcopy(self.state_dict())
        self.target_net.to(self.device)
        self.target_net.eval()

    def training_step(self, batch_size=64):
        if len(self.rb) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.rb.sample(batch_size)

        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

        chosen_q = self.forward(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(chosen_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_agent(self, n_epochs=100, train_seeds=20, val_seeds=5, batch_size=64):
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
                        with torch.no_grad():
                            state_t = torch.tensor([readings], dtype=torch.float32).to(self.device)
                            action = self.forward(state_t).argmax().item()

                    obs, reward, terminated, truncated, info = self.env.step(action)
                    next_readings = np.append(radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray,
                                             self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED)

                    self.rb.push(readings, action, reward, next_readings, terminated)
                    self.training_step(batch_size)

                    readings = next_readings
                    train_reward += reward

            train_reward /= train_seeds

            if epoch % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.state_dict())

            self.eval()
            val_reward = 0
            for seed in range(val_seeds):
                obs, info = self.env.reset(seed=seed + train_seeds)  # held-out seeds
                readings = np.append(radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray,
                                     self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED)
                terminated = truncated = False

                while not (terminated or truncated):
                    with torch.no_grad():
                        state_t = torch.tensor([readings], dtype=torch.float32).to(self.device)
                        action = self.forward(state_t).argmax().item()
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    readings = np.append(radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray,
                                         self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED)
                    val_reward += reward

            val_reward /= val_seeds
            print(f"Epoch: {epoch} | Train: {train_reward:.2f} | Val: {val_reward:.2f}")

            if val_reward > best_val_reward:
                best_val_reward = val_reward
                best_model_state = deepcopy(self.state_dict())

            if epoch % 20 == 0 and best_model_state:
                torch.save(best_model_state, 'qnetwork_replay.pt')

        torch.save(best_model_state, 'qnetwork_replay_final.pt')