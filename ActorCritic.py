import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import radar_wrapper as radar
import numpy as np
import cv2
import os
import csv

class ActorCritic(nn.Module):
    LENGTH_RAY = 70
    N_RAYS = 9
    N_ACTIONS = 5
    GAMMA = 0.99

    def __init__(self, device=torch.device('cpu')):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(self.N_RAYS+1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.policy_head = nn.Linear(64, self.N_ACTIONS)
        self.value_head = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0003)
        self.loss_fn = nn.MSELoss()
        self.to(device)
        self.device = device
        self.log_file = open('training_log.csv', 'w', newline='')
        self.logger = csv.writer(self.log_file)
        self.logger.writerow(['epoch', 'reward', 'entropy', 'value_std', 'value_mean', 'return_mean', 'advantage_std'])
        self.entropy_coef = 0.1
        self.entropy_coef_decay = 0.998

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        probs = F.softmax(self.policy_head(x), dim=1)
        value = self.value_head(x).squeeze(1)
        return probs, value

    def training_step(self, states, actions, rewards, dones, final_state):
        if not len(states):
            return

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        probs, values = self.forward(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        returns = torch.zeros_like(rewards)

        if not dones[-1]:
            final_state_tensor = torch.tensor(np.array([final_state]),
                                              dtype=torch.float32, device=self.device)
            with torch.no_grad():
                _, final_value = self.forward(final_state_tensor)
            next_return = final_value.squeeze()
        else:
            next_return = 0.0

        # Calculate returns backwards
        for t in reversed(range(len(rewards))):
            next_return = rewards[t] + self.GAMMA * next_return * (1 - dones[t])
            returns[t] = next_return

        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + 1.0 * critic_loss - self.entropy_coef * entropy
        self.entropy_coef *= self.entropy_coef_decay

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'value_mean': values.mean().item(),
            'value_std': values.std().item(),
            'return_mean': returns.mean().item(),
            'advantage_std': advantages.std().item(),
        }

    def train_agent(self, epochs):
        if os.path.exists("mlp_final.pt"):
            print("Loading existing smart brain...")
            self.load_state_dict(torch.load("mlp_final.pt", map_location=self.device))
        else:
            print("Starting from scratch...")

        env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))
            obs, info = env.reset()
            readings = radar.get_radar_readings(obs, self.N_RAYS, self.LENGTH_RAY) / self.LENGTH_RAY
            raw_speed = env.unwrapped.car.hull.linearVelocity.length
            readings = np.append(readings, raw_speed / 10.0)
            terminated = False
            truncated = False
            total_reward = 0

            states_buffer = []
            actions_buffer = []
            rewards_buffer = []
            dones_buffer = []

            while not (terminated or truncated):
                state_tensor = torch.tensor(np.array([readings]), dtype=torch.float32).to(self.device)
                probs, _ = self.forward(state_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()

                obs, reward, terminated, truncated, info = env.step(action)
                if all(readings[:-1] < 0.1):
                    reward -= 1
                next_readings = radar.get_radar_readings(obs, self.N_RAYS, self.LENGTH_RAY) / self.LENGTH_RAY
                raw_speed = env.unwrapped.car.hull.linearVelocity.length
                next_readings = np.append(next_readings, raw_speed / 10.0)

                states_buffer.append(readings)
                actions_buffer.append(action)
                rewards_buffer.append(reward)
                dones_buffer.append(terminated)

                readings = next_readings
                total_reward += reward

                if epoch % 20 == 0:
                    cv2.imshow("Game", obs)
                    cv2.waitKey(1)

            metrics = self.training_step(states_buffer, actions_buffer, rewards_buffer, dones_buffer, readings)

            if metrics:
                self.logger.writerow([epoch, total_reward, metrics['entropy'],
                                      metrics['value_std'], metrics['value_mean'],
                                      metrics['return_mean'], metrics['advantage_std']])
                self.log_file.flush()

            # save game
            if epoch % 20 == 0 and epoch > 0:
                torch.save(self.state_dict(), 'mlp.pt')

        torch.save(self.state_dict(), 'mlp_final.pt')

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
            probs, _ = self.forward(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            print(action)

            obs, reward, terminated, truncated, info = env.step(action)
            next_readings = radar.get_radar_readings(obs, self.N_RAYS, self.LENGTH_RAY) / self.LENGTH_RAY
            next_readings = np.append(next_readings, env.unwrapped.car.hull.linearVelocity.length / 10)
            readings = next_readings
            total_reward += reward

            cv2.imshow("Game", obs)
            cv2.waitKey(1)
