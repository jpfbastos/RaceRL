import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import radar_wrapper as radar
import numpy as np
import cv2

class QNetwork(nn.Module):
    LENGTH_RAY = 70
    N_RAYS = 5
    N_ACTIONS = 5
    EPSILON_MIN = 0.1  # Minimum exploration (10% random)
    EPSILON_DECAY = 0.995
    GAMMA = 0.95

    def __init__(self, device=torch.device('cpu')):
        super(QNetwork, self).__init__()
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
        return self.fc3(x)

    def training_step(self, state, action, reward, next_state, terminated):
        state_t = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)  # Shape: [1, 5]
        action_t = torch.tensor([[action]], dtype=torch.long).to(self.device)  # Shape: [1, 1]
        reward_t = torch.tensor([[reward]], dtype=torch.float32).to(self.device)  # Shape: [1, 1]
        next_state_t = torch.tensor(np.array([next_state]), dtype=torch.float32).to(self.device)  # Shape: [1, 5]

        all_q_values = self.forward(state_t)
        chosen_q = all_q_values.gather(1, action_t)

        with torch.no_grad():
            next_q_values = self.forward(next_state_t)
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)

        if terminated:
            target_q = reward_t
        else:
            target_q = reward_t + self.GAMMA * max_next_q_values

        loss = self.loss_fn(chosen_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def train_agent(self, epochs):
        epsilon = 1.0
        env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)
        for epoch in range(epochs):
            obs, info = env.reset()
            readings = radar.get_radar_readings(obs, self.N_RAYS, self.LENGTH_RAY) / self.LENGTH_RAY
            terminated = False
            truncated = False
            total_reward = 0

            while not (terminated or truncated):
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, self.N_ACTIONS)
                else:
                    state_tensor = torch.tensor(np.array([readings]), dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        q_values = self.forward(state_tensor)
                        action = q_values.argmax().item()  # Use torch argmax

                obs, reward, terminated, truncated, info = env.step(action)
                next_readings = radar.get_radar_readings(obs, self.N_RAYS, self.LENGTH_RAY) / self.LENGTH_RAY

                loss = self.training_step(readings, action, reward, next_readings, terminated)

                readings = next_readings
                total_reward += reward

                if epoch % 20 == 0:
                    cv2.imshow("Game", obs)
                    cv2.waitKey(1)

            if epsilon > self.EPSILON_MIN:
                epsilon *= self.EPSILON_DECAY

            print(f"Epoch: {epoch} | Reward: {total_reward:.2f}")

            # save game
            if epoch % 100 == 0:
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
            if np.random.rand() < 0.2:
                action = 3
            else:
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