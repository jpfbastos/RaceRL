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
        # how much we should value future rewards
        self.gamma = gamma

        # determined by me testing by hand. Speed can in truth be higher, but since the idea is to have some sort of
        # normalisation, even if the value is slightly above 1, it doesn't throw off the neural net by much as it is
        # still close to the 0-1 range of the rays.
        self.MAX_SPEED = 70
        self.N_ACTIONS = 5 # nothing, right, left, gas, brake
        self.fc1 = nn.Linear(self.n_rays+1, 64) # rays+speed
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.N_ACTIONS)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.to(self.device)
        self.env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)

    def forward(self, x):
        """
        Pass input through layers to return the probability of the agent performing each action

        :param x:
        :return:
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

    def training_step(self, states, actions, rewards):
        """
        Performs a single update step to the neural network using the REINFORCE algorithm

        :param states:
        :param actions:
        :param rewards:
        :return:
        """
        rewards = np.array(rewards)
        discounted_reward = np.zeros_like(rewards)
        Gt = 0
        # Formula is Gt = Σ_k=t^T(γ^(k-t)R_k), but instead of doing a double for-loop, we can make the process more
        # efficient by taking a running total, and since each step is the current reward plus some future reward times
        # gamma, we can write, Gt = Rt+γ*G(t+1). This means we must start with the last Gt, and propagate the changes
        # backwards, making it O(n) rather than O(n^2).
        for t in reversed(range(len(rewards))):
            Gt = rewards[t] + self.gamma * Gt
            discounted_reward[t] = Gt

        discounted_rewards = torch.tensor(discounted_reward, dtype=torch.float32).to(self.device)
        # normalise rewards for stability
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        states_tensor = torch.cat(states)
        actions_tensor = torch.stack(actions)

        # REINFORCE Loss function is L(θ_t) = -\sum_t=0^T-1[log(π_θ(a_t|s_t)*G_t]
        # The sign is negative because we would like to maximise the reward, but since NN optimisers minimise a loss
        # function, we use loss = - reward as our metric. I take the mean rather than the sum to make sure the rewards
        # are kept at sensible numbers during training, and are invariant with batch size.
        probs = self.forward(states_tensor)
        chosen_probs = probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        log_probs = torch.log(chosen_probs)
        loss = -(log_probs * discounted_rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_agent(self, n_epochs, train_seeds=20, val_seeds=5):
        """
        Trains the REINFORCE agent for n_epochs epochs on fixed random seeds, while validating on other fixed random
        seeds. We save the agent with the lowest validation loss every 20 epochs in "reinforce.pt", and then again at
        the end, under a "reinforce_final.pt".

        :param n_epochs:
        :param train_seeds:
        :param val_seeds:
        :return:
        """
        best_val_reward = float('-inf')
        best_model_state = deepcopy(self.state_dict())

        for epoch in range(n_epochs):
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

                    # get normalised readings for radar distances and speed
                    readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
                    speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
                    readings = np.append(readings, speed)

                    states.append(state_tensor)
                    actions.append(torch.tensor(action, dtype=torch.int64).to(self.device))
                    rewards.append(reward)

                    train_reward += reward

                    if epoch % 20 == 0 and seed == 0:
                        cv2.imshow("Game (train)", obs)
                        cv2.waitKey(1)

                loss = self.training_step(states, actions, rewards)

            train_reward /= train_seeds

            self.eval()
            val_reward = 0
            for seed in range(train_seeds, train_seeds+val_seeds):

                obs, info = self.env.reset(seed=seed)
                # get normalised readings for radar distances and speed
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
                    # get normalised readings for radar distances and speed
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
                torch.save(best_model_state, 'reinforce.pt')

        torch.save(best_model_state, 'reinforce_final.pt')

    def play(self, filename="reinforce_final.pt"):
        """
        Uses a saved REINFORCE model to play a game of CarRacing.

        :param filename:
        :return:
        """
        self.load_state_dict(torch.load(filename, map_location=self.device))
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

        print(f"Total Reward: {total_reward:.2f}")

if __name__ == "__main__":

    racing = REINFORCE()
    racing.train_agent(n_epochs=250)
    for _ in range(10):
        racing.play()

"""
The main difference between REINFORCE and DQN is that REINFORCE is an on-policy algorithm vs DQN which is off-policy.
This means that REINFORCE is updated using data generated from the current policy, which is unlike DQN, which learned 
from data generated from a ε-greedy policy to ensure the agent was exploring. A small alteration I performed during 
validation is that for on-policy algorithms I use the action with the highest probability instead of sampling the 
distribution, but this allows me to observe how confident the agent is in its decisions, although deviating slightly 
from the true on-policy learning.

By sampling actions from its policy distribution, REINFORCE introduces significant variance in the updates,
 as learning is based on complete trajectories that may vary widely in quality. In contrast, DQN improves stability by
using a replay buffer to decorrelate samples and a target network to stabilise the learning target.

REINFORCE relies on sampling from its policy distribution, and as a result has high variance as each training run relies 
on a small amount of steps. DQN, on the other hand, uses a replay buffer and a target network to stabilise the learning 
target. Although this isn't immediately apparent in the reward curve comparison between these two algorithms, it is 
possible this is caused by the REINFORCE agent performs around 4x worse, so the difference scale might be responsible
for this effect. Nevertheless, DQN clearly outperforms REINFORCE agent.
"""