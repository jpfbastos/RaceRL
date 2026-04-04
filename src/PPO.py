import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from src.utils import radar_wrapper as radar
import numpy as np
import cv2
from copy import deepcopy
from src.utils.ac_nets import ActorNet, CriticNet

class PPO:
    def __init__(self, device=torch.device('cpu'), n_rays=5, len_ray=70, lr=3e-4, gamma=0.99, epsilon=0.2, K = 4, batch_size = 256):
        self.device = device

        self.n_rays = n_rays
        self.len_ray = len_ray
        self.lr = lr
        # how much we should value future rewards
        self.gamma = gamma
        # clip parameter - we don't want the policy to be shifting too strongly, so this ensures we keep any updates in
        # a region where we believe training will be stable
        self.epsilon = epsilon
        self.K = K
        self.batch_size = batch_size

        self.MAX_SPEED = 70
        self.N_ACTIONS = 5 # nothing, right, left, gas, brake
        self.actor = ActorNet(self.n_rays+1, self.N_ACTIONS).to(device) # rays+speed
        self.critic = CriticNet(self.n_rays+1).to(device) # rays+speed

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)


    def training_step(self, state_buffer, action_buffer, reward_buffer, done_buffer, old_log_probs, bootstrap_buffer):
        """
        Perform one update step to both the actor and critic networks with the PPO algorithm

        :param state_buffer:
        :param action_buffer:
        :param reward_buffer:
        :param done_buffer:
        :param old_log_probs:
        :param bootstrap_buffer:
        :return:
        """
        state_t = torch.tensor(np.array(state_buffer), dtype=torch.float32).to(self.device)  # Shape: [B, 5+1]
        action_t = torch.tensor(action_buffer, dtype=torch.long).to(self.device)
        reward_t = torch.tensor(reward_buffer, dtype=torch.float32).to(self.device)
        done_t = torch.tensor(done_buffer, dtype=torch.float32).to(self.device)
        old_log_probs_t = torch.stack(old_log_probs).to(self.device).detach()
        bootstrap_t = torch.tensor(bootstrap_buffer, dtype=torch.float32).to(self.device)

        next_return = 0.0
        returns = torch.zeros_like(reward_t)
        for t in reversed(range(len(reward_t))):
            # there are 3 courses of action we can take to consider the potential value of the next action.
            # If the episode has terminated, we don't have to,
            # If it was truncated, then we must use the value of the upcoming state which was never reached,
            # Otherwise, we can use the real action which followed the current one
            if done_t[t] == 1:  # terminated
                next_return = 0.0
            elif done_t[t] == -1: # truncated
                next_return = bootstrap_t[t]
            next_return = reward_t[t] + self.gamma * next_return
            returns[t] = next_return

        dataset_size = state_t.size(0)

        # each PPO training step entails training on minibatches K times, allowing it to train on the same data multiple
        # times instead of throwing it away seeing it once.
        for k in range(self.K):
            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                s_batch = state_t[batch_idx]
                a_batch = action_t[batch_idx]
                old_log_batch = old_log_probs_t[batch_idx]
                returns_batch = returns[batch_idx]

                probs = self.actor(s_batch)
                dist = torch.distributions.Categorical(probs)
                entropy = dist.entropy()
                new_log_probs = dist.log_prob(a_batch)

                value = self.critic(s_batch).squeeze(-1)

                # unlike A2C, we can normalise the advantages because we have a batch of samples
                advantage = returns_batch - value.detach()
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)

                # PPO uses the ratio between the policy after k steps and the initial policy to guide the advantage
                # A larger ratio is good if the advantage was good (the new log probabilities have improved the chance
                # of the good action), and not good if the advantage was negative.
                # I use log probs instead of taking the ratio directly for numerical stability
                ratio = torch.exp(new_log_probs - old_log_batch)

                unclipped = ratio * advantage
                # keeps the ratio bounded
                clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage

                # min creates a pessimistic estimate, as we always assume the improvement is smaller than it might be,
                # making PPO going for the safe option
                actor_loss = -torch.min(unclipped, clipped).mean()

                # like a2c
                critic_loss = self.loss_fn(value, returns_batch)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

        # only returns the last one, but I am more curious at how it eveolves
        return entropy.mean().item(), actor_loss.item(), critic_loss.item()


    def train_agent(self, n_epochs, train_seeds=20, val_seeds=5, seeds_per_update=5):
        """
        Trains the PPO agent for n_epochs epochs on fixed random seeds, while validating on other fixed random
        seeds. We save the agent with the lowest validation loss every 20 epochs in "ppo_{actor, critic}.pt", and then
        again at the end, under "ppo_{actor, critic}_final.pt".

        :param n_epochs:
        :param train_seeds:
        :param val_seeds:
        :return:
        """
        best_val_reward = float('-inf')
        best_actor_state = None
        best_critic_state = None

        for epoch in range(n_epochs):
            self.actor.train()
            self.critic.train()
            train_reward = 0
            state_buffer = []
            action_buffer = []
            reward_buffer = []
            done_buffer = []
            old_log_probs = []
            bootstrap_buffer = []

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
                        probs = self.actor(state_tensor)
                        dist = torch.distributions.Categorical(probs)
                        action = dist.sample()
                        old_log_probs.append(dist.log_prob(action))
                        action = action.item()
                    obs, reward, terminated, truncated, info = self.env.step(action)

                    state_buffer.append(readings)
                    action_buffer.append(action)
                    reward_buffer.append(reward)
                    # useful to differentiate how we should evaluate the next action in training_step
                    if truncated:
                        done = -1
                    elif terminated:
                        done = 1
                    else:
                        done = 0
                    done_buffer.append(done)
                    bootstrap_buffer.append(0.0)

                    readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
                    speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
                    readings = np.append(readings, speed)

                    train_reward += reward

                    if epoch % 20 == 0 and seed == 0:
                        cv2.imshow("Train (PPO)", obs)
                        cv2.waitKey(1)

                if truncated:
                    # if the episode was truncated, we still want to see what the next state would've evaluated to
                    last_state_t = torch.tensor([readings], dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        bootstrap_buffer[-1] = self.critic(last_state_t).item()

                # to keep variety in the training data, but since I don't want a 20k step buffer, I split the training
                # into portions
                if (seed + 1) % seeds_per_update == 0:
                    loss = self.training_step(state_buffer, action_buffer,
                                              reward_buffer, done_buffer, old_log_probs, bootstrap_buffer)

                    state_buffer.clear()
                    action_buffer.clear()
                    reward_buffer.clear()
                    done_buffer.clear()
                    old_log_probs.clear()
                    bootstrap_buffer.clear()

            train_reward /= train_seeds

            self.actor.eval()
            self.critic.eval()
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
                        probs = self.actor(state_tensor)
                        action = probs.argmax().item()

                    obs, reward, terminated, truncated, info = self.env.step(action)
                    readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
                    speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
                    readings = np.append(readings, speed)
                    val_reward += reward

                    if epoch % 20 == 0:
                        cv2.imshow("Val (PPO)", obs)
                        cv2.waitKey(1)

            val_reward /= val_seeds

            print(f"Epoch: {epoch} | Train Reward: {train_reward:.2f}, Val Reward: {val_reward:.2f}")

            if val_reward > best_val_reward:
                best_val_reward = val_reward
                best_actor_state = deepcopy(self.actor.state_dict())
                best_critic_state = deepcopy(self.critic.state_dict())

            if epoch % 20 == 0 and best_actor_state is not None:
                torch.save(best_actor_state, '../models/ppo_actor.pt')
                torch.save(best_critic_state, '../models/ppo_critic.pt')

        torch.save(best_actor_state, '../models/ppo_actor_final.pt')
        torch.save(best_critic_state, '../models/ppo_critic_final.pt')

    def play(self, actor_filename='../models/ppo_actor_final.pt', critic_filename='../models/ppo_critic_final.pt'):
        """
        Uses saved PPO Actor and Critic models to play a game of CarRacing.

        :param actor_filename:
        :param critic_filename:
        :return:
        """
        actor_state = torch.load(actor_filename, map_location=self.device)
        critic_state = torch.load(critic_filename, map_location=self.device)
        self.actor.load_state_dict(actor_state)
        self.critic.load_state_dict(critic_state)
        self.actor.eval()
        self.critic.eval()
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
                probs = self.actor(state_tensor)
                action = probs.argmax().item()

            obs, reward, terminated, truncated, info = self.env.step(action)
            readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
            speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
            readings = np.append(readings, speed)
            total_reward += reward

            cv2.imshow("Game", obs)
            cv2.waitKey(1)

if __name__ == "__main__":
    racing = PPO()
    racing.train_agent(n_epochs=250)
    for _ in range(10):
        racing.play()
