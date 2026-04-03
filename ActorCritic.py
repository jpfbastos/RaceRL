import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import radar_wrapper as radar
import numpy as np
import cv2
from copy import deepcopy
from ac_nets import ActorNet, CriticNet

class ActorCritic:
    def __init__(self, device=torch.device('cpu'), n_rays=5, len_ray=70, lr=0.0003, gamma=0.95, lam=0.01):
        self.device = device
        self.n_rays = n_rays
        self.len_ray = len_ray
        self.lr = lr
        # how much we should value future rewards
        self.gamma = gamma
        # entropy coefficient - this encourages the policy to have probabilities closer to uniform, thus preventing the
        # agent from locking into a (potentially) suboptimal action too early on.
        self.lam = lam

        self.MAX_SPEED = 70
        self.N_ACTIONS = 5 # nothing, right, left, gas, brake
        self.actor = ActorNet(self.n_rays + 1, self.N_ACTIONS).to(device) # rays+speed
        self.critic = CriticNet(self.n_rays + 1).to(device) # rays+speed

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)

    def training_step(self, state, action, reward, next_state, terminated):
        """
        Perform one update step to both the actor and critic networks with the A2C algorithm

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param terminated:
        :return:
        """
        state_t = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)  # Shape: [1, 5+1]
        next_state_t = torch.tensor(np.array([next_state]), dtype=torch.float32).to(self.device)  # Shape: [1, 5+1]
        action_t = torch.tensor(np.array([action]), dtype=torch.long).to(self.device)

        probs = self.actor(state_t)
        dist = torch.distributions.Categorical(probs)
        # useful to see how random our distribution is, and we encourage some entropy through the loss function
        # for the agent to explore.
        entropy = dist.entropy().mean()

        value = self.critic(state_t)
        next_value = self.critic(next_state_t)
        # produce the TD target and compares it to the current value, 1-terminated makes sure that if we are in a
        # terminal state, that we don't extrapolate into the future, which wouldn't happen in reality as the episode ends
        target = reward + self.gamma * next_value.detach() * (1 - int(terminated))
        advantage = (target - value).detach()

        # actor loss shifts the log probability of actions based on the advantage, and an entropy term is included to
        # encourage exploration
        actor_loss = -dist.log_prob(action_t) * advantage - self.lam * entropy
        critic_loss = self.loss_fn(value, target)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # to prevent too large of an update, which might make learning unstable, we clip the gradients. It is usually
        # the case that we don't reach this threshold (by more than an order of magnitude)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        return entropy, actor_loss.item(), critic_loss.item()

    def train_agent(self, n_epochs, train_seeds=20, val_seeds=5):
        """
        Trains the A2C agent for n_epochs epochs on fixed random seeds, while validating on other fixed random
        seeds. We save the agent with the lowest validation loss every 20 epochs in "a2c_{actor, critic}.pt", and then
        again at the end, under "a2c_{actor, critic}_final.pt".

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
                        action_probs = self.actor(state_tensor)
                        dist = torch.distributions.Categorical(action_probs)
                        action = dist.sample().item()

                    obs, reward, terminated, truncated, info = self.env.step(action)

                    # get normalised readings for radar distances and speed
                    next_readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
                    speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
                    next_readings = np.append(next_readings, speed)

                    loss = self.training_step(readings, action, reward, next_readings, terminated)

                    if epoch % 20 == 0 and seed == 0:
                        cv2.imshow("Train (Actor Critic)", obs)
                        cv2.waitKey(1)

                    readings = next_readings
                    train_reward += reward

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
                        cv2.imshow("Game (Actor Critic)", obs)
                        cv2.waitKey(1)

            val_reward /= val_seeds

            print(f"Epoch: {epoch} | Train Reward: {train_reward:.2f}, Val Reward: {val_reward:.2f}")

            if val_reward > best_val_reward:
                best_val_reward = val_reward
                best_actor_state = deepcopy(self.actor.state_dict())
                best_critic_state = deepcopy(self.critic.state_dict())

            if epoch % 20 == 0 and best_actor_state is not None:
                torch.save(best_actor_state, 'a2c_actor.pt')
                torch.save(best_critic_state, 'a2c_critic.pt')

        torch.save(best_actor_state, 'a2c_actor_final.pt')
        torch.save(best_critic_state, 'a2c_critic_final.pt')

    def play(self, actor_filename='a2c_actor_final.pt', critic_filename='a2c_critic_final.pt'):
        """
        Uses saved Actor and Critic models to play a game of CarRacing.

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

    racing = ActorCritic()
    racing.train_agent(n_epochs=250)
    for _ in range(10):
        racing.play()

"""
For both Actor-Critic methods explained here, I will not make comparisons or reference to graphs as, for both, the 
critic did not converge due to excess variance, so the actor never was able to understand what were "good actions" with
regards to the environment. As seen, both graphs are mostly flat, and a good policy seems to appear by chance rather 
than by an incremental learning process. Nevertheless, I will explain both algorithms and propose solutions to this 
issue, which in the future I'll implement and compare.

Actor-Critic methods work by training two neural nets, instead of one. The first one, the actor, has the task of 
assigning high probabilities to high-reward actions and low probabilities to low-reward actions. This corresponds to 
the role of the neural nets we have been training so far. The new element is the introduction of the critic, is tasked
with predicting the cumulative reward we can expect given a specific state. It can then provide a baseline to the actor, 
from which we can calculate whether the actor's actions outperformed the baseline (good) or if they were suboptimal.
This gap is called the "Advantage". The actor uses the advantage to decide whether to increase or decrease the 
log-likelihood of a certain action given some state.

The explanation above goes through how the A2C algorithm essentially works. Each step the agent takes is followed by a 
training step to the actor/critic networks, which can introduce the high variance problems explained during DQN. As a 
result, the critic was not able to converge on establishing an accurate action-value approximation, which meant that
even if the actor had a very small loss, it is optimising a wrong estimate, so the actions won't produce good results
from the environment's perspective.

A potential solution for this is to use n-step returns rather than 1-step returns, which would allow for less darting 
around, and for the ability to normalise rewards and advantages in a sample, which also stabilises training.  
"""