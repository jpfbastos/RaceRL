import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import radar_wrapper as radar
import numpy as np
import cv2
from copy import deepcopy

class QNetwork(nn.Module):
    def __init__(self, device=torch.device('cpu'), n_rays=5, len_ray=70, lr=0.0003, gamma=0.95, epsilon_decay=0.95, min_epsilon=0.005):
        super(QNetwork, self).__init__()

        self.device = device
        self.n_rays = n_rays
        self.len_ray = len_ray
        self.lr = lr
        # how much we should value future rewards
        self.gamma = gamma
        # epsilon represents the probability of choosing a random action vs the one with the highest q-value
        # given the state. We decay this over time because we initially want to let the model explore different options
        # but as it improves, it is better for the model to finetune on the best actions (exploit rather than explore)
        # so we want a low epsilon
        self.epsilon_decay = epsilon_decay
        # but still have some degree of exploration so we can keep improving
        self.min_epsilon = min_epsilon

        # determined by me testing by hand. Speed can in truth be higher, but this is quite fast, so as far as buckets
        # are concerned, any speed above 70 can just be clipped to 70 to make sure we fit in the bucket
        self.MAX_SPEED = 70
        self.N_ACTIONS = 5 # nothing, right, left, gas, brake
        self.fc1 = nn.Linear(self.n_rays+1, 64) # rays+sped
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.N_ACTIONS)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.to(self.device)
        self.env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)

    def forward(self, x):
        """
        Pass input through layers with ReLU activation

        :param x:
        :return:
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def training_step(self, state, action, reward, next_state, terminated):
        """
        Performs a single update step to the Q-Network

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param terminated:
        :return:
        """
        state_t = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)  # Shape: [1, 5+1]
        action_t = torch.tensor([[action]], dtype=torch.long).to(self.device)  # Shape: [1, 1]
        reward_t = torch.tensor([[reward]], dtype=torch.float32).to(self.device)  # Shape: [1, 1]
        next_state_t = torch.tensor(np.array([next_state]), dtype=torch.float32).to(self.device)  # Shape: [1, 5+1]

        all_q_values = self.forward(state_t)
        # for each row (step), we fetch the q-value corresponding to the action taken
        chosen_q = all_q_values.gather(1, action_t)

        # if the episode is terminated, then there is no future reward, so that becomes our aim, but if
        # there is, then we also want to weigh what the next state would be to ensure we not only consider
        # immediate rewards, but also if by chasing that immediate reward we can put ourselves in a bad
        # position for the future
        if terminated:
            target_q = reward_t
        else:
            with torch.no_grad():
                next_q_values = self.forward(next_state_t)
                max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            target_q = reward_t + self.gamma * max_next_q_values

        # We're using a modification of the Q-Learning Update Rule since we are now working with continuous values, so we
        # can't update a table directly. We use mean squared error to make sure that the q-values produced by the
        # network match the expected returns.
        loss = self.loss_fn(chosen_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def train_agent(self, n_epochs=100, train_seeds=20, val_seeds=5):
        """
        Trains the Q-Network on fixed random seeds, and validates on other fixed random seeds. We save the agent
        with the lowest validation loss every 20 epochs in "qnetwork.pt", and then again at the end, under a
        "qnetwork_final.pt".

        :param n_epochs:
        :param train_seeds:
        :param val_seeds:
        :return:
        """
        best_val_reward = float('-inf')
        best_model_state = deepcopy(self.state_dict())
        for epoch in range(n_epochs):
            self.train()
            # decay epsilon
            epsilon = max(self.min_epsilon, self.epsilon_decay ** epoch)
            train_reward = 0
            for seed in range(train_seeds):
                obs, info = self.env.reset(seed=seed)
                # get normalised readings for radar distances and speed
                readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
                speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
                readings = np.append(readings, speed)
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    if np.random.rand() < epsilon:
                        action = np.random.randint(0, self.N_ACTIONS)
                    else:
                        state_tensor = torch.tensor(np.array([readings]), dtype=torch.float32).to(self.device)
                        with torch.no_grad():
                            q_values = self.forward(state_tensor)
                            action = q_values.argmax().item()

                    obs, reward, terminated, truncated, info = self.env.step(action)

                    # get normalised readings for radar distances and speed
                    next_readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
                    next_speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
                    next_readings = np.append(next_readings, next_speed)

                    loss = self.training_step(readings, action, reward, next_readings, terminated)

                    readings = next_readings
                    train_reward += reward

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
                    # here we don't use epsilon to test how our agent is building up its policy
                    with torch.no_grad():
                        q_values = self.forward(state_tensor)
                        action = q_values.argmax().item()

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

            if epoch % 20 == 0:
                torch.save(best_model_state, 'qnetwork.pt')

        torch.save(best_model_state, 'qnetwork_final.pt')

    def play(self, filename="qnetwork_final.pt"):
        """
        Uses a saved Q-network to play a game of CarRacing.

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
                q_values = self.forward(state_tensor)
                action = q_values.argmax().item()

            obs, reward, terminated, truncated, info = self.env.step(action)
            readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
            speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
            readings = np.append(readings, speed)
            total_reward += reward

            cv2.imshow("Game", obs)
            cv2.waitKey(1)

        print(f"Total Reward: {total_reward:.2f}")

if __name__ == "__main__":

    racing = QNetwork()
    racing.train_agent(n_epochs=250)
    for _ in range(10):
        racing.play()

"""
The solution to the problems highlighted above is to, instead of using a table, approximating the Q-function using a 
neural network, a.k.a. Deep-Q Learning (DQN). By using a neural net instead of a table, the agent can generalise better, 
since it can combine knowledge from different episodes to form a higher-level understanding of the environment it is in.

To convert the Q-learning update rule into something usable by a neural net, we must use a loss function which resembles
the Q-learning update rule. Since our goal is to minimise the update in the function Q(s,a) = Q(s,a) + α[r + γ * max_a'(Q(s',a')) - Q(s,a)],
this means we have to make the (r + γ * max_a'(Q(s',a')) - Q(s,a)) term as small as possible. This is done by making
the reward plus the scaled max possible q-value for the next step, and the q-value for the current state/action pair equal
to each other. Therefore, we employ the mean squared error of r + γ * max_a'(Q(s',a')) and Q(s,a).

However, now we have a moving target problem, in two ways. The first is that the future q-values, Q(s',a'), are produced
by the same network as the one we are updating, resulting in approximating to a value which, under the same conditions,
will no longer be the same as before the training step. Then, there is also the case of instability caused by high correlation 
between examples. Neural nets assume that the data is independent and identically distributed. By feeding sequential data
to the neural network, the updates will all be highly correlated, causing the training to have high variance as each race
strongly swings the network to closely behave according to that track. It then becomes less suited for other tracks, and
the process repeats, resulting in large spikes in the training reward.
"""