import torch
import radar_wrapper as radar
import numpy as np
import cv2
from ReplayBuffer import ReplayBuffer
from QNetwork import QNetwork
from copy import deepcopy
from torch.backends.mps import is_available


class QNetReplayBuf(QNetwork):

    def __init__(self, buffer_size=50_000, **kwargs):
        """
        Based off the previously constructed QNetwork class, but this one uses a replay buffer instead of feeding the
        steps sequentially. Moreover, we introduce a target_net which gets updated less often, but allows for lower
        variance during training. Further explanations/Justifications given at the end of this file.

        :param buffer_size:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.rb = ReplayBuffer(buffer_size)
        self.target_net = QNetwork(**kwargs)
        self.target_net.load_state_dict(self.state_dict(), strict=False)
        self.target_net.to(self.device)
        self.target_net.eval()

    def training_step(self, batch_size=64):
        """
        Similar to ``QNetwork.training_step``, but instead of feeding the steps sequentially we sample some `batch_size`
        steps from a replay buffer and use them to update the online network.

        :param batch_size:
        :return:
        """
        if len(self.rb) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.rb.sample(batch_size)

        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

        all_q_values = self.forward(states_t)
        # for each row (step), we fetch the q-value corresponding to the action taken
        chosen_q = all_q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(chosen_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_agent(self, n_epochs=100, train_seeds=20, val_seeds=5, batch_size=64):
        """
        Similar to ``QNetwork.train_agent``, but instead of feeding the steps sequentially we add them to a replay buffer
        which is sampled randomly. We also update the target network after every epoch.

        :param n_epochs:
        :param train_seeds:
        :param val_seeds:
        :param batch_size:
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
                readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
                speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
                readings = np.append(readings, speed)
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    if np.random.rand() < epsilon:
                        action = np.random.randint(0, self.N_ACTIONS)
                    else:
                        state_t = torch.tensor([readings], dtype=torch.float32).to(self.device)
                        with torch.no_grad():
                            q_values = self.forward(state_t)
                            action = q_values.argmax().item()

                    obs, reward, terminated, truncated, info = self.env.step(action)
                    next_readings = radar.get_radar_readings(obs, self.n_rays, self.len_ray) / self.len_ray
                    next_speed = self.env.unwrapped.car.hull.linearVelocity.length / self.MAX_SPEED
                    next_readings = np.append(next_readings, next_speed)

                    self.rb.push(readings, action, reward, next_readings, terminated)
                    loss = self.training_step(batch_size)

                    readings = next_readings
                    train_reward += reward

                    if epoch % 20 == 0 and seed == 0:
                        cv2.imshow("Game", obs)
                        cv2.waitKey(1)

                self.target_net.load_state_dict(self.state_dict(), strict=False)

            train_reward /= train_seeds

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

            if epoch % 20 == 0 and best_model_state:
                torch.save(best_model_state, 'qnetwork_replay.pt')

        torch.save(best_model_state, 'qnetwork_replay_final.pt')


if __name__ == "__main__":

    racing = QNetReplayBuf()
    racing.train_agent(n_epochs=250)
    for _ in range(10):
        racing.play(filename="qnetwork_replay_final.pt")

"""
To reduce the 2 causes of variance of DQN (highly correlated data fed in a row and moving target), we can employ two 
solutions:

1. Use a replay buffer to store transitions (s, a, r, s'), and sample from that buffer. By making the buffer size much 
larger than what a single episode (race) can collect, we can fetch random steps in minibatches from a collection of 
trajectories and from varying moments within each trajectory. This ensures that in a short time span, the network 
is exposed to many different scenarios, which approximates the i.i.d. assumption that neural networks have.
Practically, this prevents the network constantly overfitting against the most recent track it has seen.

2. Use a slower-updating target network to calculate our target q-values. Unlike the online network, which goes through 
a training step every step, the target network is frozen to the online network's weights at the end of the previous epoch.
By only updating once an epoch (every 1000 steps), the online network can, for a whole epoch, have a consistent target 
to work towards, which produces less spikes.

As we can see in the training (and to a lesser extent) in the validation rewards graphs, DQN with replay buffer and 
a target network tends to outperform outperforms regular DQN, and produces more stable results.
"""