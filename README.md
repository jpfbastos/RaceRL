# RaceRL - an exploration of multiple Reinforcement Learning algorithms in Gymnasium's `CarRacing-v3` environment

Over the past months, I have become increasnigly interested in reinforcement learning (RL) and how it can solve many different tasks just by receiving feedback from a pre-determined environment. Curious to explore, I wanted to give a shot at implementing different RL algorithms to understand the strenghts and weaknesses of each, and develop a strong understanding of concepts and best practices through having to produce these algorithms. 

I settled with CarRacing because a) I'm an avid motorsports fan! and b) because I could use a simple radar rather than the image pixels for training data. Many implementations (and DeepMind's Atari with Reinforcement Learning paper) use the pixels in the image produced by the environment to feed the agent so it can "see", much like we do, what is going on in the game and adapt. However, I wanted to have as small of a non-RL pipeline as possible so I could focus on the RL side of things. Therefore, I used a radar which would calculate the radial distances from the car to the edge of the track in several directions as the main orientation mechanism for the agent, along with the speed so the car knew when to brake. This would allow for the car to know, for example, that the left wall was closer than the right wall, so it should move right. 

I used 5 rays, resulting in the car knowing these distances:

## TODO IMG

Once this was set, it was a matter of applying different algorithms, comparing, and learning more about the exciting world of RL!

## Off-Policy Algorithms

### Tabular Q-Learning

To begin, I implemented the simplest RL algorithm, tabular Q-learning. This algorithm works by having a table with all possible states, with each entry containing a Q-value. A Q-value is the expected cumulative future reward an agent will receive by taking a specific action in a given state and following optimal policy. We update using the a variant of the Bellman Equation, called the Temporal Difference (TD) update rule ($Q(s,a) = Q(s,a) + α[r + γ * max_{a'}(Q(s',a')) - Q(s,a)]$), which iteratively adjusts the Q-table such that the TD error is minimised, and the Q-values approximate the expected rewards correctly. 

This algorithm has a major disadvantage, since we have to use discrete buckets for our radar and speed readings instead of using the continuous values, reducing how precise we can be in our movements. Having too few buckets reduces granularity of data, whereas having too many buckets makes the data sparse, meaning the Q-values are dictated by only a few sample points (curse of dimensionality). We save values in a dictionary, but if we were to save them in a table, we'd only use 25.63% of entries (where the size of each dimension is zero to the largest numbered bucket observed - buckets not in the data are not counted, so if using all buckets in truth this percentage would be even lower). 

Since entries on the table are independent, it is hard to extrapolate any trends between each entry. This results in the model not being able to generalise behaviour across the policy (e.g. if next to left wall turn right) because it would have to reach that conclusion in all other ray/speed combinations until it becomes a general rule. Therefore, we see a highly oscillating pattern as readings transition from one set of bucket to the next while moving.

### Deep Q-Learning

The solution to the problems highlighted above is to, instead of using a table, approximating the Q-function using a neural network, a.k.a. Deep-Q Learning (DQN). By using a neural net instead of a table, the agent can generalise better, since it can combine knowledge from different episodes to form a higher-level understanding of the environment it is in.

To convert the Q-learning update rule into something usable by a neural net, we must use a loss function which resembles the Q-learning update rule. Since our goal is to minimise the update in the function $Q(s,a) = Q(s,a) + α[r + γ * max_{a'}(Q(s',a')) - Q(s,a)]$, this means we have to make the ($r + γ * max_{a'}(Q(s',a')) - Q(s,a)$) term as small as possible. This is done by making the reward plus the scaled max possible q-value for the next step, and the q-value for the current state/action pair equal to each other. Therefore, we employ the mean squared error of $r + γ * max_{a'}(Q(s',a'))$ and $Q(s,a)$.

## TODO IMGS

The result is a much better performance in both training and validation as we now can use continuous inputs and can use extrapolate learnings from one state to the other states in a much easier way. 

However, now we have a moving target problem, in two ways. The first is that the future q-values, Q(s',a'), are produced by the same network as the one we are updating, resulting in approximating to a value which, under the same conditions, will no longer be the same as before the training step. Then, there is also the case of instability caused by high correlation between examples. Neural nets assume that the data is independent and identically distributed. By feeding sequential data to the neural network, the updates will all be highly correlated, causing the training to have high variance as each race strongly swings the network to closely behave according to that track. It then becomes less suited for other tracks, and the process repeats, resulting in large spikes in the training reward. This causes the huge spikes in performance which can be observed in both graphs.

#### Adding a Replay Buffer and Target Network

To reduce the 2 causes of variance of DQN (highly correlated data fed in a row and moving target), we can employ two solutions:

1. Use a replay buffer to store transitions (s, a, r, s'), and sample from that buffer. By making the buffer size much larger than what a single episode (race) can collect, we can fetch random steps in minibatches from a collection of trajectories and from varying moments within each trajectory. This ensures that in a short time span, the network is exposed to many different scenarios, which approximates the i.i.d. assumption that neural networks have. Practically, this prevents the network constantly overfitting against the most recent track it has seen.

2. Use a slower-updating target network to calculate our target q-values. Unlike the online network, which goes through a training step every step, the target network is frozen to the online network's weights at the end of the previous epoch. By only updating once an epoch (every 1000 steps), the online network can, for a whole epoch, have a consistent target to work towards, which produces less spikes.

As we can see in the training (and to a lesser extent) in the validation rewards graphs, DQN with replay buffer and a target network tends to outperform outperforms regular DQN, and produces more stable results.

## On-Policy Algorithms





