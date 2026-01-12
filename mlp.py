from torch import device
from torch.backends.mps import is_available
from ActorCritic import ActorCritic

if is_available():
    device = device("mps")
    print("Success: Using Apple M4 GPU (Metal)")
else:
    device = device("cpu")
    print("Warning: Using CPU")

network = ActorCritic(device)
#network.train_agent(1400)
for _ in range(100):
    network.play()