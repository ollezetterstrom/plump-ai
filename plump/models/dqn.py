# plump/models/dqn.py — isolated Dueling DQN, no env/encode deps.
import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_batched = x.dim() == 2
        if not is_batched:
            x = x.unsqueeze(0)
        features = self.features(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        if not is_batched:
            q_vals = q_vals.squeeze(0)
        return q_vals
