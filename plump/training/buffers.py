# plump/training/buffers.py — decoupled replay, no env/model deps.
import collections
import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: collections.deque = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buf.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones, masks = zip(*batch)
        return (
            torch.from_numpy(np.array(states, dtype=np.float32)),
            torch.from_numpy(np.array(actions, dtype=np.int64)),
            torch.from_numpy(np.array(rewards, dtype=np.float32)),
            torch.from_numpy(np.array(next_states, dtype=np.float32)),
            torch.from_numpy(np.array(dones, dtype=np.float32)),
            torch.from_numpy(np.array(masks, dtype=np.bool_)),
        )

    def __len__(self):
        return len(self.buf)
