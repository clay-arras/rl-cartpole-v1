import gymnasium as gym
import torch
import numpy as np


class _BasePolicy:
    def __init__(self, env: gym.Env, model: dict[str, torch.nn.Module] = {}) -> None:
        self.env = env
        self.model = model

        self.obs_space = np.prod(self.env.observation_space.shape)
        self.act_space = self.env.action_space.n

    def learn(self, t: int) -> None:
        raise NotImplementedError()

    def predict(self, obs) -> None:
        raise NotImplementedError()

    def save(self, path: str) -> None:
        torch.save(self.model, path)

    def load(self, path: str) -> None:
        self.model = torch.load(path)
