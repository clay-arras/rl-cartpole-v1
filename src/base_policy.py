import gymnasium as gym
import torch

class _BasePolicy:
    def __init__(self, env: gym.Env, model: dict[str, torch.nn.Module] = {}) -> None:
        self.env = env
        self.model = model
    
    def learn(self, t: int) -> None:
        raise NotImplementedError()

    def predict(self, obs) -> None:
        raise NotImplementedError()

    def save(self, path: str) -> None:
        torch.save(self.model, path)

    def load(self, path: str) -> None:
        self.model = torch.load(path)
