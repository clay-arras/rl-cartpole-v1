import torch
import gymnasium as gym
import numpy as np
from base_policy import _BasePolicy


class _BasePolicyNetwork(torch.nn.Module):
    def __init__(self, num_obs: int, num_act: int, hidden_layer: int = 64) -> None:
        super().__init__()
        self.ln1 = torch.nn.Linear(num_obs, hidden_layer)
        self.ln2 = torch.nn.Linear(hidden_layer, num_act)
        self.soft = torch.nn.Softmax(dim=-1)
        self.act = torch.nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        x = self.act(self.ln1(x))
        x = self.soft(self.ln2(x))
        return x


class ActorCritic(_BasePolicy):
    def __init__(
        self, env: gym.Env, disc_gamma=0.99, learning_rate=5e-4, save_interval=int(5e2)
    ) -> None:
        super().__init__(env)
        self.actor = _BasePolicyNetwork(self.obs_space, self.act_space)
        self.critic = _BasePolicyNetwork(self.obs_space + self.act_space, 1)
        self.optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate
        self.disc_gamma = disc_gamma
        self.save_interval = save_interval

    def _loss_policy(self, At, St) -> torch.Tensor:  # At, St, Rt, Qw_sa
        """
        St is (1, obs_space)
        At is (1,)

        change:
        - loss needs to update every TIMESTEP instead of at the end of each episode
        - discounted rewards is replaced by the values.
        """
        oh_A = (
            torch.unsqueeze(At, axis=1)
            == torch.unsqueeze(torch.arange(self.act_space), axis=0)
        ).int()
        assert oh_A.shape == (1, self.act_space)

        prob = self.actor(St)
        adv = self.critic(torch.cat([St, oh_A], axis=-1))
        J = -torch.sum(adv * oh_A * torch.log(prob + 1e-8))
        return J

    def _loss_value(self, Rt, val_prev, val_curr) -> torch.Tensor:
        J = -torch.sum(val_prev, Rt + self.disc_gamma * val_curr)
        return J

    def learn(self, t: int) -> None:
        rets, costs = [1], [1]
        for i in range(t):
            St, _ = self.env.reset()
            St = torch.tensor(St.reshape(1, self.obs_space))

            done = False
            A, S, R = [], [], []

            while not done:
                p = self.model(St).squeeze(axis=0)
                S.append(St)

                At = np.random.choice(np.arange(self.act_space), p=p.detach().numpy())
                St, Rt, term, trunc, _ = self.env.step(At)
                St = torch.tensor(St.reshape(1, self.obs_space))

                A.append(torch.tensor(At))
                R.append(torch.tensor(Rt))
                done = term or trunc

            A, S, R = torch.stack(A), torch.stack(S), torch.stack(R)
            J = self._loss(A, S, R, rets[-self.save_interval :])

            self.optimizer.zero_grad()
            J.backward()
            with torch.no_grad():
                self.optimizer.step()

            costs.append(J)
            rets.append(torch.sum(R))
            if i % self.save_interval == 0:
                print(
                    "cost: ",
                    torch.mean(torch.tensor(costs[-self.save_interval :])),
                    "avg_ret: ",
                    torch.mean(torch.tensor(rets[-self.save_interval :])),
                )
        self.env.close()

    # def predict(self, St: np.ndarray) -> int:
    #     St = torch.tensor(St.reshape(1, self.obs_space))
    #     p = self.model(St).squeeze(axis=0)

    #     At = np.random.choice(np.arange(self.act_space), p=p.detach().numpy())
    #     return At
