from src.base_policy import _BasePolicy

import torch
from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked as typechecker
import gymnasium as gym
import numpy as np


class _BasePolicyClassifierNetwork(torch.nn.Module):
    def __init__(self, nin: int, nout: int, hidden_layer: int = 64) -> None:
        super().__init__()
        self.ln1 = torch.nn.Linear(nin, hidden_layer)
        self.ln2 = torch.nn.Linear(hidden_layer, nout)
        self.soft = torch.nn.Softmax(dim=-1)
        self.act = torch.nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        x = self.act(self.ln1(x))
        x = self.soft(self.ln2(x))
        return x


class _BasePolicyRegressionNetwork(torch.nn.Module):
    def __init__(self, nin: int, nout: int, hidden_layer: int = 64) -> None:
        super().__init__()
        self.ln1 = torch.nn.Linear(nin, hidden_layer)
        self.ln2 = torch.nn.Linear(hidden_layer, nout)
        self.act = torch.nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        x = self.act(self.ln1(x))
        x = self.ln2(x)
        return x


class ActorCritic(_BasePolicy):
    def __init__(
        self, env: gym.Env, disc_gamma=0.99, learning_rate=5e-4, save_interval=int(5e2)
    ) -> None:
        super().__init__(env)
        self.obs_space = np.prod(self.env.observation_space.shape)
        self.act_space = self.env.action_space.n

        self.actor = _BasePolicyClassifierNetwork(self.obs_space, self.act_space)
        self.critic = _BasePolicyRegressionNetwork(self.obs_space, 1)
        self.actor_optimizer = torch.optim.RMSprop(
            self.actor.parameters(), lr=learning_rate
        )
        self.critic_optimizer = torch.optim.RMSprop(
            self.critic.parameters(), lr=learning_rate
        )

        self.learning_rate = learning_rate
        self.disc_gamma = disc_gamma
        self.save_interval = save_interval

    @jaxtyped(typechecker=typechecker)
    def _loss_policy(
        self,
        adV: Float[torch.Tensor, "1"],
        At: Int[torch.Tensor, "1"],
        St: Float[torch.Tensor, "1 obs_space"],
    ) -> Float[torch.Tensor, "1"]:
        """
        description: this is the loss calculation for the policy,
        calculated by the sum of -ohA_t V log(policy(St)).
        - ohA_t is (act_space,) and represents one-hot encoded action
        - adV is (1,) and represents the advantage given by the critic,
          consisting of Rt + gamma V(St+1) - V(St)
        - policy(St) is (act_space,)

        inputs:
        - adV is (1,) and represents Rt + gamma V(St+1) - V(St), aka the
          estimated reward / advantage given the state. should NOT be gradient
          tracked
        - At is (1,) and represents the action taken at timestep t
        - St is (1, obs_space) and represents the state at timestep t

        outputs:
        - J is (1,) and represents the cost tensor
        """
        ohA_t = torch.nn.functional.one_hot(At, num_classes=self.act_space)
        assert isinstance(ohA_t, Int[torch.Tensor, "1 act_space"])
        policy_St = self.actor(St)
        assert isinstance(policy_St, Float[torch.Tensor, "1 act_space"])
        J = -torch.sum(ohA_t * adV * torch.log(policy_St + 1e-8))
        return J.reshape(1)

    @jaxtyped(typechecker=typechecker)
    def _loss_value(
        self,
        V_prev: Float[torch.Tensor, "1"],
        V_curr: Float[torch.Tensor, "1"],
        Rt: float,
    ) -> Float[torch.Tensor, "1"]:
        """
        description: this should use MSE, where the actual value is estimated
        to be (gamma * V(St+1) + Rt), and predicted value is V(St)

        inputs:
        - V_prev is (1,) and represents V(St). this tensor is what we are
          backpropogating over, similar to y_pred
        - V_curr is (1,) and represents V(St+1). this tensor should not be
          gradient tracked and should be detached. treat this like y_actual
        - Rt is (1,) and represents the reward gotten from the environment

        outputs:
        - J is (1,) and represents a cost tensor
        """
        loss_fn = torch.nn.MSELoss()
        J = loss_fn(V_prev, self.disc_gamma * V_curr + Rt)
        return J.reshape(1)

    def learn(self, t: int) -> None:
        """
        description:
        inside the training loop for each episode, it needs to do the following
        - get action probabilities from actor net
        - sample At from action probabilities
        - sample St+1, Rt from the environment
        - get V(St+1) from critic net
        - update the actor weights
        - update the critic weights
        - store V_prev as V(St+1)

        inputs:
        - t is an int representing the number of timesteps to take

        outputs: None
        """
        rets = []
        for timestep in range(t):
            St, _ = self.env.reset()
            St = torch.tensor(St.reshape(1, self.obs_space))

            done = False
            St_prev = None
            R = []

            while not done:
                logits = self.actor(St)
                assert isinstance(logits, Float[torch.Tensor, "1 act_space"])
                mnom = torch.distributions.categorical.Categorical(probs=logits)
                At = mnom.sample()
                assert isinstance(At, Int[torch.Tensor, "1"])

                St, Rt, term, trunc, _ = self.env.step(
                    At.detach().item()
                )  # .numpy() instead of .item()?
                St = torch.tensor(St.reshape(1, self.obs_space))
                assert isinstance(St, Float[torch.Tensor, "1 obs_space"])

                if St_prev is not None:
                    V_curr = self.critic(St).reshape(
                        1,
                    )
                    assert isinstance(V_curr, Float[torch.Tensor, "1"])
                    V_prev = self.critic(St_prev).reshape(
                        1,
                    )
                    assert isinstance(V_prev, Float[torch.Tensor, "1"])
                    adV = Rt + self.disc_gamma * V_curr.detach() - V_prev.detach()

                    aJ = self._loss_policy(adV, At, St_prev)
                    self.actor_optimizer.zero_grad()
                    aJ.backward()
                    self.actor_optimizer.step()

                    cJ = self._loss_value(V_prev, V_curr, Rt)
                    self.critic_optimizer.zero_grad()
                    cJ.backward()
                    self.critic_optimizer.step()

                St_prev = St.detach()
                R.append(torch.tensor(Rt))
                done = term or trunc

            rets.append(torch.sum(torch.stack(R)))
            if timestep % self.save_interval == 0:
                print(
                    "avg_ret: ",
                    torch.mean(torch.tensor(rets[-self.save_interval :])),
                )
        self.env.close()

    def predict(self) -> int:
        pass


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    policy = ActorCritic(env)

    policy.learn(10000)
