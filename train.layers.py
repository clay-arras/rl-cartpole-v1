from collections import defaultdict
import gymnasium as gym
import torch
import numpy as np

EPOCHS = int(2e5)
GAMMA = 0.99
LR = 5e-4
EPS = 1e-5
SAVE_INTERVAL = 500

env = gym.make("CartPole-v1")
resume = False

NUM_OBS = np.prod(env.observation_space.shape)
NUM_ACT = env.action_space.n
HIDDEN_LAYER = 16


class PolicyNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = torch.nn.Linear(NUM_OBS, HIDDEN_LAYER)
        self.ln2 = torch.nn.Linear(HIDDEN_LAYER, NUM_ACT)
        self.soft = torch.nn.Softmax(dim=-1)
        self.act = torch.nn.ReLU()
    
    def forward(self, x) -> torch.Tensor:
        x = self.act(self.ln1(x))
        x = self.soft(self.ln2(x))
        return x

model = PolicyNetwork()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)


def discount_Rt(rewards):
    t = rewards.shape[-1]
    pows = torch.arange(t)
    mask = torch.tril(torch.outer(torch.pow(GAMMA, pows), torch.pow(GAMMA, -pows)))

    return rewards.reshape(1, t) @ mask


def policy_backward(A, S, R, t, rets):
    probs = model(S)

    Gt = discount_Rt(R)
    baseline = torch.mean(torch.tensor(rets), dtype=torch.float32)
    R = Gt - baseline

    oh_A = (torch.unsqueeze(A, axis=1) == torch.unsqueeze(torch.arange(NUM_ACT), axis=0)).int()
    oh_A = oh_A.reshape(t, 1, NUM_ACT)
    R = R.reshape(t, 1, 1)

    J = -torch.sum(R * oh_A * torch.log(probs + EPS))
    return J


def main():
    rets, costs = [1], [1]
    for i in range(EPOCHS):
        St, info = env.reset()
        St = torch.tensor(St.reshape(1, NUM_OBS))

        done = False
        t = 0
        A, S, R = [], [], []

        while not done:
            p = model(St).squeeze(axis=0)
            S.append(St)

            At = np.random.choice(np.arange(NUM_ACT), p=p.detach().numpy())
            St, Rt, term, trunc, info = env.step(At)
            St = torch.tensor(St.reshape(1, NUM_OBS))

            A.append(torch.tensor(At))
            R.append(torch.tensor(Rt))

            t += 1
            done = term or trunc

        A, S, R = torch.stack(A), torch.stack(S), torch.stack(R)
        J = policy_backward(A, S, R, t, rets)

        # we need to divide by t because we want to MEAN, since grads are all accumulated with sum
        optimizer.zero_grad()
        J.backward()
        with torch.no_grad():
            optimizer.step()

        costs.append(J)
        rets.append(torch.sum(R))
        if i % SAVE_INTERVAL == 0:
            print(
                "cost: ",
                torch.mean(torch.tensor(costs[-SAVE_INTERVAL:])),
                "avg_ret: ",
                torch.mean(torch.tensor(rets[-SAVE_INTERVAL:])),
            )
            # np.save("ckpt/w1.npy", model["W1"])
            # np.save("ckpt/w2.npy", model["W2"])

    env.close()


if __name__ == "__main__":
    main()
