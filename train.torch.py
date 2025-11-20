from collections import defaultdict
import gymnasium as gym
import torch
import numpy as np

EPOCHS = int(2e5)
GAMMA = 0.99
LR = 5e-4
EPS = 1e-5
BETA = 0.90
SAVE_INTERVAL = 500

env = gym.make("CartPole-v1")
resume = False

NUM_OBS = env.observation_space.shape[0]
NUM_ACT = env.action_space.n
HIDDEN_LAYER = 16

grad_buffer = defaultdict(lambda: [])
rmsprop_cache = defaultdict(lambda: 0)
model = {}
if resume:
    model["W1"] = torch.load("ckpt/w1.pt")
    model["W2"] = torch.load("ckpt/w2.pt")
else:
    model["W1"] = torch.randn(NUM_OBS, HIDDEN_LAYER)
    model["W2"] = torch.randn(HIDDEN_LAYER, NUM_ACT)


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def softmax(x):
    z = torch.max(x, axis=1, keepdims=True).values
    return torch.exp(x - z) / torch.sum(torch.exp(x - z), axis=1, keepdims=True)


def relu(x):
    a = torch.maximum(x, torch.zeros_like(x))
    return a


def discount_Rt(rewards):
    t = rewards.shape[-1]
    pows = torch.arange(t)
    mask = torch.tril(torch.outer(torch.pow(GAMMA, pows), torch.pow(GAMMA, -pows)))

    return rewards.reshape(1, t) @ mask


def rmsprop_update(grads):
    param_update = {}
    for k, g in grads.items():
        grad = g.mean(axis=0)
        rmsprop_cache[k] = BETA * rmsprop_cache[k] + (1 - BETA) * grad**2
        param_update[k] = (LR / torch.sqrt(rmsprop_cache[k] + EPS)) * grad
    return param_update


def policy_forward(St):
    assert St.shape == (1, NUM_OBS)
    z1 = St @ model["W1"]
    a1 = relu(z1)
    z2 = a1 @ model["W2"]
    a2 = softmax(z2)

    grad_buffer["z1"].append(z1)
    grad_buffer["a1"].append(a1)
    grad_buffer["z2"].append(z2)
    grad_buffer["a2"].append(a2)

    assert a2.shape == (1, NUM_ACT)
    return a2[0]


def policy_backward(A, S, R, t, rets):
    Gt = discount_Rt(R)
    baseline = torch.mean(torch.tensor(rets), dtype=torch.float32)
    R = Gt - baseline

    for k in grad_buffer:
        grad_buffer[k] = torch.tensor([x.tolist() for x in grad_buffer[k]])

    oh_A = (torch.unsqueeze(A, axis=1) == torch.unsqueeze(torch.arange(NUM_ACT), axis=0)).int()
    oh_A = oh_A.reshape(t, 1, NUM_ACT)
    R = R.reshape(t, 1, 1)

    J = -torch.sum(R * oh_A * torch.log(grad_buffer["a2"] + EPS))

    dJdz2 = R * (grad_buffer["a2"] - oh_A)
    dJdW2 = grad_buffer["a1"].swapaxes(-1, -2) @ dJdz2

    dJda1 = dJdz2 @ model["W2"].T
    dJdz1 = dJda1 * (grad_buffer["z1"] > 0)
    dJdW1 = S.swapaxes(-1, -2) @ dJdz1

    return {"dJdW1": dJdW1, "dJdW2": dJdW2}, J


def main():
    rets, costs = [1], [1]
    for i in range(EPOCHS):
        St, info = env.reset()
        St = torch.tensor(St.reshape(1, NUM_OBS))

        done = False
        t = 0
        A, S, R = [], [], []

        while not done:
            p = policy_forward(St)
            S.append(St)

            At = np.random.choice(np.arange(NUM_ACT), p=p.numpy())
            St, Rt, term, trunc, info = env.step(At)
            St = torch.tensor(St.reshape(1, NUM_OBS))

            A.append(torch.tensor(At))
            R.append(torch.tensor(Rt))

            t += 1
            done = term or trunc

        A, S, R = torch.stack(A), torch.stack(S), torch.stack(R)
        grads, J = policy_backward(A, S, R, t, rets)
        param_update = rmsprop_update(grads)

        model["W1"] -= param_update["dJdW1"]
        model["W2"] -= param_update["dJdW2"]

        grad_buffer.clear()

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
