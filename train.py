from collections import defaultdict
import gymnasium as gym
import numpy as np

EPOCHS = int(2e5)
GAMMA = 0.99
LR = 1e-5
EPS = 1e-5
BETA = 0.90
SAVE_INTERVAL = 10000

resume = False

grad_buffer = defaultdict(lambda: [])
rmsprop_cache = defaultdict(lambda: 0)
model = {}
if resume:
    model["W1"] = np.load("ckpt/w1.npy")
    model["W2"] = np.load("ckpt/w2.npy")
else:
    model["W1"] = np.random.randn(4, 16)
    model["W2"] = np.random.randn(16, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    a = x.copy()
    a[a < 0] = 0
    return a


def discount_Rt(n: int, gamma: float):
    return ((1 - np.pow(GAMMA, np.arange(1, n + 1))) / (1 - GAMMA))[::-1]


def rmsprop_update(grads):
    param_update = {}
    for k, g in grads.items():
        grad = g.mean(axis=0)
        rmsprop_cache[k] = BETA * rmsprop_cache[k] + (1 - BETA) * grad**2
        param_update[k] = (LR / np.sqrt(rmsprop_cache[k] + EPS)) * grad
    return param_update


def policy_forward(St):
    z1 = St @ model["W1"]
    a1 = relu(z1)
    z2 = a1 @ model["W2"]
    a2 = np.clip(
        sigmoid(z2), EPS, 1 - EPS
    )  # clip is not gradient tracked. hopefully doesn't make a difference :p
    p = a2[0, 0]

    grad_buffer["z1"].append(z1)
    grad_buffer["a1"].append(a1)
    grad_buffer["z2"].append(z2)
    grad_buffer["a2"].append(a2)

    return p


def policy_backward(A, S, t, rets):
    Gt = discount_Rt(t, GAMMA)
    baseline = np.mean(
        discount_Rt(np.ceil(np.mean(rets[-SAVE_INTERVAL:])), GAMMA)
    )  # scuffed baseline
    R = Gt - baseline

    for k in grad_buffer:
        grad_buffer[k] = np.array(grad_buffer[k])

    J = -np.sum(
        R * ((1 - A) * np.log(grad_buffer["a2"]) + A * np.log(1 - grad_buffer["a2"]))
    )
    A = A.reshape(t, 1, 1)
    R = R.reshape(t, 1, 1)

    dJdJ = 1
    dJda2 = dJdJ * R * -1 * ((1 - A) / grad_buffer["a2"] - A / (1 - grad_buffer["a2"]))
    dJdz2 = dJda2 * (1 - grad_buffer["a2"]) * grad_buffer["a2"]

    dJda1 = dJdz2 @ model["W2"].T
    dJdW2 = grad_buffer["a1"].swapaxes(-1, -2) @ dJdz2

    dJdz1 = dJda1 * (grad_buffer["z1"] > 0)
    dJdW1 = S.swapaxes(-1, -2) @ dJdz1

    return {"dJdW1": dJdW1, "dJdW2": dJdW2}, J


def main():
    env = gym.make("CartPole-v1")

    rets, costs = [1], [1]
    for i in range(EPOCHS):
        St, info = env.reset()
        St = St.reshape(1, 4)

        done = False
        tot_ret = 0
        t = 0
        A, S = [], []

        while not done:
            p = policy_forward(St)
            S.append(St)

            at = np.random.choice([0, 1], p=[p, 1 - p])  # action at timestep t
            St, Rt, term, trunc, info = env.step(at)
            St = St.reshape(1, 4)

            A.append(at)

            tot_ret += Rt
            t += 1
            done = term or trunc

        A, S = np.array(A), np.array(S)
        grads, J = policy_backward(A, S, t, rets)
        param_update = rmsprop_update(grads)

        # model["W1"] -= LR * param_update["dJdW1"]
        # model["W2"] -= LR * param_update["dJdW2"]
        model["W1"] -= LR * grads["dJdW1"].mean(axis=0)
        model["W2"] -= LR * grads["dJdW2"].mean(axis=0)

        grad_buffer.clear()

        costs.append(J)
        rets.append(tot_ret)
        if i % SAVE_INTERVAL == 0:
            print(
                "cost: ",
                np.mean(np.array(costs[-SAVE_INTERVAL:])),
                "avg_ret: ",
                np.mean(np.array(rets[-SAVE_INTERVAL:])),
            )
            # np.save("ckpt/w1.npy", model["W1"])
            # np.save("ckpt/w2.npy", model["W2"])

    env.close()


if __name__ == "__main__":
    main()
