from collections import defaultdict
import gymnasium as gym
import numpy as np

EPOCHS = int(2e5)
GAMMA = 0.99
LR = 5e-4
EPS = 1e-5
BETA = 0.90
SAVE_INTERVAL = 5000

env = gym.make("LunarLander-v3", continuous=False)
resume = False

NUM_OBS = env.observation_space.shape[0]
NUM_ACT = env.action_space.n
HIDDEN_LAYER = 128

grad_buffer = defaultdict(lambda: [])
rmsprop_cache = defaultdict(lambda: 0)
model = {}
if resume:
    model["W1"] = np.load("ckpt/w1_lunar.npy")
    model["W2"] = np.load("ckpt/w2_lunar.npy")
else:
    model["W1"] = np.random.randn(NUM_OBS, HIDDEN_LAYER) / np.sqrt(
        NUM_OBS
    )  # he initialization
    model["W2"] = np.random.randn(HIDDEN_LAYER, NUM_ACT) / np.sqrt(HIDDEN_LAYER)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    z = np.max(x, axis=1, keepdims=True)
    return np.exp(x - z) / np.sum(np.exp(x - z), axis=1, keepdims=True)


def relu(x):
    a = x.copy()
    a[a < 0] = 0
    return a


# def discount_Rt(n: int):
#     return ((1 - np.pow(GAMMA, np.arange(1, n + 1))) / (1 - GAMMA))[::-1]


def discount_Rt(rews):  # rews is (t, )
    t = rews.shape[-1]
    Gs = np.zeros(rews.shape)
    Gs[-1] = rews[-1]
    for i in range(t - 2, -1, -1):
        Gs[i] = rews[i] + GAMMA * Gs[i + 1]
    return Gs


def rmsprop_update(grads):
    param_update = {}
    for k, g in grads.items():
        grad = g.mean(axis=0)
        rmsprop_cache[k] = BETA * rmsprop_cache[k] + (1 - BETA) * grad**2
        param_update[k] = (LR / np.sqrt(rmsprop_cache[k] + EPS)) * grad
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
    baseline = np.mean(rets)
    R = Gt - baseline

    for k in grad_buffer:
        grad_buffer[k] = np.array(grad_buffer[k])

    oh_A = np.zeros((t, NUM_ACT))
    oh_A[np.arange(t), A] = 1

    R = R.reshape(t, 1, 1)
    oh_A = oh_A.reshape(t, 1, NUM_ACT)

    J = -np.sum(R * oh_A * np.log(grad_buffer["a2"] + EPS))

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
        St = St.reshape(1, NUM_OBS)

        done = False
        t = 0
        A, S, R = [], [], []

        while not done:
            p = policy_forward(St)
            S.append(St)

            at = np.random.choice(
                [a for a in range(NUM_ACT)], p=p
            )  # action at timestep t
            St, Rt, term, trunc, info = env.step(at)
            St = St.reshape(1, NUM_OBS)

            A.append(at)
            R.append(Rt)

            t += 1
            done = term or trunc

        A, S, R = np.array(A), np.array(S), np.array(R)
        grads, J = policy_backward(A, S, R, t, rets)
        param_update = rmsprop_update(grads)

        model["W1"] -= param_update["dJdW1"]
        model["W2"] -= param_update["dJdW2"]

        grad_buffer.clear()

        costs.append(J)
        rets.append(np.sum(R))
        if i % SAVE_INTERVAL == 0:
            print(
                "cost: ",
                np.mean(np.array(costs[-SAVE_INTERVAL:])),
                "avg_ret: ",
                np.mean(np.array(rets[-SAVE_INTERVAL:])),
            )
            np.save("ckpt/w1_lunar.npy", model["W1"])
            np.save("ckpt/w2_lunar.npy", model["W2"])

    env.close()


if __name__ == "__main__":
    main()
