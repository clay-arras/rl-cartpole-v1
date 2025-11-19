from collections import defaultdict
import gymnasium as gym
import autograd.numpy as np
from autograd import grad

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
    model["W1"] = np.load("ckpt/w1.npy")
    model["W2"] = np.load("ckpt/w2.npy")
else:
    model["W1"] = np.random.randn(NUM_OBS, HIDDEN_LAYER)
    model["W2"] = np.random.randn(HIDDEN_LAYER, NUM_ACT)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    z = np.max(x, axis=1, keepdims=True)
    return np.exp(x - z) / np.sum(np.exp(x - z), axis=1, keepdims=True)


def relu(x):
    # a = x.copy()
    # a[a < 0] = 0
    a = np.maximum(x, 0)
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
    assert St.shape == (1, NUM_OBS)
    z1 = St @ model["W1"]
    a1 = relu(z1)
    z2 = a1 @ model["W2"]
    a2 = np.clip(
        softmax(z2), EPS, 1 - EPS
    )  # clip is not gradient tracked. hopefully doesn't make a difference :p

    grad_buffer["z1"].append(z1)
    grad_buffer["a1"].append(a1)
    grad_buffer["z2"].append(z2)
    grad_buffer["a2"].append(a2)

    assert a2.shape == (1, NUM_ACT)
    return a2[0]


def policy_backward(model, A, oh_A, S, t, rets):
    S = S.reshape(t, NUM_OBS)
    z1 = S @ model["W1"]
    a1 = relu(z1)
    z2 = a1 @ model["W2"]
    a2 = softmax(z2)

    Gt = discount_Rt(t, GAMMA)
    baseline = np.mean(
        discount_Rt(np.ceil(np.mean(rets[-SAVE_INTERVAL:])), GAMMA)
    )  # scuffed baseline
    R = Gt - baseline


    # for k in grad_buffer:
    #     grad_buffer[k] = np.array(grad_buffer[k])

    R = R.reshape(t, 1, 1)
    oh_A = oh_A.reshape(t, 1, NUM_ACT)
    a2 = a2.reshape(t, 1, NUM_ACT)

    J = -np.sum(R * oh_A * np.log(a2 + 1e-8))

    # dJdz2 = R * (grad_buffer["a2"] - oh_A)
    # dJdW2 = grad_buffer["a1"].swapaxes(-1, -2) @ dJdz2

    # dJda1 = dJdz2 @ model["W2"].T
    # dJdz1 = dJda1 * (grad_buffer["z1"] > 0)
    # dJdW1 = S.swapaxes(-1, -2) @ dJdz1

    # return {"dJdW1": dJdW1, "dJdW2": dJdW2}, J
    return J


def main():
    rets, costs = [1], [1]
    for i in range(EPOCHS):
        St, info = env.reset()
        St = St.reshape(1, NUM_OBS)

        done = False
        tot_ret = 0
        t = 0
        A, S = [], []

        while not done:
            p = policy_forward(St)
            S.append(St)

            at = np.random.choice(
                [a for a in range(NUM_ACT)], p=p
            )  # action at timestep t
            St, Rt, term, trunc, info = env.step(at)
            St = St.reshape(1, NUM_OBS)

            A.append(at)

            tot_ret += Rt
            t += 1
            done = term or trunc

        A, S = np.array(A), np.array(S)
        # grads, J = policy_backward(A, S, t, rets)

        oh_A = np.zeros((t, NUM_ACT))
        oh_A[np.arange(t), A] = 1

        J = policy_backward(model, A, oh_A, S, t, rets)
        grads = grad(policy_backward)(model, A, oh_A, S, t, rets)

        param_update = rmsprop_update(grads)

        model["W1"] -= param_update["W1"]
        model["W2"] -= param_update["W2"]
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
