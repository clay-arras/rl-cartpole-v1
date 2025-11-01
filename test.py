import numpy as np
import gymnasium as gym


DISPLAY = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    a = x.copy()
    a[a > 0] = 0
    return a


def discount_Rt(n: int, gamma: float):
    return np.pow(gamma, np.arange(n))[::-1]


def main():
    W1 = np.load("ckpt/w1.npy")
    W2 = np.load("ckpt/w2.npy")

    if DISPLAY == True:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1")

    rets = []

    iterations = 1 if DISPLAY else 100

    for _ in range(iterations):
        St, info = env.reset()
        St = St.reshape(1, 4)

        done = False
        tot_ret = 0

        while not done:
            z1 = St @ W1
            a1 = relu(z1)
            z2 = a1 @ W2
            a2 = sigmoid(z2)
            p = a2[0, 0]

            at = np.random.choice([0, 1], p=[p, 1 - p])  # action at timestep t
            St, Rt, term, trunc, info = env.step(at)
            St = St.reshape(1, 4)

            tot_ret += Rt
            done = term or trunc

        rets.append(tot_ret)

    print("avg_ret: ", np.mean(rets))
    env.close()


if __name__ == "__main__":
    main()
