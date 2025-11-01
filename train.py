import gymnasium as gym
import numpy as np

EPOCHS = int(2e5)
GAMMA = 0.99
LR = 1e-5
EPS = 1e-5


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    a = x.copy()
    a[a < 0] = 0
    return a


def discount_Rt(n: int, gamma: float):
    return ((1 - np.pow(GAMMA, np.arange(1, n+1))) / (1 - GAMMA))[::-1]


def main():
    env = gym.make("CartPole-v1")

    # W1 = np.random.randn(4, 16)
    # W2 = np.random.randn(16, 1)
    W1 = np.load("ckpt/w1.npy")
    W2 = np.load("ckpt/w2.npy")

    rets, costs = [1], [1]

    for i in range(EPOCHS):
        St, info = env.reset()
        St = St.reshape(1, 4)

        done = False
        tot_ret = 0
        t = 0
        At, A1, A2, Z1, S = [], [], [], [], []

        while not done:
            z1 = St @ W1
            a1 = relu(z1)
            z2 = a1 @ W2
            a2 = np.clip(sigmoid(z2), EPS, 1-EPS) # clip is not gradient tracked. hopefully doesn't make a difference :p
            p = a2[0, 0]

            S.append(St)

            at = np.random.choice([0, 1], p=[p, 1 - p])  # action at timestep t
            St, Rt, term, trunc, info = env.step(at)
            St = St.reshape(1, 4)

            At.append(at)
            A1.append(a1)
            A2.append(a2)
            Z1.append(z1)

            tot_ret += Rt
            t += 1
            done = term or trunc

        At, A1, A2, Z1, S = (
            np.array(At),
            np.array(A1),
            np.array(A2),
            np.array(Z1),
            np.array(S),
        )
        Gt = discount_Rt(t, GAMMA)
        baseline = np.mean(discount_Rt(np.ceil(np.mean(rets[-10000:])), GAMMA))  # scuffed baseline
        R = Gt - baseline

        J = -np.sum(R * ((1 - At) * np.log(A2) + At * np.log(1 - A2))) 
        At = At.reshape(t, 1, 1)
        R = R.reshape(t, 1, 1)

        dJdJ = 1
        dJda2 = dJdJ * R * -1 * ((1 - At) / A2 - At / (1 - A2))
        dJdz2 = dJda2 * (1 - A2) * A2

        dJda1 = dJdz2 @ W2.T
        dJdW2 = A1.swapaxes(-1, -2) @ dJdz2

        dJdz1 = dJda1 * (Z1 > 0)
        dJdW1 = S.swapaxes(-1, -2) @ dJdz1

        W1 -= LR * dJdW1.mean(axis=0)
        W2 -= LR * dJdW2.mean(axis=0)

        costs.append(J)
        rets.append(tot_ret)
        if i % 10000 == 0:
            print(
                "cost: ",
                np.mean(np.array(costs[-10000:])),
                "avg_ret: ",
                np.mean(np.array(rets[-10000:])),
            )
            np.save("ckpt/w1.npy", W1)
            np.save("ckpt/w2.npy", W2)

    env.close()


if __name__ == "__main__":
    main()
