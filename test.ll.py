from collections import defaultdict
import gymnasium as gym
import numpy as np

# import jax.numpy as np

EPOCHS = 100
DISPLAY = True

if DISPLAY == True:
    env = gym.make("LunarLander-v3", continuous=False, render_mode="human")
else:
    env = gym.make("LunarLander-v3", continuous=False)

NUM_OBS = env.observation_space.shape[0]
NUM_ACT = env.action_space.n
HIDDEN_LAYER = 128

model = {}
model["W1"] = np.load("ckpt/w1_lunar.npy")
model["W2"] = np.load("ckpt/w2_lunar.npy")


def softmax(x):
    z = np.max(x, axis=1, keepdims=True)
    return np.exp(x - z) / np.sum(np.exp(x - z), axis=1, keepdims=True)


def relu(x):
    a = x.copy()
    a[a < 0] = 0
    return a


def policy_forward(St):
    assert St.shape == (1, NUM_OBS)
    z1 = St @ model["W1"]
    a1 = relu(z1)
    z2 = a1 @ model["W2"]
    a2 = softmax(z2)

    assert a2.shape == (1, NUM_ACT)
    return a2[0]


def main():
    iterations = 1 if DISPLAY else EPOCHS

    rets = []
    for _ in range(iterations):
        St, info = env.reset()
        St = St.reshape(1, NUM_OBS)

        done = False
        t = 0
        R = []

        while not done:
            p = policy_forward(St)
            at = np.random.choice(
                [a for a in range(NUM_ACT)], p=p
            )  # action at timestep t
            St, Rt, term, trunc, info = env.step(at)
            St = St.reshape(1, NUM_OBS)
            R.append(Rt)

            t += 1
            done = term or trunc
        rets.append(np.sum(R))

    print("avg_ret: ", np.mean(rets))
    env.close()


if __name__ == "__main__":
    main()
