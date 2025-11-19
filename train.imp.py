from collections import defaultdict
import gymnasium as gym
import autograd.numpy as np
from autograd import grad

resume = False
hidden_layer = 16

epochs = int(1e5)
gamma = 0.99
lr = 5e-4
eps = 1e-5
beta = 0.90
save_interval = 500

env = gym.make("CartPole-v1")
num_obs = env.observation_space.shape[0]
num_act = env.action_space.n


grad_buffer = defaultdict(lambda: [])
rmsprop_cache = defaultdict(lambda: 0)

model = {}
if resume:
    model["W1"] = np.load("ckpt/w1_lunar.npy")
    model["W2"] = np.load("ckpt/w2_lunar.npy")
else:  # he initialization
    model["W1"] = np.random.randn(num_obs, hidden_layer) / np.sqrt(num_obs)
    model["W2"] = np.random.randn(hidden_layer, num_act) / np.sqrt(hidden_layer)


def softmax(x):
    z = np.max(x, axis=1, keepdims=True)
    return np.exp(x - z) / np.sum(np.exp(x - z), axis=1, keepdims=True)


def relu(x):
    a = np.maximum(x, 0)
    return a


def discount_Rt(rewards):
    t = rewards.shape[-1]
    pows = np.arange(t)
    mask = np.outer(np.pow(gamma, pows), np.pow(gamma, -pows)) * np.tri(t)

    return rewards.reshape(1, t) @ mask


def rmsprop_update(grads):
    param_update = {}
    for k, g in grads.items():
        grad = g.mean(axis=0)
        rmsprop_cache[k] = beta * rmsprop_cache[k] + (1 - beta) * grad**2
        param_update[k] = (lr / np.sqrt(rmsprop_cache[k] + eps)) * grad
    return param_update


def policy_forward(model, St, t):
    z1 = St @ model["W1"]
    a1 = relu(z1)
    z2 = a1 @ model["W2"]
    a2 = softmax(z2)

    return a2[0]


def policy_backward(model, A, S, R, t, rets):
    S = S.reshape(t, num_obs)
    z1 = S @ model["W1"]
    a1 = relu(z1)
    z2 = a1 @ model["W2"]
    a2 = softmax(z2)

    Gt = discount_Rt(R)
    # baseline = np.mean(np.array(rets))
    # adv = Gt - baseline
    adv = (Gt - np.mean(Gt)) / (np.std(Gt) + 1e-8)


    oh_A = np.expand_dims(A, axis=1) == np.expand_dims(np.arange(num_act), axis=0)

    adv = adv.reshape(t, 1, 1)
    oh_A = oh_A.reshape(t, 1, num_act)
    a2 = a2.reshape(t, 1, num_act)

    J = -np.sum(adv * oh_A * np.log(a2 + eps))
    return J


def main():
    rets, costs = [1], [1]
    for i in range(epochs):
        St, _ = env.reset()
        St = St.reshape(1, num_obs)

        done = False
        t = 0
        A, S, R = [], [], []

        while not done:
            p = policy_forward(model, St, t)
            S.append(St)

            At = np.random.choice(a=np.arange(num_act), p=p)  # action at timestep t
            St, Rt, term, trunc, _ = env.step(At.tolist())
            St = St.reshape(1, num_obs)

            R.append(Rt)
            A.append(At)
            t += 1
            done = term or trunc

        A, S, R = np.array(A), np.array(S), np.array(R)
        J = policy_backward(model, A, S, R, t, rets)

        grads = grad(policy_backward)(model, A, S, R, t, rets)
        param_update = rmsprop_update(grads)

        model["W1"] -= param_update["W1"]
        model["W2"] -= param_update["W2"]
        grad_buffer.clear()

        costs.append(J)
        rets.append(np.sum(R))

        if i % save_interval == 0:
            print(
                "cost: ",
                np.mean(np.array(costs[-save_interval:])),
                "avg_ret: ",
                np.mean(np.array(rets[-save_interval:])),
            )
            # np.save("ckpt/w1_lunar.npy", model["W1"])
            # np.save("ckpt/w2_lunar.npy", model["W2"])

    env.close()


if __name__ == "__main__":
    main()
