from collections import defaultdict
import gymnasium as gym
import autograd.numpy as np
from autograd import grad

resume = False
hidden_layer = 128

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
    model["W1"] = np.random.rand(num_obs, hidden_layer) / np.sqrt(num_obs)
    model["W2"] = np.random.rand(hidden_layer, num_act) / np.sqrt(hidden_layer)


def softmax(x):
    z = np.max(x, axis=1, keepdims=True)
    return np.exp(x - z) / np.sum(np.exp(x - z), axis=1, keepdims=True)


def relu(x):  # PURE
    a = np.maximum(x, 0)
    return a


def discount_Rt(rewards):
    """
    np.outer(np.pow(gamma, pows), np.pow(gamma, -pows)) produces the following for t=3
    [[  1, y^-1, y^-2]
     [  y,    1, y^-1]
     [y^2,    y,    1]]

    we mask it and then matmul for discounted cumsum
    """
    t = rewards.shape[-1]
    pows = np.arange(t)
    mask = np.outer(np.pow(gamma, pows), np.pow(gamma, -pows)) * np.tri(t)

    return rewards.reshape(1, t) @ mask


def rmsprop_update(grads):
    param_update = {}
    updated_cache = {}
    for k, g in grads.items():
        # grad = g.mean(axis=0)
        grad = g
        updated_cache[k] = beta * rmsprop_cache[k] + (1 - beta) * grad**2
        param_update[k] = (lr / np.sqrt(updated_cache[k] + eps)) * grad
    return param_update


def policy_forward(model, St, t):
    z1 = St @ model["W1"]
    a1 = relu(z1)
    z2 = a1 @ model["W2"]
    a2 = softmax(z2)

    # grad_buffer["z1"][t] = z1
    # grad_buffer["a1"][t] = a1
    # grad_buffer["z2"][t] = z2
    grad_buffer["a2"][t] = a2

    return a2[0]


def policy_backward(model, A, S, R, t, rets):
    S = S.reshape(500, num_obs)
    z1 = S @ model["W1"]
    a1 = relu(z1)
    z2 = a1 @ model["W2"]
    a2 = softmax(z2)

    Gt = discount_Rt(R)
    baseline = np.mean(np.array(rets))
    R = Gt - baseline

    # for k in grad_buffer:
    #     grad_buffer[k] = np.array(grad_buffer[k])

    oh_A = np.expand_dims(A, axis=1) == np.expand_dims(np.arange(num_act), axis=0)

    R = R.reshape(500, 1, 1)
    oh_A = oh_A.reshape(500, 1, num_act)

    J = -np.sum(R * oh_A * np.log(grad_buffer["a2"] + eps))

    # dJdz2 = R * (grad_buffer["a2"] - oh_A)
    # dJdW2 = grad_buffer["a1"].swapaxes(-1, -2) @ dJdz2

    # dJda1 = dJdz2 @ model["W2"].T
    # dJdz1 = dJda1 * (grad_buffer["z1"] > 0)
    # dJdW1 = S.swapaxes(-1, -2) @ dJdz1

    # return {"dJdW1": dJdW1, "dJdW2": dJdW2}, J
    return J


def main():

    rets, costs = [1], [1]
    for i in range(epochs):
        St, _ = env.reset()
        St = St.reshape(1, num_obs)

        done = False
        t = 0
        # A, S, R = [], [], []
        A = np.zeros((500,))
        S = np.zeros((500, 1, 4))
        R = np.zeros((500,))

        grad_buffer["a2"] = np.zeros((500, 1, 2))
        # print(grad_buffer["a2"].shape)

        while not done:
            p = policy_forward(model, St, t)
            S[t] = St

            at = np.random.choice(a=np.arange(num_act), p=p)  # action at timestep t
            St, Rt, term, trunc, _ = env.step(at.tolist())
            St = St.reshape(1, num_obs)

            A[t] = at
            R[t] = Rt

            t += 1
            done = term or trunc

        # A, S, R = np.array(A), np.array(S), np.array(R)
        # print(A.shape, S.shape, R.shape)

        # probs = np.array(grad_buffer["a2"])
        J = policy_backward(model, A, S, R, t, rets)
        grads = grad(policy_backward)(model, A, S, R, t, rets)
        # param_update = rmsprop_update(grads)

        model["W1"] -= lr * grads["W1"]
        model["W2"] -= lr * grads["W2"]
        # model["W1"] -= param_update["W1"]
        # model["W2"] -= param_update["W2"]

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
