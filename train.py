from collections import defaultdict
import gymnasium as gym
import autograd.numpy as np
# import jax.numpy as np
# import jax


def softmax(x): 
    z = np.max(x, axis=1, keepdims=True)
    return np.exp(x - z) / np.sum(np.exp(x - z), axis=1, keepdims=True)


def relu(x): # PURE
    a = np.maximum(x, 0)
    return a


def discount_Rt(rewards, gamma):  # rews is (t, )
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


def rmsprop_update(grads, rmsprop_cache, beta, lr, eps):
    param_update = {}
    updated_cache = {}
    for k, g in grads.items():
        grad = g.mean(axis=0)
        updated_cache[k] = beta * rmsprop_cache[k] + (1 - beta) * grad**2
        param_update[k] = (lr / np.sqrt(updated_cache[k] + eps)) * grad
    return param_update, updated_cache


# @jax.jit
def policy_forward(St, model, grad_buffer, num_obs, num_act):
#    assert St.shape == (1, num_obs)
    z1 = St @ model["W1"]
    a1 = relu(z1)
    z2 = a1 @ model["W2"]
    a2 = softmax(z2)

    grad_buffer["z1"].append(z1)
    grad_buffer["a1"].append(a1)
    grad_buffer["z2"].append(z2)
    grad_buffer["a2"].append(a2)

    # assert a2.shape == (1, num_act)
    return a2[0], grad_buffer


def policy_backward(A, S, R, t, rets, grad_buffer, model, gamma, eps, num_act):
    Gt = discount_Rt(R, gamma)
    baseline = np.mean(np.array(rets))
    R = Gt - baseline

    for k in grad_buffer:
        grad_buffer[k] = np.array(grad_buffer[k])

    oh_A = np.expand_dims(A, axis=1) == np.expand_dims(np.arange(num_act), axis=0)

    R = R.reshape(t, 1, 1)
    oh_A = oh_A.reshape(t, 1, num_act)

    J = -np.sum(R * oh_A * np.log(grad_buffer["a2"] + eps))

    dJdz2 = R * (grad_buffer["a2"] - oh_A)
    dJdW2 = grad_buffer["a1"].swapaxes(-1, -2) @ dJdz2

    dJda1 = dJdz2 @ model["W2"].T
    dJdz1 = dJda1 * (grad_buffer["z1"] > 0)
    dJdW1 = S.swapaxes(-1, -2) @ dJdz1

    return {"dJdW1": dJdW1, "dJdW2": dJdW2}, J


def main():
    epochs = int(100)
    gamma = 0.99
    lr = 5e-4
    eps = 1e-5
    beta = 0.90
    save_interval = 500
    
    seed = 42
    # key = jax.random.key(seed)
    
    env = gym.make("CartPole-v1")
    resume = False
    
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.n
    hidden_layer = 128
    
    grad_buffer = defaultdict(lambda: [])
    rmsprop_cache = defaultdict(lambda: 0)
    model = {}
    if resume:
        model["W1"] = np.load("ckpt/w1_lunar.npy")
        model["W2"] = np.load("ckpt/w2_lunar.npy")
    else:
        # model["W1"] = jax.random.uniform(key, (num_obs, hidden_layer)) / np.sqrt(
        #     num_obs
        # )  # he initialization
        # model["W2"] = jax.random.uniform(key, (hidden_layer, num_act)) / np.sqrt(hidden_layer)
        model["W1"] = np.random.rand(num_obs, hidden_layer) / np.sqrt(
            num_obs
        )  # he initialization
        model["W2"] = np.random.rand(hidden_layer, num_act) / np.sqrt(hidden_layer)
    
    rets, costs = [1], [1]
    for i in range(epochs):
        St, _ = env.reset()
        St = St.reshape(1, num_obs)

        done = False
        t = 0
        A, S, R = [], [], []

        while not done:
            p, grad_buffer = policy_forward(St, model, grad_buffer, num_obs, num_act)
            S.append(St)

            at = np.random.choice(a=np.arange(num_act), p=p)  # action at timestep t
            St, Rt, term, trunc, _ = env.step(at.tolist())
            St = St.reshape(1, num_obs)

            A.append(at)
            R.append(Rt)

            t += 1
            done = term or trunc

        A, S, R = np.array(A), np.array(S), np.array(R)
        grads, J = policy_backward(A, S, R, t, rets, grad_buffer, model, gamma, eps, num_act)
        param_update, rmsprop_cache = rmsprop_update(grads, rmsprop_cache, beta, lr, eps)

        model["W1"] -= param_update["dJdW1"]
        model["W2"] -= param_update["dJdW2"]

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
