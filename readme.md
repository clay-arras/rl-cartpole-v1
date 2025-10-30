# cartpole-v1 solution

notes: 
- used policy gradients, where gradients are updated at the end of each episode. 
- model is one hidden layer, 4 -> 8 -> 1. output is the probability of going left
- we're saving certain variables to do backprop on them later, after the end of the episode.
- we're using binary cross entropy

observations: 
- model does well on the first couple of training iterations and achieves a max avg_ret of 96. then it starts overfitting and starts getting smaller avg_ret (not sure how RL agents even overfit)
- the cost should be positive and we're optimizing it towards 0. I'm not sure why but the cost is negative here, I probably messed up some gradient calculation.