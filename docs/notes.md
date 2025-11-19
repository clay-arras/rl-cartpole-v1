
<!-- use np.vectorize
batch up policy backwards with vectorize instead of doing through it with batch matmul -->

## need to do: 
- get rid of policy_backwards and move to autograd
- need to figure out how to incorporate the advantage in the loss

define a forward function, and a loss function






--- note to self: once I have the policy implemented it will be a good idea to spin up some pytest tests