# Speaker notes: XOR and the three branches of calculus

1. **Set up the puzzle.** Show the four XOR points: the output is 1 when the
   inputs differ. A single straight line cannot separate the two classes, so
   the network needs a hidden layer.
2. **Differential calculus (backpropagation).** Trace one gradient backward:
   output error -> output weights -> hidden activations -> hidden weights.
   In `gradients`, point out `dz2`, `da1`, and `dz1`; each is a chain-rule
   multiplication. The hidden `tanh` derivative is `1 - a1**2`; at the output,
   the sigmoid derivative `a2*(1-a2)` cancels with the binary cross-entropy
   derivative, leaving `dz2 = (a2-y)/m`.
3. **Vector calculus (many partials at once).** Explain that a weight matrix is
   a vector of parameters in flattened form. Matrix products such as
   `a1.T @ dz2` calculate all corresponding partial derivatives together.
4. **Integral calculus (averaging).** The loss uses the mean across the four
   samples. This finite average estimates expected loss under the data
   distribution, analogous to an integral. Full-batch gradients use every
   sample on every step, keeping this classroom example deterministic.
5. **Momentum over time.** Compare the loss curves. The update
   `v_t = beta*v_(t-1) + grad_t` retains a decaying sum of earlier gradients,
   like a discrete integral with fading memory. The parameter step uses this
   accumulated direction.
6. **Interpret the figure.** The right panel shows the nonlinear boundary
   learned by the momentum model. The colored regions indicate predicted
   classes; the four markers are the training data.
7. **Close carefully.** Momentum often reaches a low loss sooner, but this is
   not a universal guarantee; it depends on initialization and hyperparameters.
   The seed is fixed here so the two curves and predictions can be reproduced.
