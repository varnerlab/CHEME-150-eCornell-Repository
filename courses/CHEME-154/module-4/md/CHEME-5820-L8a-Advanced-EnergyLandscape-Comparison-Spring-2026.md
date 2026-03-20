# Advanced: Energy Landscape Comparison

This notebook provides a side-by-side comparison of the three energy-based architectures from weeks 6 to 8: classical Hopfield networks, modern Hopfield networks, and Boltzmann machines. We show how each model is a special case of the energy-based model (EBM) framework introduced in L8a.

> __Learning Objectives:__
>
> By the end of this notebook, you should be able to:
>
> * __Compare energy functions:__ Identify the energy function, state space, and inference strategy for each of the three architectures and explain how they differ.
> * __Connect gradients to update rules:__ Derive the gradient of each energy function and show how the update rule in each model follows from gradient descent or stochastic sampling on that energy.
> * __Recognize the EBM abstraction:__ Explain how all three models fit the general EBM pattern of energy function, Boltzmann distribution, and inference via minimization or sampling.

Let's get started!

___

## Setup and Notation
We use the following notation throughout this notebook:

> **Common notation:**
> * $N$: state dimension (number of nodes or features)
> * $K$: number of stored memories (or patterns)
> * $\mathbf{s}$: state vector
> * $\beta > 0$: inverse temperature parameter
> * $E(\mathbf{s})$: energy function
> * $P_{\beta}(\mathbf{s}) \propto \exp(-\beta\,E(\mathbf{s}))$: Boltzmann distribution induced by the energy

Each model specifies a particular choice of $E$, state space, and inference procedure.

___

## Energy Functions
The following table summarizes the energy function, state space, and key objects for each architecture.

| | **Classical Hopfield** | **Modern Hopfield** | **Boltzmann Machine** |
|---|---|---|---|
| **State space** | $\mathbf{s}\in\{-1,1\}^{N}$ | $\mathbf{s}\in\mathbb{R}^{N}$ | $\mathbf{s}\in\{-1,1\}^{N}$ |
| **Energy** | $E(\mathbf{s}) = -\frac{1}{2}\mathbf{s}^{\top}\mathbf{W}\mathbf{s} - \mathbf{b}^{\top}\mathbf{s}$ | $E(\mathbf{s}) = -\operatorname{lse}_{\beta}(\mathbf{X}^{\top}\mathbf{s}) + \frac{1}{2}\lVert\mathbf{s}\rVert_{2}^{2} + C$ | $E(\mathbf{s}) = -\frac{1}{2}\mathbf{s}^{\top}\mathbf{W}\mathbf{s} - \mathbf{b}^{\top}\mathbf{s}$ |
| **Parameters** | $\mathbf{W}\in\mathbb{R}^{N\times N}$, $\mathbf{b}\in\mathbb{R}^{N}$ | $\mathbf{X}\in\mathbb{R}^{N\times K}$, $\beta > 0$ | $\mathbf{W}\in\mathbb{R}^{N\times N}$, $\mathbf{b}\in\mathbb{R}^{N}$ |
| **Inference** | Deterministic minimization | Deterministic minimization | Stochastic sampling |
| **Goal** | Retrieve nearest stored pattern | Retrieve nearest stored pattern | Sample from $P_{\beta}(\mathbf{s})$ |

where $C = \frac{1}{\beta}\log K + \frac{1}{2}M^{2}$ is a constant that ensures the modern Hopfield energy is non-negative, and $M = \max_{i}\lVert\mathbf{m}_{i}\rVert_{2}$.

The classical Hopfield network and Boltzmann machine share the same energy function. Their difference is entirely in how they use it: Hopfield descends, Boltzmann samples.

___

## Gradients and Update Rules
The update rule in each model is derived from the gradient (or discrete analog) of the energy function.

> **Classical Hopfield gradient and update.**
> The energy is quadratic in $\mathbf{s}$. The change in energy when flipping node $i$ from $s_{i}$ to $-s_{i}$ is:
> $$\Delta E_{i} = -2s_{i}\left(\sum_{j\neq i}w_{ij}s_{j} + b_{i}\right) = -2s_{i}\,h_{i}$$
> where $h_{i} = \sum_{j}w_{ij}s_{j} + b_{i}$ is the local field at node $i$. The asynchronous update rule sets $s_{i} \gets \operatorname{sign}(h_{i})$, which guarantees $\Delta E_{i} \leq 0$.

This is a coordinate-wise greedy descent on the energy surface.

> **Modern Hopfield gradient and update.**
> The gradient of the LSE energy is:
> $$\nabla E(\mathbf{s}) = \mathbf{s} - \mathbf{X}\operatorname{softmax}(\beta\mathbf{X}^{\top}\mathbf{s}) = \mathbf{s} - \mathbf{T}(\mathbf{s})$$
> The retrieval update $\mathbf{s}^{t+1} = \mathbf{T}(\mathbf{s}^{t})$ is gradient descent with step size $\eta = 1$. The relaxed update $(1-\eta)\mathbf{s}^{t} + \eta\,\mathbf{T}(\mathbf{s}^{t})$ uses step size $\eta\in(0,1]$.

This is full gradient descent on a smooth energy surface.

> **Boltzmann machine update.**
> The Boltzmann machine uses the same energy as the classical Hopfield network but updates stochastically. Each node samples its next state from:
> $$P(s_{i} = 1 \mid s_{\lnot i}) = \frac{1}{1 + \exp(-2\beta\,h_{i})}$$
> This is Gibbs sampling from the conditional distribution, which converges to the joint Boltzmann distribution $P_{\beta}(\mathbf{s}) \propto \exp(-\beta\,E(\mathbf{s}))$.

The table below summarizes the update rules:

| | **Update rule** | **Type** |
|---|---|---|
| **Classical Hopfield** | $s_{i} \gets \operatorname{sign}(h_{i})$ | Deterministic, coordinate-wise |
| **Modern Hopfield** | $\mathbf{s} \gets \mathbf{X}\operatorname{softmax}(\beta\mathbf{X}^{\top}\mathbf{s})$ | Deterministic, full-state |
| **Boltzmann machine** | $s_{i} \sim \operatorname{Bernoulli}(\sigma(2\beta\,h_{i}))$ | Stochastic, coordinate-wise |

___

## The EBM Abstraction
All three models fit the same abstract pattern:

> **Energy-Based Model pattern:**
> 1. **Define** an energy function $E(\mathbf{s})$ over a state space.
> 2. **Induce** a probability distribution $P_{\beta}(\mathbf{s}) \propto \exp(-\beta\,E(\mathbf{s}))$.
> 3. **Perform inference** by either minimizing $E$ (retrieval) or sampling from $P_{\beta}$ (generation).

The choice of energy function determines the expressiveness of the model (binary vs. continuous, linear vs. log-sum-exp). The choice of inference strategy determines the task (retrieval vs. generation). The inverse temperature $\beta$ controls the sharpness of the distribution.

The stochastic attention sampler introduced in L8a extends this pattern by applying Langevin dynamics to the modern Hopfield energy, enabling *continuous-state stochastic sampling* with the same energy that previously supported only deterministic retrieval.

| | **Classical Hopfield** | **Modern Hopfield** | **Boltzmann Machine** | **Stochastic Attention** |
|---|---|---|---|---|
| **State** | Binary | Continuous | Binary | Continuous |
| **Energy** | Quadratic | LSE | Quadratic | LSE |
| **Inference** | Minimize | Minimize | Sample (Gibbs) | Sample (Langevin) |
| **Output** | Single pattern | Single pattern | Distribution | Distribution |

___

## Summary

This notebook compared the energy functions, gradients, and update rules of the three architectures from weeks 6 to 8.

> __Key Takeaways:__
>
> * **Same energy, different inference:** Classical Hopfield and Boltzmann machines share the same quadratic energy function but differ in inference strategy. Hopfield uses deterministic descent (retrieval), while Boltzmann uses stochastic sampling (generation).
> * **Modern Hopfield generalizes the energy:** The LSE energy enables continuous states, exponential storage capacity, and fast convergence via gradient descent. Its gradient has a closed form involving softmax, which is the retrieval map.
> * **EBMs provide the unifying abstraction:** All four methods (including stochastic attention) fit the pattern of energy definition, Boltzmann distribution, and inference via minimization or sampling. The inverse temperature $\beta$ controls the balance between sharp retrieval and diverse generation.

___

## References
* [Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences, 79(8), 2554-2558.](https://www.pnas.org/doi/10.1073/pnas.79.8.2554)
* [Ramsauer, H., Schafl, B., Lehner, J., et al. (2020). Hopfield Networks is All You Need. ArXiv, abs/2008.02217.](https://arxiv.org/abs/2008.02217)
* [LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M.A., & Huang, F.J. (2006). A Tutorial on Energy-Based Learning. In Predicting Structured Data, MIT Press.](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)

___


