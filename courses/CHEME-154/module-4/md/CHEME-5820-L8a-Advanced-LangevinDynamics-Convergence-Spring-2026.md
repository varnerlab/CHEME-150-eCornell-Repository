# Advanced: Langevin Dynamics and Convergence on the Hopfield Energy

This notebook provides a deeper treatment of the stochastic attention sampler introduced in L8a. We derive the continuous-time SDE formulation, state the regularity conditions on the modern Hopfield energy, and establish convergence of the Unadjusted Langevin Algorithm (ULA) to the Boltzmann distribution.

> __Learning Objectives:__
>
> By the end of this notebook, you should be able to:
>
> * __Derive the continuous-time SDE:__ Starting from the modern Hopfield energy, write the overdamped Langevin SDE and explain how the discrete-time ULA approximates it.
> * __Verify regularity conditions:__ Check that the modern Hopfield energy satisfies the Lipschitz gradient and dissipativity conditions required for Langevin convergence.
> * __State the convergence guarantee:__ Describe the convergence rate of ULA on the Hopfield energy in terms of the step size, inverse temperature, and spectral properties of the memory matrix, and identify the strong log-concavity regime $\beta\cdot\sigma_{\max}^{2}(\mathbf{X}) < 2$.

Let's get started!

___

## Continuous-Time Formulation
The stochastic attention update in L8a is a discretization of an overdamped Langevin stochastic differential equation (SDE). We start from the continuous-time formulation and then discretize.

> **Overdamped Langevin SDE.** Given an energy function $E:\mathbb{R}^{N}\to\mathbb{R}$ and inverse temperature $\beta > 0$, the overdamped Langevin diffusion is the SDE:
> $$d\mathbf{S}_{t} = -\nabla E(\mathbf{S}_{t})\,dt + \sqrt{\frac{2}{\beta}}\,d\mathbf{B}_{t}$$
> where $\mathbf{B}_{t}$ is a standard $N$-dimensional Brownian motion. Under regularity conditions on $E$, the process $\mathbf{S}_{t}$ has a unique stationary distribution equal to the Boltzmann distribution:
> $$\pi_{\beta}(\mathbf{s}) = \frac{1}{Z(\beta)}\exp\left(-\beta\,E(\mathbf{s})\right)$$

The drift term $-\nabla E(\mathbf{S}_{t})$ pushes the process toward energy minima, while the diffusion term $\sqrt{2/\beta}\,d\mathbf{B}_{t}$ adds thermal fluctuations that enable exploration.

> **Substituting the Hopfield energy.** For the modern Hopfield energy with $\nabla E(\mathbf{s}) = \mathbf{s} - \mathbf{T}(\mathbf{s})$, the SDE becomes:
> $$d\mathbf{S}_{t} = \left(\mathbf{T}(\mathbf{S}_{t}) - \mathbf{S}_{t}\right)dt + \sqrt{\frac{2}{\beta}}\,d\mathbf{B}_{t}$$
> The drift decomposes into a contraction toward the origin ($-\mathbf{S}_{t}\,dt$) and an attention pull toward the memory-weighted average ($\mathbf{T}(\mathbf{S}_{t})\,dt$).

The Euler-Maruyama discretization with step size $\eta$ gives:
$$\mathbf{s}^{t+1} = \mathbf{s}^{t} - \eta\nabla E(\mathbf{s}^{t}) + \sqrt{\frac{2\eta}{\beta}}\,\boldsymbol{\xi}^{t}$$
where $\boldsymbol{\xi}^{t}\sim\mathcal{N}(\mathbf{0},\mathbf{I}_{N})$. This is the Unadjusted Langevin Algorithm (ULA), which is Algorithm 1 from L8a.

___

## Regularity Conditions
Convergence of ULA requires regularity conditions on the energy function. We verify these for the modern Hopfield energy.

> **Condition 1: Lipschitz continuous gradient.** The gradient $\nabla E$ is $L$-Lipschitz if there exists $L > 0$ such that for all $\mathbf{s},\mathbf{s}'\in\mathbb{R}^{N}$:
> $$\lVert\nabla E(\mathbf{s}) - \nabla E(\mathbf{s}')\rVert_{2} \leq L\lVert\mathbf{s} - \mathbf{s}'\rVert_{2}$$
>
> For the Hopfield energy, $\nabla E(\mathbf{s}) = \mathbf{s} - \mathbf{X}\operatorname{softmax}(\beta\mathbf{X}^{\top}\mathbf{s})$. The first term is linear with Lipschitz constant $1$. The second term $\mathbf{T}(\mathbf{s})$ is the composition of the linear map $\beta\mathbf{X}^{\top}$ with softmax and then $\mathbf{X}$. Since softmax is Lipschitz with constant $1$ in the $\ell_{2}$ norm, we get:
> $$L \leq 1 + \beta\,\sigma_{\max}^{2}(\mathbf{X})$$
> where $\sigma_{\max}(\mathbf{X})$ is the largest singular value of $\mathbf{X}$. The gradient is globally Lipschitz, which is a strong regularity property.

The Lipschitz constant controls the maximum curvature of the energy landscape and determines the largest stable step size for ULA.

> **Condition 2: Dissipativity.** The energy $E$ is dissipative if there exist constants $a > 0$ and $b \geq 0$ such that for all $\mathbf{s}\in\mathbb{R}^{N}$:
> $$\langle\nabla E(\mathbf{s}),\mathbf{s}\rangle \geq a\lVert\mathbf{s}\rVert_{2}^{2} - b$$
>
> For the Hopfield energy:
> $$\langle\nabla E(\mathbf{s}),\mathbf{s}\rangle = \lVert\mathbf{s}\rVert_{2}^{2} - \langle\mathbf{T}(\mathbf{s}),\mathbf{s}\rangle \geq \lVert\mathbf{s}\rVert_{2}^{2} - M\lVert\mathbf{s}\rVert_{2}$$
> where $M = \max_{i}\lVert\mathbf{m}_{i}\rVert_{2}$ since $\mathbf{T}(\mathbf{s})$ lies in the convex hull of the memories with $\lVert\mathbf{T}(\mathbf{s})\rVert_{2} \leq M$. By Young's inequality:
> $$\langle\nabla E(\mathbf{s}),\mathbf{s}\rangle \geq \frac{1}{2}\lVert\mathbf{s}\rVert_{2}^{2} - \frac{1}{2}M^{2}$$
> So dissipativity holds with $a = 1/2$ and $b = M^{2}/2$.

Dissipativity ensures that the Langevin process does not escape to infinity: the energy gradient always pushes large states back toward the origin.

___

## Convergence of ULA on the Hopfield Energy
With the regularity conditions verified, we can state the convergence guarantee for ULA.

### Strong log-concavity regime
The Boltzmann distribution $\pi_{\beta}(\mathbf{s}) \propto \exp(-\beta\,E(\mathbf{s}))$ is log-concave if and only if $\beta\,E(\mathbf{s})$ is convex. The Hessian of the Hopfield energy is:
$$\nabla^{2}E(\mathbf{s}) = \mathbf{I}_{N} - \beta\,\mathbf{X}\left(\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^{\top}\right)\mathbf{X}^{\top}$$
where $\mathbf{p} = \operatorname{softmax}(\beta\mathbf{X}^{\top}\mathbf{s})$. The matrix $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^{\top}$ is the softmax covariance and is positive semidefinite with spectral norm at most $1/4$. Therefore:
$$\nabla^{2}E(\mathbf{s}) \succeq \left(1 - \frac{\beta}{4}\sigma_{\max}^{2}(\mathbf{X})\right)\mathbf{I}_{N}$$

A sharper bound uses the fact that the spectral norm of the softmax covariance is at most $1/2$ (since the largest eigenvalue of $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^{\top}$ is bounded by the variance of a Bernoulli variable).

> **Strong convexity condition.** The energy $E(\mathbf{s})$ is $\mu$-strongly convex with:
> $$\mu = 1 - \frac{\beta}{2}\sigma_{\max}^{2}(\mathbf{X})$$
> This is positive when $\beta\cdot\sigma_{\max}^{2}(\mathbf{X}) < 2$. In this regime, the Boltzmann distribution $\pi_{\beta}$ is strongly log-concave with parameter $\beta\mu$.

Strong log-concavity guarantees the Boltzmann distribution is unimodal and ULA converges exponentially fast.

> **Convergence rate (strongly log-concave regime).** When $\beta\cdot\sigma_{\max}^{2}(\mathbf{X}) < 2$, ULA with step size $\eta \leq 2/(\beta L)$ satisfies:
> $$W_{2}(\mu_{t},\pi_{\beta}) \leq (1 - \eta\mu)^{t}\,W_{2}(\mu_{0},\pi_{\beta}) + C\sqrt{\eta}$$
> where $W_{2}$ is the 2-Wasserstein distance, $\mu_{t}$ is the distribution of $\mathbf{s}^{t}$, $L = 1 + \beta\sigma_{\max}^{2}(\mathbf{X})$ is the Lipschitz constant, $\mu = 1 - \frac{\beta}{2}\sigma_{\max}^{2}(\mathbf{X})$ is the strong convexity parameter, and $C$ depends on $N$, $\beta$, and $L$.

The first term shows exponential contraction at rate $1 - \eta\mu$ per step. The second term is the discretization bias from using finite step size $\eta > 0$. As $\eta \to 0$, the bias vanishes but convergence slows.

### Beyond strong log-concavity
When $\beta\cdot\sigma_{\max}^{2}(\mathbf{X}) \geq 2$, the energy is no longer strongly convex and the Boltzmann distribution may be multimodal. Convergence still occurs due to dissipativity, but the rate degrades.

> **General convergence (dissipative regime).** Under the Lipschitz gradient and dissipativity conditions verified above, ULA converges to $\pi_{\beta}$ at a polynomial rate. The convergence guarantee uses a Lyapunov function argument: dissipativity ensures the iterates have bounded moments, and the Lipschitz condition controls the discretization error.

In the multimodal regime, the sampler may require many steps to transition between modes (metastability). This is analogous to the slow mixing of Gibbs sampling in Boltzmann machines with frustrated interactions.

___

## Summary

This notebook derived the continuous-time SDE, verified regularity conditions, and established convergence guarantees for the Langevin sampler on the modern Hopfield energy.

> __Key Takeaways:__
>
> * __The Hopfield energy has strong regularity:__ The gradient is globally Lipschitz with constant $L = 1 + \beta\sigma_{\max}^{2}(\mathbf{X})$, and the energy is dissipative with parameters $a = 1/2$ and $b = M^{2}/2$. These properties ensure ULA is well-behaved.
> * __Strong log-concavity holds when $\beta\cdot\sigma_{\max}^{2}(\mathbf{X}) < 2$:__ In this regime, the Boltzmann distribution is unimodal and ULA converges exponentially fast. The strong convexity parameter $\mu = 1 - \frac{\beta}{2}\sigma_{\max}^{2}(\mathbf{X})$ controls the convergence rate.
> * __Dissipativity ensures convergence beyond the log-concave regime:__ Even when the energy is not strongly convex and the distribution is multimodal, dissipativity guarantees bounded moments and polynomial convergence, though mode transitions may be slow.

___

## References
* [Ramsauer, H., Schafl, B., Lehner, J., et al. (2020). Hopfield Networks is All You Need. ArXiv, abs/2008.02217.](https://arxiv.org/abs/2008.02217)
* [Durmus, A. & Moulines, E. (2017). Nonasymptotic convergence analysis for the unadjusted Langevin algorithm. The Annals of Applied Probability, 27(3), 1551-1587.](https://arxiv.org/abs/1507.05021)
* [Dalalyan, A.S. (2017). Theoretical guarantees for approximate sampling from smooth and log-concave densities. Journal of the Royal Statistical Society: Series B, 79(3), 651-676.](https://arxiv.org/abs/1412.7392)
* [Varner, J.D. et al. (2026). Stochastic Attention via Langevin Dynamics on the Modern Hopfield Energy Landscape.](https://github.com/varnerlab/stochastic-attention-study-paper)

___


