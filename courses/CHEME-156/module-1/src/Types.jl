"""
    MySisoLegSHippoModel

Discretized single-input single-output structured state space model with
HiPPO-LegS initialization.

The continuous-time system is

    dx/dt = A x + B u
        y = C x + D u

where `A ∈ ℝ^{h×h}` and `B ∈ ℝ^{h×1}` are the LegS HiPPO matrices, `C ∈ ℝ^{1×h}`
is a learned readout row vector, and `D ∈ ℝ` is a scalar feedforward term
(fixed to 0 for the lectures). The model stores both the continuous matrices
`(A, B)` and the bilinear-discretized matrices `(Ā, B̄)` used for training
and inference.

# Fields
- `h::Int`: hidden-state dimension (number of LegS basis coefficients).
- `Δt::Float64`: time step used to discretize the continuous system.
- `A::Matrix{Float64}`: continuous-time state matrix, size `(h, h)`.
- `B::Matrix{Float64}`: continuous-time input matrix, size `(h, 1)`.
- `Ā::Matrix{Float64}`: discrete-time state matrix, size `(h, h)`.
- `B̄::Matrix{Float64}`: discrete-time input matrix, size `(h, 1)`.
- `C::Matrix{Float64}`: readout row vector, size `(1, h)`; updated by `fit_C!`.
- `D::Float64`: feedforward scalar; set to 0 unless the task requires it.
- `x₀::Vector{Float64}`: initial hidden state, length `h`; defaults to zeros.
"""
mutable struct MySisoLegSHippoModel
    h::Int
    Δt::Float64
    A::Matrix{Float64}
    B::Matrix{Float64}
    Ā::Matrix{Float64}
    B̄::Matrix{Float64}
    C::Matrix{Float64}
    D::Float64
    x₀::Vector{Float64}
end

"""
    MyMimoLegSHippoModel

Discretized multi-input multi-output (MIMO) structured state space model with
block-diagonal HiPPO-LegS initialization.

The continuous-time system is

    dx/dt = A x + B u
        y = C x + D u

with vector input `u ∈ ℝ^{d_in}` and vector output `y ∈ ℝ^{d_out}`. The block-
diagonal construction runs one independent LegS filter per input channel: the
total hidden-state dimension is `H = h * d_in`, `A ∈ ℝ^{H×H}` is block-diagonal
with `d_in` copies of the `h×h` LegS state matrix, and `B ∈ ℝ^{H×d_in}` is
block-rectangular so that input channel `j` drives only its own `h`-dimensional
block of the hidden state. The readout `C ∈ ℝ^{d_out×H}` is dense, so every
output channel can mix information from every input channel's history.

# Fields
- `h::Int`: per-channel hidden-state dimension (LegS basis order).
- `d_in::Int`: number of input channels.
- `d_out::Int`: number of output channels.
- `Δt::Float64`: time step used to discretize the continuous system.
- `A::Matrix{Float64}`: continuous-time state matrix, size `(h*d_in, h*d_in)`.
- `B::Matrix{Float64}`: continuous-time input matrix, size `(h*d_in, d_in)`.
- `Ā::Matrix{Float64}`: discrete-time state matrix, size `(h*d_in, h*d_in)`.
- `B̄::Matrix{Float64}`: discrete-time input matrix, size `(h*d_in, d_in)`.
- `C::Matrix{Float64}`: readout matrix, size `(d_out, h*d_in)`; updated by `fit_C!`.
- `D::Matrix{Float64}`: feedforward matrix, size `(d_out, d_in)`.
- `x₀::Vector{Float64}`: initial hidden state, length `h*d_in`; defaults to zeros.

The struct is mutable because training updates the readout matrix `C` in place
while leaving the state-space dynamics fixed.
"""
mutable struct MyMimoLegSHippoModel
    h::Int
    d_in::Int
    d_out::Int
    Δt::Float64
    A::Matrix{Float64}
    B::Matrix{Float64}
    Ā::Matrix{Float64}
    B̄::Matrix{Float64}
    C::Matrix{Float64}
    D::Matrix{Float64}
    x₀::Vector{Float64}
end

"""
    mutable struct MyFedBatchCHOParameters

Holds all kinetic parameters, yield coefficients, feed concentrations, and
feed state for a fed-batch CHO antibody production model. Ported verbatim
from the L12d LSTM lab so the simulation code drops in unchanged.
"""
mutable struct MyFedBatchCHOParameters

    # growth kinetics -
    mu_max::Float64
    K_glc::Float64
    K_gln::Float64
    K_I_lac::Float64
    K_I_amm::Float64
    k_d::Float64
    KD_lac::Float64    # half-saturation for lactate in death rate (mM)
    KD_amm::Float64    # half-saturation for ammonia in death rate (mM)

    # product formation (Luedeking-Piret: q_P = alpha_P * mu + beta_P) -
    alpha_P::Float64   # growth-associated coefficient (mg/gDW)
    beta_P::Float64    # non-growth-associated coefficient (mg/gDW/h)

    # yield coefficients -
    Y_X_glc::Float64
    Y_X_gln::Float64
    Y_P_glc::Float64
    Y_P_gln::Float64
    Y_lac_glc::Float64
    Y_amm_gln::Float64

    # feed concentrations -
    S_glc_f::Float64
    S_gln_f::Float64

    # feed policy -
    F_max::Float64
    Glc_min::Float64
    Glc_max::Float64
    feed_on::Float64
end
