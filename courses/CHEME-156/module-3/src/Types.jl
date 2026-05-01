abstract type AbstractHebbianMemoryModel end

"""
    MyHMemMemory

Rate-domain Hebbian Memory module from Limbacher and Legenstein (NeurIPS 2020).
Holds the dynamic association matrix `W` together with the constants of the
write rule (`γ_plus`, `γ_minus`, `w_max`).

### Fields
- `m::Int64`: dimension of the key-vector and value-vector (square memory).
- `W::Matrix{Float64}`: association matrix of shape `(m, m)`. Element `W[k, j]`
   stores the synaptic weight from pre-synaptic key-coordinate `j` to
   post-synaptic value-coordinate `k`.
- `γ_plus::Float64`:  positive constant scaling the Hebbian "fire together,
   wire together" term.
- `γ_minus::Float64`: positive constant scaling the Oja-style forgetting term.
- `w_max::Float64`: soft upper bound on entries of `W`.
"""
mutable struct MyHMemMemory <: AbstractHebbianMemoryModel
    m::Int64
    W::Matrix{Float64}
    γ_plus::Float64
    γ_minus::Float64
    w_max::Float64

    MyHMemMemory() = new()
end

abstract type AbstractSpikingHebbianMemoryModel end

"""
    MySpikingHMemNetwork

Spiking Hebbian Memory Network from
[Limbacher, Özdenizci, and Legenstein (2022)](https://arxiv.org/abs/2205.11276),
stripped down to the essentials needed for one-shot key-value recall:

* A `key-layer` of `ell_key` LIF neurons.
* A `value-layer` of `ell_value` LIF neurons.
* An `association matrix` `W_assoc` of shape `(ell_value, ell_key)` connecting
  the key-layer (pre-synaptic) to the value-layer (post-synaptic).
* Shared LIF dynamical hyperparameters `(ϑ, τ_m, Δ_abs)` and shared trace
  time constant `τ_trace`.
* Hebbian-plasticity hyperparameters `(γ_plus, γ_minus, w_max)`.

The encoders `W^{s,key}, W^{s,val}, W^{r,key}` from the full paper are absent
here; we drive the key- and value-layers directly with rate-coded input spike
trains, which keeps the lab focused on the Hebbian write/read dynamics.

### Fields
- `ell_key::Int64`: number of key-layer LIF neurons (pre-synaptic).
- `ell_value::Int64`: number of value-layer LIF neurons (post-synaptic).
- `W_assoc::Matrix{Float64}`: association matrix of shape `(ell_value, ell_key)`.
   `W_assoc[k, j]` is the synaptic weight from key-neuron `j` to value-neuron `k`.
- `ϑ::Float64`: firing threshold (shared across all LIF neurons).
- `τ_m::Float64`: membrane time constant.
- `τ_trace::Float64`: synaptic-trace time constant for κ.
- `Δt::Float64`: simulation time step.
- `Δ_abs::Int64`: absolute refractory period in time steps.
- `γ_plus::Float64`: Hebbian "fire together, wire together" gain.
- `γ_minus::Float64`: Oja-style forgetting gain.
- `w_max::Float64`: soft upper bound on entries of `W_assoc`.
"""
mutable struct MySpikingHMemNetwork <: AbstractSpikingHebbianMemoryModel
    ell_key::Int64
    ell_value::Int64
    W_assoc::Matrix{Float64}
    ϑ::Float64
    τ_m::Float64
    τ_trace::Float64
    Δt::Float64
    Δ_abs::Int64
    γ_plus::Float64
    γ_minus::Float64
    w_max::Float64

    MySpikingHMemNetwork() = new()
end
