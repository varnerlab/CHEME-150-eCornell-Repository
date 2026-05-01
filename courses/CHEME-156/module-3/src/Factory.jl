"""
    build(modeltype::Type{MyHMemMemory};
        m::Int64, γ_plus::Float64 = 0.3, γ_minus::Float64 = 0.3,
        w_max::Float64 = 1.0) -> MyHMemMemory

Construct an empty `MyHMemMemory` with association matrix `W = 0` of shape
`(m, m)`. The defaults for `γ_plus`, `γ_minus`, and `w_max` follow Table 2 of
[Limbacher, Özdenizci, and Legenstein (2022)](https://arxiv.org/abs/2205.11276).

### Keyword arguments
- `m`: dimension of the key and value vectors.
- `γ_plus`: positive constant scaling the Hebbian "fire together, wire together"
   term in the write rule.
- `γ_minus`: positive constant scaling the Oja-style forgetting term.
- `w_max`: soft upper bound on entries of `W`.

### Returns
- a populated `MyHMemMemory` instance with `W` initialized to zeros.
"""
function build(modeltype::Type{MyHMemMemory};
    m::Int64,
    γ_plus::Float64 = 0.3,
    γ_minus::Float64 = 0.3,
    w_max::Float64 = 1.0)::MyHMemMemory

    model = modeltype()
    model.m       = m
    model.W       = zeros(Float64, m, m)
    model.γ_plus  = γ_plus
    model.γ_minus = γ_minus
    model.w_max   = w_max
    return model
end

"""
    build(modeltype::Type{MySpikingHMemNetwork};
        ell_key::Int64, ell_value::Int64,
        ϑ::Float64       = 0.1,
        τ_m::Float64     = 20.0,
        τ_trace::Float64 = 20.0,
        Δt::Float64      = 1.0,
        Δ_abs::Int64     = 3,
        γ_plus::Float64  = 0.3,
        γ_minus::Float64 = 0.3,
        w_max::Float64   = 1.0) -> MySpikingHMemNetwork

Construct a `MySpikingHMemNetwork` with empty association matrix
`W_assoc = 0` of shape `(ell_value, ell_key)`. The defaults match
Table 2 of [Limbacher, Özdenizci, and Legenstein (2022)](https://arxiv.org/abs/2205.11276):
firing threshold `ϑ = 0.1`, refractory period `Δ_abs = 3` ms, membrane and
trace time constants `τ_m = τ_trace = 20` ms, soft weight upper bound
`w_max = 1.0`, and Hebbian gains `γ_plus = γ_minus = 0.3`.

### Returns
- a populated `MySpikingHMemNetwork` instance with `W_assoc` initialized to zeros.
"""
function build(modeltype::Type{MySpikingHMemNetwork};
    ell_key::Int64,
    ell_value::Int64,
    ϑ::Float64       = 0.1,
    τ_m::Float64     = 20.0,
    τ_trace::Float64 = 20.0,
    Δt::Float64      = 1.0,
    Δ_abs::Int64     = 3,
    γ_plus::Float64  = 0.3,
    γ_minus::Float64 = 0.3,
    w_max::Float64   = 1.0)::MySpikingHMemNetwork

    model = modeltype()
    model.ell_key   = ell_key
    model.ell_value = ell_value
    # Empty memory: every (key, value) association starts at W = 0 and is
    # written incrementally by `write_phase!`. The matrix is shaped
    # (ell_value, ell_key) so that I_value = W_assoc * S_key has the right
    # length without any transpose at recall time.
    model.W_assoc   = zeros(Float64, ell_value, ell_key)
    model.ϑ         = ϑ
    model.τ_m       = τ_m
    model.τ_trace   = τ_trace
    model.Δt        = Δt
    model.Δ_abs     = Δ_abs
    model.γ_plus    = γ_plus
    model.γ_minus   = γ_minus
    model.w_max     = w_max
    return model
end
