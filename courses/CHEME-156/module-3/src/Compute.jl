"""
    random_sparse_pair(m::Int; sparsity::Float64 = 0.1,
        rng::AbstractRNG = Random.GLOBAL_RNG) -> Tuple{Vector{Float64}, Vector{Float64}}

Draw a random sparse non-negative key-value pair `(k, v)` in `R^m`. Each entry
is drawn from `Uniform(0, 1)` with probability `sparsity` and is zero otherwise.
This mimics the regime that the ReLU encoders in H-Mem produce: sparse,
non-negative key and value activations.

### Arguments
- `m::Int`: dimension of the key and value vectors.

### Keyword arguments
- `sparsity::Float64`: probability that any individual entry of `k` or `v` is
   non-zero. Defaults to `0.1`.
- `rng::AbstractRNG`: random number generator.

### Returns
- `(k::Vector{Float64}, v::Vector{Float64})`, both of length `m`.
"""
function random_sparse_pair(m::Int; sparsity::Float64 = 0.1,
    rng::AbstractRNG = Random.GLOBAL_RNG)::Tuple{Vector{Float64}, Vector{Float64}}

    @assert 0.0 < sparsity <= 1.0 "sparsity must lie in (0, 1]"
    k = [rand(rng) < sparsity ? rand(rng) : 0.0 for _ in 1:m]
    v = [rand(rng) < sparsity ? rand(rng) : 0.0 for _ in 1:m]
    return k, v
end

"""
    hebbian_write!(model::MyHMemMemory, k::Vector{Float64}, v::Vector{Float64})
        -> MyHMemMemory

Apply one step of the Hebbian write rule from
[Limbacher and Legenstein (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/file/f6876a9f998f6472cc26708e27444456-Paper.pdf),
Eq. 1, to the in-place association matrix `model.W`:

    ΔW[k_idx, j] = γ_plus * (w_max - W[k_idx, j]) * v[k_idx] * k[j]
                 - γ_minus * W[k_idx, j] * k[j]^2

The first term strengthens the synapse between co-active pre-synaptic
(key-side) coordinate `j` and post-synaptic (value-side) coordinate `k_idx`.
The second term, in the spirit of [Oja's rule](https://link.springer.com/article/10.1007/BF00275687),
weakens connections from currently active key-coordinates to all targets, so
that fresh associations overwrite stale ones.

### Arguments
- `model`: `MyHMemMemory` whose association matrix `W` is updated in place.
- `k::Vector{Float64}`: key vector of length `m` (pre-synaptic activation).
- `v::Vector{Float64}`: value vector of length `m` (post-synaptic activation).

### Returns
- `model`, with `W` updated in place.
"""
function hebbian_write!(model::MyHMemMemory,
    k::Vector{Float64}, v::Vector{Float64})::MyHMemMemory

    @assert length(k) == model.m "key length $(length(k)) does not match m = $(model.m)"
    @assert length(v) == model.m "value length $(length(v)) does not match m = $(model.m)"

    γ_plus  = model.γ_plus
    γ_minus = model.γ_minus
    w_max   = model.w_max
    W       = model.W

    for j in 1:model.m, k_idx in 1:model.m
        plus_term  = γ_plus  * (w_max - W[k_idx, j]) * v[k_idx] * k[j]
        minus_term = γ_minus * W[k_idx, j] * k[j]^2
        W[k_idx, j] += plus_term - minus_term
    end
    return model
end

"""
    hebbian_read(model::MyHMemMemory, k_query::Vector{Float64}) -> Vector{Float64}

Read out the value vector associated with `k_query` by applying the matrix-vector
product `v_recall = W * k_query`. This is the simple feedforward read rule from
[Limbacher and Legenstein (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/file/f6876a9f998f6472cc26708e27444456-Paper.pdf),
Eq. 5.

### Arguments
- `model`: `MyHMemMemory` whose association matrix `W` is queried.
- `k_query::Vector{Float64}`: query key of length `m`.

### Returns
- `v_recall::Vector{Float64}` of length `m`.
"""
function hebbian_read(model::MyHMemMemory, k_query::Vector{Float64})::Vector{Float64}
    @assert length(k_query) == model.m "query length $(length(k_query)) does not match m = $(model.m)"
    return model.W * k_query
end

"""
    corrupt_key(k::Vector{Float64}, p::Float64;
        rng::AbstractRNG = Random.GLOBAL_RNG) -> Vector{Float64}

Return a corrupted copy of the key `k` in which a fraction `p` of the entries
are replaced by an independent `Uniform(0, 1)` sample. The remaining `(1 - p)`
fraction of entries are passed through unchanged. `p = 0` returns a perfect
copy, `p = 1` returns an entirely fresh random key.

### Arguments
- `k::Vector{Float64}`: clean key vector.
- `p::Float64`: corruption probability per entry, in `[0, 1]`.

### Keyword arguments
- `rng::AbstractRNG`: random number generator.

### Returns
- `k_noisy::Vector{Float64}` of the same length as `k`.
"""
function corrupt_key(k::Vector{Float64}, p::Float64;
    rng::AbstractRNG = Random.GLOBAL_RNG)::Vector{Float64}

    @assert 0.0 <= p <= 1.0 "corruption probability p must lie in [0, 1]"
    return [rand(rng) < p ? rand(rng) : k[i] for i in eachindex(k)]
end

"""
    H(x::Real) -> Int

Heaviside step function. Returns 1 if `x >= 0`, otherwise 0.
"""
H(x::Real)::Int = x >= 0 ? 1 : 0

"""
    poisson_encode(rates::AbstractVector{<:Real}, T::Int;
        rng::AbstractRNG = Random.GLOBAL_RNG) -> Matrix{Int}

Generate a binary spike-train matrix of shape `(d, T)` from a per-neuron rate
vector of length `d`. At every time step `t`, draw `u[j, t] ~ Bernoulli(rates[j])`
independently across input neurons. Each `rates[j]` must lie in `[0, 1]`.

### Returns
- `Matrix{Int}` of shape `(d, T)` with entries in `{0, 1}`.
"""
function poisson_encode(rates::AbstractVector{<:Real}, T::Int;
    rng::AbstractRNG = Random.GLOBAL_RNG)::Matrix{Int}

    @assert all(0 .<= rates .<= 1) "rates must lie in [0, 1]"
    d = length(rates)
    U = zeros(Int, d, T)
    # Bernoulli per (neuron, time-step) — independent draws, so different
    # neurons with the same rate produce decorrelated spike trains, which
    # is what the Hebbian rule needs to tell stored keys apart.
    for t in 1:T, j in 1:d
        U[j, t] = rand(rng) < rates[j] ? 1 : 0
    end
    return U
end

"""
    simulate_lif_layer(I::Matrix{Float64};
        ϑ::Float64, τ_m::Float64, Δt::Float64, Δ_abs::Int)
        -> Tuple{Matrix{Float64}, Matrix{Int}}

Simulate a layer of LIF neurons given a precomputed input-current matrix
`I` of shape `(N, T)`. The LIF recursion is

    V[j, t+1] = α * V[j, t] + (1 - α) * I[j, t] - ϑ * z[j, t]
    z[j, t+1] = (V[j, t+1] >= ϑ) and (refractory countdown is zero)

with decay factor `α = exp(-Δt / τ_m)`. After a spike, neuron `j` is held
silent for `Δ_abs` time steps. This matches Eq. 2 of Limbacher et al. (2022).

### Returns
- `V::Matrix{Float64}` of shape `(N, T)`: membrane potential trajectory.
- `S::Matrix{Int}`     of shape `(N, T)`: binary spike train.
"""
function simulate_lif_layer(I::Matrix{Float64};
    ϑ::Float64, τ_m::Float64, Δt::Float64, Δ_abs::Int
    )::Tuple{Matrix{Float64}, Matrix{Int}}

    N, T = size(I)
    α    = exp(-Δt / τ_m)

    # Pad an extra column at index 1 to hold the "t = 0" initial condition
    # (V = 0, no spike), so the recursion can read V[j, t] / S[j, t] without
    # a special-case branch on the first iteration. We strip that column on
    # return so the output is aligned to t = 1..T.
    V = zeros(Float64, N, T + 1)
    S = zeros(Int,     N, T + 1)
    r = zeros(Int, N)  # per-neuron refractory countdown

    for t in 1:T, j in 1:N
        # Soft reset: subtracting ϑ * S (instead of clamping V to 0) keeps
        # any "overshoot" above threshold, which is the formulation used in
        # Limbacher et al. (2022) and avoids artificially flattening high-
        # input neurons.
        V[j, t + 1] = α * V[j, t] + (1 - α) * I[j, t] - ϑ * S[j, t]
        if r[j] == 0 && H(V[j, t + 1] - ϑ) == 1
            # Fire and arm the refractory counter; spikes during the next
            # Δ_abs steps are forbidden no matter how high V climbs.
            S[j, t + 1] = 1
            r[j] = Δ_abs
        else
            S[j, t + 1] = 0
            r[j] = max(r[j] - 1, 0)
        end
    end
    return V[:, 2:end], S[:, 2:end]
end

"""
    spike_trace(S::AbstractMatrix{<:Integer};
        τ_trace::Float64, Δt::Float64) -> Matrix{Float64}

Compute the synaptic trace `κ` of a binary spike matrix `S` of shape `(N, T)`.
The trace obeys the low-pass recursion

    κ[j, t+1] = α_trace * κ[j, t] + (1 - α_trace) * S[j, t]

with `α_trace = exp(-Δt / τ_trace)`. This matches the trace definition in
Sec. 4.2.2 of Limbacher et al. (2022): `κ` increments by `1 - exp(-Δt/τ_trace)`
on a spike and decays exponentially with time constant `τ_trace` between spikes.

### Returns
- `κ::Matrix{Float64}` of the same shape as `S`.
"""
function spike_trace(S::AbstractMatrix{<:Integer};
    τ_trace::Float64, Δt::Float64)::Matrix{Float64}

    N, T = size(S)
    α    = exp(-Δt / τ_trace)
    κ = zeros(Float64, N, T)
    # Causal filter: κ at time t depends on the spike at time t - 1, so a
    # spike contributes to plasticity only on the *next* step (matches the
    # one-step pre/post delay in the Hebbian rule of Limbacher et al. 2022).
    # Loop starts at t = 2 because κ[:, 1] = 0 by definition.
    for t in 2:T, j in 1:N
        κ[j, t] = α * κ[j, t - 1] + (1 - α) * S[j, t - 1]
    end
    return κ
end

"""
    write_phase!(model::MySpikingHMemNetwork,
        U_key::Matrix{Int}, U_value::Matrix{Int};
        I_scale::Float64 = 1.0) -> NamedTuple

Run one (key, value) write episode through the spiking H-Mem network.

The key-layer and value-layer are LIF neurons driven by the binary input
spike matrices `U_key` (shape `(ell_key, T)`) and `U_value` (shape
`(ell_value, T)`) scaled by `I_scale`. At every time step `t`, the
association matrix `W_assoc` is updated by the spike-trace Hebbian rule
(Eq. 6 of Limbacher et al. 2022):

    ΔW[k, j] = γ_plus * (w_max - W[k, j]) * κ_value[k, t] * κ_key[j, t]
             - γ_minus * W[k, j] * κ_key[j, t]^2

`model.W_assoc` is mutated in place.

### Returns
A NamedTuple with fields:
- `V_key, S_key, κ_key`: trajectories of the key-layer membrane potential,
  spike train, and synaptic trace.
- `V_value, S_value, κ_value`: same for the value-layer.
"""
function write_phase!(model::MySpikingHMemNetwork,
    U_key::Matrix{Int}, U_value::Matrix{Int};
    I_scale::Float64 = 1.0)

    @assert size(U_key, 1) == model.ell_key
    @assert size(U_value, 1) == model.ell_value
    @assert size(U_key, 2) == size(U_value, 2)
    T = size(U_key, 2)

    # During the *write* phase both layers are clamped by their own external
    # input, so the value-layer is forced to fire the target pattern. The
    # association matrix is *not* in the loop here — recall is what reads it.
    I_key   = I_scale .* Float64.(U_key)
    I_value = I_scale .* Float64.(U_value)

    V_key, S_key   = simulate_lif_layer(I_key;
        ϑ = model.ϑ, τ_m = model.τ_m, Δt = model.Δt, Δ_abs = model.Δ_abs)
    V_value, S_value = simulate_lif_layer(I_value;
        ϑ = model.ϑ, τ_m = model.τ_m, Δt = model.Δt, Δ_abs = model.Δ_abs)

    κ_key   = spike_trace(S_key;   τ_trace = model.τ_trace, Δt = model.Δt)
    κ_value = spike_trace(S_value; τ_trace = model.τ_trace, Δt = model.Δt)

    γ_plus  = model.γ_plus
    γ_minus = model.γ_minus
    w_max   = model.w_max
    W       = model.W_assoc
    # Online STDP-style update — applied at *every* time step, not as one
    # batched outer product, because that's what neuromorphic hardware
    # actually does and because the (w_max - W) saturation term is non-linear
    # in W so the time-step-by-time-step path differs from a closed form.
    for t in 1:T, j in 1:model.ell_key, k in 1:model.ell_value
        # plus_term: soft-saturating Hebbian growth — the (w_max - W) factor
        # halts at the cap, so a heavily-reinforced synapse stops growing
        # instead of running away.
        plus_term  = γ_plus  * (w_max - W[k, j]) * κ_value[k, t] * κ_key[j, t]
        # minus_term: forgetting gated by *only* the key-side trace squared.
        # Crucially κ_value is absent — that's what lets a fresh write
        # overwrite a stale association at the same key: when the key fires
        # again with a *different* value, the active key-coordinate's old
        # outgoing weights decay regardless of whether the old value happens
        # to also be active.
        minus_term = γ_minus * W[k, j] * κ_key[j, t]^2
        W[k, j] += plus_term - minus_term
    end

    return (V_key = V_key, S_key = S_key, κ_key = κ_key,
            V_value = V_value, S_value = S_value, κ_value = κ_value)
end

"""
    recall_phase(model::MySpikingHMemNetwork, U_query_key::Matrix{Int};
        I_scale::Float64 = 1.0, c_assoc::Float64 = 1.0) -> NamedTuple

Run one recall episode. Only the key-layer receives external input
(`U_query_key`, shape `(ell_key, T)` scaled by `I_scale`); the value-layer
receives input *only* through the association matrix as
`I_value[k, t] = c_assoc * sum_j W_assoc[k, j] * S_key[j, t]`. This is the
recall current of Eq. 8 of Limbacher et al. (2022). The default `c_assoc = 1.0`
is bumped up from the paper's `c = 0.2` (Eq. 5) so that the current drives
the value-layer above threshold at the smaller population sizes used in this
lab; keep it close to `0.2` if you scale the lab up to populations of
~100 neurons to match the paper.

### Returns
A NamedTuple with fields:
- `V_key, S_key, κ_key`: key-layer trajectories.
- `V_value, S_value, κ_value`: value-layer trajectories.
"""
function recall_phase(model::MySpikingHMemNetwork, U_query_key::Matrix{Int};
    I_scale::Float64 = 1.0, c_assoc::Float64 = 0.2)

    @assert size(U_query_key, 1) == model.ell_key
    T = size(U_query_key, 2)

    I_key = I_scale .* Float64.(U_query_key)
    V_key, S_key = simulate_lif_layer(I_key;
        ϑ = model.ϑ, τ_m = model.τ_m, Δt = model.Δt, Δ_abs = model.Δ_abs)

    # Value-layer is driven through W_assoc by the *raw* key spikes S_key,
    # not the trace κ_key — the readout of an associative memory is the
    # weighted sum of currently-firing pre-synaptic neurons, exactly how a
    # neuromorphic chip would route a key activation to a stored value.
    # Note: W_assoc is read here but never updated, so recall is read-only.
    I_value = c_assoc .* (model.W_assoc * Float64.(S_key))
    V_value, S_value = simulate_lif_layer(I_value;
        ϑ = model.ϑ, τ_m = model.τ_m, Δt = model.Δt, Δ_abs = model.Δ_abs)

    # Traces are returned only for diagnostics / plotting — recall does not
    # consume them.
    κ_key   = spike_trace(S_key;   τ_trace = model.τ_trace, Δt = model.Δt)
    κ_value = spike_trace(S_value; τ_trace = model.τ_trace, Δt = model.Δt)

    return (V_key = V_key, S_key = S_key, κ_key = κ_key,
            V_value = V_value, S_value = S_value, κ_value = κ_value)
end

"""
    image_to_rate_input(image::Matrix{Float64}, T::Int;
        max_rate::Float64 = 0.5,
        rng::AbstractRNG = Random.GLOBAL_RNG) -> Matrix{Int}

Rate-code a grayscale image (entries in `[0, 1]`) into a Poisson spike-train
matrix of shape `(prod(size(image)), T)`, using one input neuron per pixel.
Pixel `p` fires at each time step with probability `max_rate * image[p]`.

### Returns
- `Matrix{Int}` of shape `(numel(image), T)` with entries in `{0, 1}`.
"""
function image_to_rate_input(image::Matrix{Float64}, T::Int;
    max_rate::Float64 = 0.5,
    rng::AbstractRNG = Random.GLOBAL_RNG)::Matrix{Int}

    @assert 0 <= max_rate <= 1
    # Scale by max_rate so even a fully-on pixel never saturates at p = 1
    # (which would produce a deterministic spike on every step and lose
    # the Poisson-noise structure the network is calibrated for). The
    # clamp guards against tiny floating-point overshoots in the loaded
    # grayscale values — without it a pixel at 1.0000001 would trip the
    # `0 <= rate <= 1` assertion in poisson_encode.
    rates = clamp.(max_rate .* vec(image), 0.0, 1.0)
    return poisson_encode(rates, T; rng = rng)
end

"""
    decode_value_spikes(S_value::AbstractMatrix{<:Integer},
        rows::Int, cols::Int) -> Matrix{Float64}

Decode a value-layer spike matrix `S_value` of shape `(rows*cols, T)` back
into a `(rows, cols)` image by averaging spikes per neuron over time and
reshaping. The result is in `[0, 1]` (per-neuron firing fraction).
"""
function decode_value_spikes(S_value::AbstractMatrix{<:Integer},
    rows::Int, cols::Int)::Matrix{Float64}

    @assert size(S_value, 1) == rows * cols
    # Inverse of the rate-coding step: per-neuron mean firing fraction over
    # the recall window estimates the firing probability that *would have*
    # produced this spike train, which under the encoder is exactly
    # max_rate * pixel. The output sits in [0, max_rate], so the lab
    # rescales it to [0, 1] before display.
    rates = vec(mean(S_value; dims = 2))
    return reshape(rates, rows, cols)
end
