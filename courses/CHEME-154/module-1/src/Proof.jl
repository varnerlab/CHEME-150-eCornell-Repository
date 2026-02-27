"""
    hebbian_weights(patterns::Matrix{Int}) -> Matrix{Float64}

Compute the classical Hebbian weight matrix from a collection of binary
memories.

The input `patterns` is a `K x N` matrix whose rows are memories with
entries in `{-1, +1}`. The returned matrix is:

```math
W = \\frac{1}{N} P^\\top P,
```

with the diagonal forced to zero (`w_ii = 0`) to remove self-coupling.
"""
function hebbian_weights(patterns::Matrix{Int})
    _, N = size(patterns)
    W = (patterns' * patterns) ./ N
    W[diagind(W)] .= 0.0
    return Matrix{Float64}(W)
end

"""
    energy(state::Vector{Int}, W::Matrix{Float64}, bias::Vector{Float64}=zeros(Float64, length(state))) -> Float64

Evaluate the classical Hopfield energy for a binary state vector.

The energy is:

```math
E(\\mathbf{s}) = -\\frac{1}{2}\\mathbf{s}^\\top W\\mathbf{s} - \\mathbf{b}^\\top\\mathbf{s}.
```

Lower values correspond to more stable states under asynchronous updates.
"""
function energy(
    state::Vector{Int},
    W::Matrix{Float64},
    bias::Vector{Float64}=zeros(Float64, length(state)),
)
    return -0.5 * dot(state, W * state) - dot(bias, state)
end

"""
    corrupt_pattern(pattern::Vector{Int}, flip_fraction::Float64, rng::AbstractRNG) -> Vector{Int}

Create a noisy cue by flipping a fraction of bits in a binary pattern.

The number of flipped entries is `round(Int, flip_fraction * N)` where
`N = length(pattern)`. Each selected bit is multiplied by `-1`.
"""
function corrupt_pattern(pattern::Vector{Int}, flip_fraction::Float64, rng::AbstractRNG)
    s = copy(pattern)
    n_flip = round(Int, flip_fraction * length(s))
    if n_flip > 0
        # Sample unique indices and flip their signs to inject corruption.
        idx = randperm(rng, length(s))[1:n_flip]
        s[idx] .*= -1
    end
    return s
end

"""
    hamming_distance(a::Vector{Int}, b::Vector{Int}) -> Int

Count the number of positions where two binary state vectors differ.
"""
hamming_distance(a::Vector{Int}, b::Vector{Int}) = count(a .!= b)

"""
    async_retrieve(initial_state::Vector{Int}, W::Matrix{Float64};
                   bias::Vector{Float64}=zeros(Float64, length(initial_state)),
                   max_sweeps::Int=80,
                   rng::AbstractRNG=Random.default_rng()) -> Tuple{Vector{Int}, Vector{Float64}}

Run asynchronous Hopfield retrieval from `initial_state` and record the
energy after each accepted flip.

One sweep updates each neuron once in random order. For neuron `i`, the
local field is `h_i = dot(W[i, :], s) + bias[i]`. The proposal is:

- `+1` if `h_i > 0`
- `-1` if `h_i < 0`
- unchanged if `h_i == 0` (tie-preserving update)

Returns `(s_final, energies)` where `energies[1]` is the initial energy.
"""
function async_retrieve(
    initial_state::Vector{Int},
    W::Matrix{Float64};
    bias::Vector{Float64}=zeros(Float64, length(initial_state)),
    max_sweeps::Int=80,
    rng::AbstractRNG=Random.default_rng(),
)
    s = copy(initial_state)
    N = length(s)
    energies = Float64[energy(s, W, bias)]

    for _ in 1:max_sweeps
        flipped = false

        # Asynchronous update: visit each neuron once in random order.
        for i in randperm(rng, N)
            h_i = dot(@view(W[i, :]), s) + bias[i]
            proposed = h_i == 0 ? s[i] : (h_i > 0 ? 1 : -1)

            # Record energy only when a flip is accepted.
            if proposed != s[i]
                s[i] = proposed
                flipped = true
                push!(energies, energy(s, W, bias))
            end
        end

        # Stop early when an entire sweep produces no state changes.
        !flipped && break
    end

    return s, energies
end

"""
    recall_success_rate(N::Int, alpha_values::Vector{Float64};
                        trials::Int=60,
                        flip_fraction::Float64=0.10,
                        max_sweeps::Int=80,
                        seed::Int=5821) -> Vector{Tuple{Float64, Int, Float64}}

Estimate exact-recall success rate versus load `alpha = K / N`.

For each `alpha`, this function:
1. Sets `K = round(Int, alpha * N)` with a minimum of 1.
2. Samples `K` random binary memories.
3. Encodes a Hopfield matrix with `hebbian_weights`.
4. Corrupts one sampled memory and runs asynchronous retrieval.
5. Counts success when the final state exactly matches the target memory.

Returns rows of `(alpha, K, success_rate)`.
"""
function recall_success_rate(
    N::Int,
    alpha_values::Vector{Float64};
    trials::Int=60,
    flip_fraction::Float64=0.10,
    max_sweeps::Int=80,
    seed::Int=5821,
)
    rng = MersenneTwister(seed)
    rows = Tuple{Float64, Int, Float64}[]

    for alpha in alpha_values
        K = max(1, round(Int, alpha * N))
        success = 0

        for _ in 1:trials
            patterns = rand(rng, (-1, 1), K, N)
            W = hebbian_weights(patterns)

            idx = rand(rng, 1:K)
            target = vec(patterns[idx, :])
            noisy = corrupt_pattern(target, flip_fraction, rng)
            retrieved, _ = async_retrieve(noisy, W; max_sweeps=max_sweeps, rng=rng)

            retrieved == target && (success += 1)
        end

        push!(rows, (alpha, K, success / trials))
    end

    return rows
end
