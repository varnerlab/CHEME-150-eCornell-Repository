"""
    sinusoidal_pe(d, n_max) -> Matrix{Float32}

Compute a `(d, n_max)` matrix of sinusoidal positional encodings.
Position `pos` (1-indexed) and dimension index `i` use the formula:
    PE[2i-1, pos] = sin((pos-1) / 10000^((2i-2)/d))
    PE[2i,   pos] = cos((pos-1) / 10000^((2i-2)/d))

Adding this to the token embeddings before attention gives the model
information about word order, which is otherwise lost because
self-attention is permutation-equivariant.
"""
function sinusoidal_pe(d::Int, n_max::Int)::Matrix{Float32}
    PE = zeros(Float32, d, n_max)
    for pos in 1:n_max
        for i in 0:2:(d - 1)
            denom = Float32(10000.0)^(Float32(i) / Float32(d))
            PE[i + 1, pos] = sin(Float32(pos - 1) / denom)
            if i + 2 <= d
                PE[i + 2, pos] = cos(Float32(pos - 1) / denom)
            end
        end
    end
    return PE
end

"""
    SelfAttention(d, d_k, d_v)

Single-head scaled dot-product self-attention as a Flux layer. Holds three
learnable projection matrices `W_Q`, `W_K`, `W_V` of shape `(d, d_k)`,
`(d, d_k)`, and `(d, d_v)` respectively, where `d` is the input embedding
dimension and `d_k`, `d_v` are the key and value dimensions.

When called as `sa(X, mask)`:

* `X` is a `(d, n, B)` tensor of token embeddings (n = sequence length, B = batch size)
* `mask` is a `(n, B)` `Bool` matrix where `true` marks valid (non-padding) positions

Returns `(output, weights)` where `output` is `(d_v, n, B)` and `weights` is
`(n, n, B)` with `weights[i, j, b]` giving the attention weight from query `i`
to key `j` in batch `b`.

Implements the equation
    softmax( Q Kᵀ / sqrt(d_k) ) V
from the L13a lecture, with key positions corresponding to padding masked out
to large negative scores so they receive zero attention weight.
"""
struct SelfAttention
    WQ::Matrix{Float32}  # query projection matrix (d × d_k)
    WK::Matrix{Float32}  # key projection matrix (d × d_k)
    WV::Matrix{Float32}  # value projection matrix (d × d_v)
end

# Tell Flux that SelfAttention is a trainable layer (so WQ, WK, WV are updated during training)
Flux.@layer SelfAttention

"""
    SelfAttention(d, d_k, d_v)

Constructor: initialize projection matrices with small random values scaled by 1/√d
to keep initial outputs in a reasonable range (Xavier-style initialization).
"""
function SelfAttention(d::Int, d_k::Int, d_v::Int)
    s = 1.0f0 / sqrt(Float32(d))  # scale factor for initialization
    return SelfAttention(
        randn(Float32, d, d_k) .* s,   # W_Q: projects input → queries
        randn(Float32, d, d_k) .* s,   # W_K: projects input → keys
        randn(Float32, d, d_v) .* s,   # W_V: projects input → values
    )
end

"""
    (sa::SelfAttention)(X, mask)

Forward pass: compute scaled dot-product self-attention.

Steps:
1. Project X into Q, K, V using the learned weight matrices.
2. Compute attention scores as Q Kᵀ / √d_k.
3. Mask out padding positions by adding a large negative value to their scores.
4. Apply row-wise softmax so each query's weights sum to 1.
5. Compute the output as the weighted sum of value vectors.

Returns `(output, weights)` where:
- `output` is (d_v, n, B): the context-aware representation of each token
- `weights` is (n, n, B): the attention weight matrix (useful for visualization)
"""
function (sa::SelfAttention)(X::AbstractArray{<:Real, 3},
                              mask::AbstractMatrix{Bool})
    d, n, B = size(X)
    d_k = size(sa.WQ, 2)

    # Step 1: Project input X to queries (Q), keys (K), and values (V).
    # We flatten (d, n, B) → (d, n*B) for a single matrix multiply, then reshape back.
    Xmat = reshape(X, d, n * B)                       # (d, n*B)
    Q = reshape(sa.WQ' * Xmat, d_k, n, B)             # (d_k, n, B) — each column is a query vector
    K = reshape(sa.WK' * Xmat, d_k, n, B)             # (d_k, n, B) — each column is a key vector
    V = reshape(sa.WV' * Xmat, :, n, B)               # (d_v, n, B) — each column is a value vector

    # Step 2: Compute raw attention scores.
    # scores[i, j, b] = <Q[:, i, b], K[:, j, b]> / √d_k
    # This measures how much query i should attend to key j.
    scores = NNlib.batched_mul(permutedims(Q, (2, 1, 3)), K) ./ sqrt(Float32(d_k))  # (n, n, B)

    # Step 3: Mask out padding positions.
    # Where mask is false (padding), subtract a large constant (1e9) so softmax assigns ~0 weight.
    mask_f = reshape(Float32.(mask), 1, n, B)          # (1, n, B) — broadcasts over the query dimension
    scores = scores .+ (mask_f .- 1.0f0) .* 1.0f9     # padding positions get score ≈ -1e9

    # Step 4: Apply softmax along the key dimension (dim=2).
    # Each row of the resulting matrix is a probability distribution over keys.
    weights = softmax(scores; dims = 2)                # (n, n, B) — each row sums to 1

    # Step 5: Compute weighted sum of value vectors.
    # output[:, i, b] = Σⱼ weights[i, j, b] * V[:, j, b]
    output = NNlib.batched_mul(V, permutedims(weights, (2, 1, 3)))   # (d_v, n, B)

    return output, weights
end

"""
    MultiHeadAttention(d, d_k, d_v, n_heads, d_out)

Multi-head self-attention layer. Runs `n_heads` independent `SelfAttention`
heads in parallel, concatenates their outputs along the embedding dimension,
and projects the result through a learned output matrix `W_O` of shape
`(n_heads * d_v, d_out)`.

Each head learns a different attention pattern, allowing the model to attend
to different aspects of the input simultaneously (e.g., one head may attend
to subject words, another to incongruity signals).

Returns `(output, avg_weights)` where `output` is `(d_out, n, B)` and
`avg_weights` is the mean of all heads' attention weight matrices `(n, n, B)`.
"""
struct MultiHeadAttention
    heads::Vector{SelfAttention}   # H independent attention heads
    WO::Matrix{Float32}           # output projection (n_heads * d_v, d_out)
end

Flux.@layer MultiHeadAttention

function MultiHeadAttention(d::Int, d_k::Int, d_v::Int, n_heads::Int, d_out::Int)
    heads = [SelfAttention(d, d_k, d_v) for _ in 1:n_heads]
    s = 1.0f0 / sqrt(Float32(n_heads * d_v))
    WO = randn(Float32, n_heads * d_v, d_out) .* s
    return MultiHeadAttention(heads, WO)
end

function (mha::MultiHeadAttention)(X::AbstractArray{<:Real, 3},
                                    mask::AbstractMatrix{Bool})

    # run each head independently and collect outputs + weights
    head_results = [h(X, mask) for h in mha.heads]

    # concatenate head outputs along the embedding dimension: (n_heads*d_v, n, B)
    concat = cat([out for (out, _) in head_results]...; dims=1)

    # project concatenated output through W_O: (d_out, n, B)
    d_cat, n, B = size(concat)
    output = reshape(mha.WO' * reshape(concat, d_cat, n * B), :, n, B)

    # average attention weights across heads for visualization
    avg_weights = mean([w for (_, w) in head_results])

    return output, avg_weights
end

"""
    masked_mean_pool(out, mask) -> Matrix{Float32}

Mean-pool a `(d, n, B)` tensor along its sequence dimension (dim=2), averaging only
over positions where `mask` is true. Returns a `(d, B)` matrix with one pooled
vector per batch element.

This collapses the sequence into a single fixed-size vector per headline,
which can then be passed to the MLP classification head.
"""
function masked_mean_pool(out::AbstractArray{<:Real, 3},
                           mask::AbstractMatrix{Bool})
    _, n, B = size(out)
    mask_f = reshape(Float32.(mask), 1, n, B)          # (1, n, B) — broadcast-compatible with out
    counts = sum(mask_f; dims = 2)                     # (1, 1, B) — number of valid tokens per headline
    pooled = sum(out .* mask_f; dims = 2) ./ max.(counts, 1.0f0)  # avoid division by zero
    return dropdims(pooled; dims = 2)                  # (d, B) — one vector per headline
end

"""
    AttentionClassifier(d_emb, d_k, d_v, n_heads, d_hidden, n_classes, n_max)

A classifier built from four components:
1. Sinusoidal positional encoding added to the input token embeddings.
2. Multi-head self-attention that produces context-aware token representations.
3. Masked mean-pooling that averages those representations over valid positions.
4. A two-layer MLP head (Dense → relu → Dense) that maps the pooled vector to class logits.

Takes padded token sequences `(X, mask)` and returns class logits of shape `(n_classes, B)`.
"""
struct AttentionClassifier{A}
    PE::Matrix{Float32}    # sinusoidal positional encoding (d_emb, n_max) — not trained
    attention::A           # multi-head or single-head self-attention layer
    head::Chain            # MLP classification head (pooled vector → class logits)
end

# Tell Flux that AttentionClassifier is a trainable layer
Flux.@layer AttentionClassifier trainable=(attention, head)

"""Single-head constructor (backward-compatible with L13a: no PE, no dropout)."""
function AttentionClassifier(d_emb::Int, d_k::Int, d_v::Int,
                              d_hidden::Int, n_classes::Int;
                              n_max::Int=32)
    return AttentionClassifier(
        zeros(Float32, d_emb, n_max),                                            # no-op PE (zeros): preserves L13a behavior
        SelfAttention(d_emb, d_k, d_v),                                          # single-head attention
        Chain(Dense(d_v => d_hidden, relu), Dense(d_hidden => n_classes)),       # MLP head (no dropout, matches L13a checkpoint)
    )
end

"""Multi-head constructor."""
function AttentionClassifier(d_emb::Int, d_k::Int, d_v::Int, n_heads::Int,
                              d_hidden::Int, n_classes::Int;
                              n_max::Int=32, dropout::Float64=0.0)
    d_out = d_v  # output dimension of multi-head attention matches d_v for the MLP head
    return AttentionClassifier(
        sinusoidal_pe(d_emb, n_max),                                             # positional encoding
        MultiHeadAttention(d_emb, d_k, d_v, n_heads, d_out),                    # multi-head attention
        Chain(Dense(d_out => d_hidden, relu), Dropout(dropout),
              Dense(d_hidden => n_classes)),                                      # MLP head with dropout
    )
end

"""
    (m::AttentionClassifier)(X, mask) -> Matrix{Float32}

Forward pass: add positional encoding to the token embeddings, run self-attention
over the sequence, pool over valid positions, and pass through the MLP head
to produce class logits.
"""
function (m::AttentionClassifier)(X::AbstractArray{<:Real, 3},
                                   mask::AbstractMatrix{Bool})
    _, n, _ = size(X)
    X_pe = X .+ m.PE[:, 1:n, :]          # add positional encoding (broadcasts over batch)
    out, _ = m.attention(X_pe, mask)      # (d_v, n, B) — context-aware token representations
    pooled = masked_mean_pool(out, mask)  # (d_v, B) — one vector per headline
    return m.head(pooled)                 # (n_classes, B) — class logits
end

"""
    attention_weights(m::AttentionClassifier, X, mask) -> Array{Float32, 3}

Forward `m` over `(X, mask)` and return only the attention weights `(n, n, B)`,
discarding the classification logits. For multi-head attention, returns the
average attention weights across all heads.

This is used for post-hoc visualization of what the model attends to.
"""
function attention_weights(m::AttentionClassifier,
                            X::AbstractArray{<:Real, 3},
                            mask::AbstractMatrix{Bool})::Array{Float32, 3}
    _, n, _ = size(X)
    X_pe = X .+ m.PE[:, 1:n, :]
    _, w = m.attention(X_pe, mask)
    return w
end

"""
    per_head_attention_weights(m::AttentionClassifier, X, mask) -> Vector{Array{Float32, 3}}

Forward `m` over `(X, mask)` and return a vector of per-head attention weight
matrices. Each element is an `(n, n, B)` array from a single head.

This is used to inspect what individual heads have learned to attend to.
For single-head models, returns a one-element vector.
"""
function per_head_attention_weights(m::AttentionClassifier,
                                    X::AbstractArray{<:Real, 3},
                                    mask::AbstractMatrix{Bool})::Vector{Array{Float32, 3}}
    _, n, _ = size(X)
    X_pe = X .+ m.PE[:, 1:n, :]

    if m.attention isa MultiHeadAttention
        return [let (_, w) = h(X_pe, mask); Array(w) end for h in m.attention.heads]
    else
        _, w = m.attention(X_pe, mask)
        return [Array(w)]
    end
end
