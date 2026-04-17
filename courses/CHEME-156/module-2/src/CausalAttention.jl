# This file implements the masked multi-head self-attention layer used inside
# each decoder block. The comments focus on the shape reshaping because that is
# usually the part students find hardest to trace.

"""
    CausalAttention(d_model, n_heads, ctx_len)

Multi-head causal self-attention as a Flux layer. Holds four learnable weight
matrices `W_Q`, `W_K`, `W_V`, `W_O` of shape `(d_model, d_model)` (with all `H`
heads packed into the column dimension), plus a precomputed `(ctx_len, ctx_len)`
causal mask whose entries are `-1e9` strictly above the diagonal and `0` on or
below it. The mask is stored as a non-trainable buffer.

When called as `ca(X)`:

* `X` is a `(d_model, T, B)` tensor of token embeddings (T = sequence length, B = batch size)
* `T` must satisfy `T ≤ ctx_len`

Returns the attention output as a `(d_model, T, B)` tensor. Use
[`causal_attention_weights`](@ref) to recover the per-head attention probability
matrices for visualization.
"""
struct CausalAttention
    WQ::Matrix{Float32}    # query projection
    WK::Matrix{Float32}    # key projection
    WV::Matrix{Float32}    # value projection
    WO::Matrix{Float32}    # output projection after concatenating heads
    mask::Matrix{Float32}  # cached causal mask up to the maximum context length
    n_heads::Int           # number of attention heads
    d_head::Int            # hidden width handled by one head
end

Flux.@layer CausalAttention trainable=(WQ, WK, WV, WO)

function CausalAttention(d_model::Int, n_heads::Int, ctx_len::Int)
    @assert d_model % n_heads == 0 "d_model ($d_model) must be divisible by n_heads ($n_heads)"
    d_head = d_model ÷ n_heads
    s = 1.0f0 / sqrt(Float32(d_model))

    # Precompute the full causal mask once. During a forward pass we slice out
    # the top-left `T x T` block that matches the current sequence length.
    mask = zeros(Float32, ctx_len, ctx_len)
    for i in 1:ctx_len, j in 1:ctx_len
        if j > i
            mask[i, j] = -1.0f9
        end
    end

    return CausalAttention(
        randn(Float32, d_model, d_model) .* s,
        randn(Float32, d_model, d_model) .* s,
        randn(Float32, d_model, d_model) .* s,
        randn(Float32, d_model, d_model) .* s,
        mask,
        n_heads,
        d_head,
    )
end

"""
    _causal_attention_forward(ca, X) -> Tuple{Array{Float32, 3}, Array{Float32, 3}}

Internal helper that computes both the attention output and the raw attention
probabilities. Keeping the two together avoids duplicating the tensor logic in
the main forward pass and the visualization helper.
"""
function _causal_attention_forward(ca::CausalAttention, X::AbstractArray{<:Real, 3})
    d_model, T, B = size(X)
    n_heads = ca.n_heads
    d_head = ca.d_head

    # Flatten the time and batch axes so each projection can be done with one
    # matrix multiplication, then reshape back to `(d_model, T, B)`.
    Xmat = reshape(X, d_model, T * B)
    Q = reshape(ca.WQ' * Xmat, d_model, T, B)
    K = reshape(ca.WK' * Xmat, d_model, T, B)
    V = reshape(ca.WV' * Xmat, d_model, T, B)

    # Split the hidden dimension into heads. We then fold `n_heads` into the
    # batch axis so `NNlib.batched_mul` can process every head in parallel.
    Q4 = reshape(Q, d_head, n_heads, T, B)
    K4 = reshape(K, d_head, n_heads, T, B)
    V4 = reshape(V, d_head, n_heads, T, B)
    Qh = reshape(permutedims(Q4, (1, 3, 2, 4)), d_head, T, n_heads * B)
    Kh = reshape(permutedims(K4, (1, 3, 2, 4)), d_head, T, n_heads * B)
    Vh = reshape(permutedims(V4, (1, 3, 2, 4)), d_head, T, n_heads * B)

    # Every score matrix compares all query positions against all key positions
    # inside one head. Dividing by `sqrt(d_head)` keeps the logits from growing
    # too large as the head width increases.
    scores = NNlib.batched_mul(permutedims(Qh, (2, 1, 3)), Kh) ./ sqrt(Float32(d_head))

    # Mask out future positions so token `t` cannot attend to anything after `t`.
    mask_TT = ca.mask[1:T, 1:T]
    scores = scores .+ reshape(mask_TT, T, T, 1)

    # Softmax over keys turns each row of scores into attention probabilities.
    weights = softmax(scores; dims = 2)

    # Use those probabilities to form a weighted combination of the value
    # vectors at each position.
    out = NNlib.batched_mul(Vh, permutedims(weights, (2, 1, 3)))

    # Undo the earlier reshape/permutation so the heads are concatenated back
    # into a single `(d_model, T, B)` representation.
    out4 = reshape(out, d_head, T, n_heads, B)
    out_perm = permutedims(out4, (1, 3, 2, 4))
    out_full = reshape(out_perm, d_model, T, B)

    # Final linear projection mixes information across heads.
    out_mat = reshape(out_full, d_model, T * B)
    out_proj = reshape(ca.WO' * out_mat, d_model, T, B)

    return out_proj, weights
end

function (ca::CausalAttention)(X::AbstractArray{<:Real, 3})
    # The public layer call returns only the transformed hidden states. The
    # attention weights are still available through `causal_attention_weights`.
    out, _ = _causal_attention_forward(ca, X)
    return out
end

"""
    causal_attention_weights(ca::CausalAttention, X) -> Array{Float32, 4}

Run the causal attention layer on `X` and return the attention probability
matrices, reshaped to `(T, T, n_heads, B)`. Useful for visualization.
"""
function causal_attention_weights(ca::CausalAttention,
                                   X::AbstractArray{<:Real, 3})::Array{Float32, 4}
    _, weights = _causal_attention_forward(ca, X)
    T, _, _ = size(weights)
    n_heads = ca.n_heads
    B = size(X, 3)
    # Undo the "heads folded into batch" trick used internally so downstream
    # plotting code can index attention maps as `(query, key, head, example)`.
    return reshape(weights, T, T, n_heads, B)
end
