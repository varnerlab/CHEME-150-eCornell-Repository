# This file contains one decoder block: the repeated unit that makes a
# transformer deep. Students can read it as "attention update, then MLP update,"
# with a residual connection around each stage.

"""
    DecoderBlock(d_model, n_heads, ctx_len; d_ff=4*d_model)

A single decoder-only transformer block: pre-LayerNorm, causal multi-head
self-attention, residual; then pre-LayerNorm, position-wise feedforward,
residual. The block accepts a `(d_model, T, B)` tensor and returns one of the
same shape.

Architecture
============

```
Y = X + CausalAttention(LayerNorm(X))
Z = Y + FFN(LayerNorm(Y))
```

The position-wise FFN is a two-layer MLP `Dense(d_model => d_ff, gelu)
→ Dense(d_ff => d_model)` applied independently to each token position. Both
LayerNorms operate on the `d_model` dimension.
"""
struct DecoderBlock
    ln1::LayerNorm         # pre-normalization before self-attention
    attn::CausalAttention  # masked self-attention sublayer
    ln2::LayerNorm         # pre-normalization before the feedforward network
    ffn::Chain             # position-wise MLP applied to every token state
end

Flux.@layer DecoderBlock

function DecoderBlock(d_model::Int, n_heads::Int, ctx_len::Int; d_ff::Int = 4 * d_model)
    return DecoderBlock(
        LayerNorm(d_model),
        CausalAttention(d_model, n_heads, ctx_len),
        LayerNorm(d_model),
        # The same two-layer MLP is reused independently at every time step.
        Chain(Dense(d_model => d_ff, gelu), Dense(d_ff => d_model)),
    )
end

function (b::DecoderBlock)(X::AbstractArray{<:Real, 3})
    # Pre-norm transformer block:
    # 1. normalize, attend, and add the residual shortcut
    Y = X .+ b.attn(b.ln1(X))
    # 2. normalize again, apply the MLP, and add the second shortcut
    Z = Y .+ b.ffn(b.ln2(Y))
    return Z
end
