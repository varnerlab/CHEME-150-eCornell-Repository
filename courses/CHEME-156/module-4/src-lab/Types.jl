"""
    MyCustomConvolutionLayerModel{A, B, F} <: GNNLayer

A custom message-passing graph-convolution layer with a self-plus-neighbors update rule.

For each node `i` in a graph at layer `ℓ`, the forward pass computes
`x_i^(ℓ+1) = act.(W1 * x_i^(ℓ) + W2 * sum_{j in N(i)} x_j^(ℓ) + b)`,
where `W1` transforms the node's own features, `W2` transforms the sum of its
neighbors' features, `b` is a per-layer bias broadcast across nodes, and `act`
is an elementwise activation function. The `+` aggregation is the unweighted
sum of neighbor features; swap to `mean` or `max` to change the aggregator.

# Type parameters
- `A <: AbstractMatrix`: type of the two weight matrices `W1` and `W2`.
- `B <: AbstractVector`: type of the bias vector `b`.
- `F`:                   type of the activation function `act`
                         (typically `typeof(relu)` or `typeof(identity)`).

# Fields
- `W1::A`:  self-transform matrix of shape `(d_out, d_in)`.
- `W2::A`:  neighbor-transform matrix of shape `(d_out, d_in)`.
- `b::B`:   per-layer bias vector of length `d_out`, broadcast across all nodes.
- `act::F`: elementwise activation, applied after the affine combination.

The struct subtypes [`GNNLayer`](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GraphNeuralNetworks.jl/stable/GraphNeuralNetworks/api/basic/#GraphNeuralNetworks.GNNLayer)
from `GraphNeuralNetworks.jl`, which is what allows it to be used as a stage
inside a `GNNChain`. The default constructor in `src/Factory.jl` initializes
`W1` and `W2` with Glorot-uniform weights and `b` to zeros.
"""
struct MyCustomConvolutionLayerModel{A<:AbstractMatrix, B<:AbstractVector, F} <: GNNLayer
	W1::A   # self-transform: applied to the node's own feature vector
	W2::A   # neighbor-transform: applied to the sum of the node's neighbors' feature vectors
	b::B    # per-layer bias of length d_out, broadcast across all nodes within the layer
	act::F  # elementwise activation function (e.g., relu, identity)
end
