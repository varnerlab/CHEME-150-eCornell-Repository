"""
    build(modeltype::Type{MyGCNLayer};
        d_in::Int64, d_out::Int64,
        activation::Function = relu,
        rng::AbstractRNG = Random.GLOBAL_RNG) -> MyGCNLayer

Construct a `MyGCNLayer` with weight matrix `W` of shape `(d_in, d_out)`
initialized to Glorot uniform: each entry is drawn from
`Uniform(-bound, bound)` with `bound = sqrt(6 / (d_in + d_out))`.

### Keyword arguments
- `d_in`: input feature dimension.
- `d_out`: output feature dimension.
- `activation`: elementwise activation, applied componentwise. Defaults to `relu`.
- `rng`: random number generator used to draw `W`.

### Returns
- a populated `MyGCNLayer` instance with `W` initialized.
"""
function build(modeltype::Type{MyGCNLayer};
    d_in::Int64,
    d_out::Int64,
    activation::Function = relu,
    rng::AbstractRNG = Random.GLOBAL_RNG)::MyGCNLayer

    bound = sqrt(6.0 / (d_in + d_out))
    W = bound .* (2.0 .* rand(rng, d_in, d_out) .- 1.0)

    layer = modeltype()
    layer.d_in = d_in
    layer.d_out = d_out
    layer.W = W
    layer.activation = activation
    return layer
end

"""
    MyCustomConvolutionLayerModel(dims::Pair{Int, Int}, act = identity)

Construct a [`MyCustomConvolutionLayerModel`](src/Types.jl) layer with weight matrices
initialized by Glorot-uniform sampling and the bias vector initialized to zeros.

# Arguments
- `dims::Pair{Int, Int}`: input/output feature dimensions, written `nin => nout`.
- `act`:                  elementwise activation function (default `identity`).

# Returns
- A `MyCustomConvolutionLayerModel` whose
  `W1, W2 ∈ R^{nout × nin}` are independently Glorot-uniform-initialized and whose
  bias `b ∈ R^{nout}` is zero. This `dims => activation` signature is the standard
  Flux convention for layer constructors and is what makes the type composable
  inside a [`GNNChain`](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GraphNeuralNetworks.jl/stable/GraphNeuralNetworks/api/basic/#GraphNeuralNetworks.GNNChain).
"""
function MyCustomConvolutionLayerModel((nin, nout)::Pair, act=identity)

	# Glorot/Xavier-uniform initialization keeps activation variance approximately
	# constant across layers; the magic factor sqrt(6/(nin+nout)) is provided by Flux.
	W1 = Flux.glorot_uniform(nout, nin)
	W2 = Flux.glorot_uniform(nout, nin)

	# Allocate the bias with the same eltype as W1 (so f32/gpu transfers stay consistent),
	# then zero it. fill!(...) returns the same array, mutating it in place.
	b = fill!(similar(W1, nout), 0)

	return MyCustomConvolutionLayerModel(W1, W2, b, act)
end
