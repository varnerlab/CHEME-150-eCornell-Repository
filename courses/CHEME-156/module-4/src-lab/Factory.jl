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
