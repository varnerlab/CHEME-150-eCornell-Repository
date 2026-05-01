"""
    relu(x) = max(zero(x), x)

Elementwise rectified linear unit. Used as the default activation in
`MyGCNLayer`.
"""
relu(x) = max(zero(x), x)

"""
    karate_club_adjacency() -> Matrix{Float64}

Return the dense `(34, 34)` adjacency matrix of Zachary's Karate Club graph
(34 nodes, 78 undirected edges). The graph is loaded from
`Graphs.smallgraph(:karate)` and converted to a symmetric `Float64` matrix
with zeros on the diagonal.

### Returns
- `A::Matrix{Float64}` of shape `(34, 34)`, symmetric, with `A[i, j] = 1` if
   nodes `i` and `j` are connected and `0` otherwise.
"""
function karate_club_adjacency()::Matrix{Float64}
    g = smallgraph(:karate)
    n = nv(g)
    A = zeros(Float64, n, n)
    for e in edges(g)
        i, j = src(e), dst(e)
        A[i, j] = 1.0
        A[j, i] = 1.0
    end
    return A
end

"""
    karate_club_communities() -> Vector{Int64}

Return the original two-community labels from
[Zachary (1977)](https://www.jstor.org/stable/3629752): `1` for Mr. Hi's
faction (the instructor's group, 16 nodes), `2` for Officer John A.'s faction
(the administrator's group, 18 nodes). The labels are indexed `1:34`,
matching the node numbering of `Graphs.smallgraph(:karate)`.

### Returns
- `labels::Vector{Int64}` of length `34`, with each entry in `{1, 2}`.
"""
function karate_club_communities()::Vector{Int64}
    # Mr. Hi (label 1):  nodes 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 20, 22
    # Officer (label 2): nodes 9, 10, 15, 16, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34
    mr_hi   = Set([1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 20, 22])
    labels  = ones(Int64, 34)
    for i in 1:34
        labels[i] = i in mr_hi ? 1 : 2
    end
    return labels
end

"""
    normalized_adjacency(A::AbstractMatrix) -> Matrix{Float64}

Return the symmetric-normalized propagation matrix
`P = D̂^(-1/2) (A + I) D̂^(-1/2)` of [Kipf and Welling (2017)](https://arxiv.org/abs/1609.02907),
where `Â = A + I` adds a self-loop at every node and `D̂` is the diagonal
degree matrix of `Â` (`D̂[i, i] = 1 + sum_j A[i, j]`).

### Arguments
- `A::AbstractMatrix`: adjacency matrix of shape `(N, N)`. Need not be binary.

### Returns
- `P::Matrix{Float64}` of shape `(N, N)`, symmetric.
"""
function normalized_adjacency(A::AbstractMatrix)::Matrix{Float64}
    @assert size(A, 1) == size(A, 2) "adjacency matrix must be square"
    n = size(A, 1)
    A_hat = A + I(n)
    d_hat = vec(sum(A_hat; dims = 2))
    D_inv_sqrt = Diagonal(1.0 ./ sqrt.(d_hat))
    return D_inv_sqrt * A_hat * D_inv_sqrt
end

"""
    gcn_forward(layer::MyGCNLayer, P::AbstractMatrix, H::AbstractMatrix) -> Matrix{Float64}

Apply one symmetric-normalized graph convolution layer to feature matrix `H`:

    H_out = layer.activation.(P * H * layer.W)

### Arguments
- `layer`: `MyGCNLayer` whose `W` and `activation` define the layer map.
- `P::AbstractMatrix`: propagation matrix of shape `(N, N)`, e.g. from
   `normalized_adjacency`.
- `H::AbstractMatrix`: input feature matrix of shape `(N, layer.d_in)`.

### Returns
- `H_out::Matrix{Float64}` of shape `(N, layer.d_out)`.
"""
function gcn_forward(layer::MyGCNLayer, P::AbstractMatrix, H::AbstractMatrix)::Matrix{Float64}
    @assert size(H, 2) == layer.d_in "H has $(size(H, 2)) input features but layer expects $(layer.d_in)"
    @assert size(P, 1) == size(P, 2) == size(H, 1) "P and H disagree on N"
    Z = P * H * layer.W
    return layer.activation.(Z)
end

"""
    gcn_stack_forward(layers::Vector{MyGCNLayer}, P::AbstractMatrix,
        X::AbstractMatrix) -> Matrix{Float64}

Apply a stack of `MyGCNLayer`s sequentially, sharing the same propagation
matrix `P` across all layers:

    H_0   = X
    H_l+1 = layers[l+1].activation.(P * H_l * layers[l+1].W)
    H_out = H_L

### Arguments
- `layers`: ordered vector of `MyGCNLayer` instances. The output dim of layer
   `l` must match the input dim of layer `l + 1`.
- `P::AbstractMatrix`: propagation matrix of shape `(N, N)`.
- `X::AbstractMatrix`: input feature matrix of shape `(N, layers[1].d_in)`.

### Returns
- `H_out::Matrix{Float64}` of shape `(N, layers[end].d_out)`.
"""
function gcn_stack_forward(layers::Vector{MyGCNLayer}, P::AbstractMatrix,
    X::AbstractMatrix)::Matrix{Float64}

    H = Matrix{Float64}(X)
    for layer in layers
        H = gcn_forward(layer, P, H)
    end
    return H
end

"""
    pca_2d(H::AbstractMatrix) -> Matrix{Float64}

Project the rows of `H` onto the top-2 principal components and return the
resulting `(N, 2)` matrix of 2D coordinates. Each column of `H` is
mean-centered before the SVD; the projection is onto the right-singular
vectors with the two largest singular values.

### Arguments
- `H::AbstractMatrix`: feature matrix of shape `(N, d)` with `d >= 2`.

### Returns
- `Y::Matrix{Float64}` of shape `(N, 2)`. Column 1 is PC1, column 2 is PC2.
"""
function pca_2d(H::AbstractMatrix)::Matrix{Float64}
    @assert size(H, 2) >= 2 "PCA needs at least 2 features"
    H_centered = H .- mean(H; dims = 1)
    F = svd(H_centered)
    return H_centered * F.V[:, 1:2]
end

"""
    pairwise_cosine_similarity(H::AbstractMatrix) -> Matrix{Float64}

Return the `(N, N)` matrix of pairwise cosine similarities between the rows
of `H`. Used to quantify how oversmoothing collapses node embeddings as the
GCN depth grows: at large depth every entry tends to one.

### Arguments
- `H::AbstractMatrix`: feature matrix of shape `(N, d)`.

### Returns
- `S::Matrix{Float64}` of shape `(N, N)`, with `S[i, j]` the cosine similarity
   between row `i` and row `j` of `H`. Rows of `H` with zero norm are mapped
   to a similarity of zero.
"""
function pairwise_cosine_similarity(H::AbstractMatrix)::Matrix{Float64}
    n = size(H, 1)
    norms = [norm(H[i, :]) for i in 1:n]
    S = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        if norms[i] > 0 && norms[j] > 0
            S[i, j] = dot(H[i, :], H[j, :]) / (norms[i] * norms[j])
        end
    end
    return S
end

"""
    eval_loss_accuracy(model, data_loader, device) -> (; loss, acc)

Compute the average logit-cross-entropy loss and top-1 classification accuracy of
`model` over every mini-batch yielded by `data_loader`.

# Arguments
- `model`:        a `GNNChain` (or any callable accepting `(g::GNNGraph, x)` that
                  returns per-graph logits of shape `(num_classes, num_graphs_in_batch)`).
- `data_loader`:  an [`MLUtils.DataLoader`](https://github.com/JuliaML/MLUtils.jl)
                  iterating over `(g, y)` pairs, where `g` is a batched `GNNGraph`
                  and `y` is a one-hot `(num_classes, num_graphs)` matrix.
- `device`:       a Flux device handle, typically `Flux.cpu` or `Flux.gpu`. Each
                  batch is moved to this device before the forward pass.

# Returns
- A `NamedTuple` `(loss, acc)` where:
  - `loss` is the per-graph mean of `Flux.logitcrossentropy`, rounded to 4 digits;
  - `acc`  is the top-1 classification accuracy as a percentage in `[0, 100]`,
           rounded to 2 digits.

Class accuracy uses `onecold(ŷ) == onecold(y)` (argmax of logits matches argmax of
one-hot targets) rather than per-element equality on the one-hot vectors.
"""
function eval_loss_accuracy(model, data_loader, device)

    # Running totals over all mini-batches in the loader.
    loss = 0.0
    acc  = 0
    ntot = 0

    for (g, y) in data_loader

        # Move the batched graph and the one-hot label matrix to the chosen device.
        g, y = g |> device, y |> device

        # Number of graphs in this batch; one column of `y` per graph.
        n = size(y, 2)

        # Forward pass: GNNChain returns per-graph logits of shape (num_classes, n).
        ŷ = model(g, g.ndata.x)

        # Weight each batch's loss by its size so the final mean is over graphs,
        # not over (variable-sized) batches.
        loss += logitcrossentropy(ŷ, y) * n

        # Top-1 class accuracy: argmax of logits matches argmax of one-hot labels.
        acc += sum(onecold(ŷ) .== onecold(y))

        ntot += n
    end

    # Average over total number of graphs and rescale accuracy to a percent.
    return (loss = round(loss / ntot, digits = 4),
            acc  = round(acc  * 100 / ntot, digits = 2))
end

"""
    train!(model; epochs::Int = 200, η::Float64 = 1e-2, infotime::Int = 10) -> nothing

Train `model` in place on the global `train_loader`, evaluating on the global
`test_loader` every `infotime` epochs and printing train/test loss and accuracy.

This is the *interactive* training entry point used in the lab notebook: it relies
on the globals `train_loader` and `test_loader` defined at notebook scope. For
programmatic use inside a function, where those globals do not exist, use
[`train_one_config`](src/Compute.jl) instead.

# Arguments
- `model`: a `GNNChain` to train. Modified in place: parameters change, and
           `model` is left on the chosen device.

# Keyword arguments
- `epochs::Int`:   number of full passes over the training set (default `200`).
- `η::Float64`:    Adam learning rate (default `1e-2`).
- `infotime::Int`: reporting cadence; train/test stats are printed at epoch `0`
                   and every `infotime` epochs thereafter (default `10`).

# Side effects
- Prints train and test `(loss, acc)` to stdout at evaluation epochs.
- Mutates the parameters of `model` via Adam updates.
"""
function train!(model; epochs=200, η=1e-2, infotime=10)

    # Pick the compute device. Flip to Flux.gpu for GPU training.
    # device = Flux.gpu
    device = Flux.cpu

    # Move the model to the chosen device. `|> device` is the Flux idiom.
    model = model |> device

    # Set up Adam optimizer state for all of the model's trainable parameters.
    # Flux.setup walks the model with @layer-registered types and collects W1/W2/b.
    opt_state = Flux.setup(Adam(η), model)

    # Closure that runs evaluation on both loaders and prints the result.
    # Lives inside train! so it can close over `model` and `device` without arguments.
    function report(epoch)
        train = eval_loss_accuracy(model, train_loader, device)
        test  = eval_loss_accuracy(model, test_loader,  device)
        println("# epoch = $epoch")
        println("train = $train")
        println("test = $test")
    end

    # Initial evaluation before any optimization (sanity check on weight init).
    report(0)

    for epoch in 1:epochs

        # One pass over the training set. Each iteration of the inner loop is one
        # mini-batch and one Adam step.
        for (g, y) in train_loader

            # Move this mini-batch to the device.
            g, y = g |> device, y |> device

            # Compute gradient of `logitcrossentropy(model(g), y)` w.r.t. all model
            # parameters. The do-block defines the loss function in-line.
            grads = Flux.gradient(model) do model
                ŷ = model(g, g.ndata.x)
                logitcrossentropy(ŷ, y)
            end

            # Apply one Adam step. grads[1] is the gradient w.r.t. the first
            # gradient-tracked argument (the model itself).
            Flux.Optimise.update!(opt_state, model, grads[1])
        end

        # Periodic reporting; epoch % infotime == 0 fires for 10, 20, ..., epochs.
        epoch % infotime == 0 && report(epoch)
    end
end


"""
    (l::MyCustomConvolutionLayerModel)(g::GNNGraph, x::AbstractMatrix) -> AbstractMatrix

Forward pass of a single message-passing layer.

For each node `i` in the (possibly batched) graph `g`, this computes
`x_i^new = act.(W1 * x_i + W2 * sum_{j ∈ N(i)} x_j + b)`.

# Arguments
- `g::GNNGraph`:       the (batched) graph defining the node connectivity.
- `x::AbstractMatrix`: per-node feature matrix of shape `(d_in, num_nodes)`.
                       Each column is one node's feature vector.

# Returns
- An `AbstractMatrix` of shape `(d_out, num_nodes)` with the updated per-node features.

# How the neighbor sum is computed
1. [`apply_edges`](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GraphNeuralNetworks.jl/stable/GNNGraphs/api/messagepassing/#GNNlib.apply_edges)
   runs the message function `(xi, xj, e) -> W2 * xj` over every directed edge `i → j`,
   producing one `d_out`-vector per edge. Result has shape `(d_out, num_edges)`.
2. [`aggregate_neighbors`](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GraphNeuralNetworks.jl/stable/GNNGraphs/api/messagepassing/#GNNlib.aggregate_neighbors)
   reduces those edge messages with `+` over each node's incoming edges, producing one
   `d_out`-vector per node. Result has shape `(d_out, num_nodes)`.

The two steps can be fused into a single
[`propagate`](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GraphNeuralNetworks.jl/stable/GNNGraphs/api/messagepassing/#GNNlib.propagate) call
(see the commented line below); we keep them split for readability.
"""
function (l::MyCustomConvolutionLayerModel)(g::GNNGraph, x::AbstractMatrix)

	# Message function: each edge i->j contributes l.W2 * x_j (the *neighbor's* feature
	# vector projected by W2). The underscore-prefixed args are required by the
	# apply_edges signature but unused in this message rule (no source-node or
	# edge-feature dependence).
	message(_xi, xj, _e) = l.W2 * xj

	# Run the message function over every edge.
	m = apply_edges(message, g, xj=x) # size [d_out, num_edges]

	# Sum-aggregate edge messages into per-node neighbor sums.
	xnew = aggregate_neighbors(g, +, m) # size [d_out, num_nodes]

	# Equivalent to the two lines above in one call:
	# m = propagate(message, g, +, xj=x)

	# Combine self-projection (W1*x), neighbor sum (xnew), and bias, then apply activation.
	# `.+` and `.act.` broadcast the per-layer bias and the activation across all nodes.
	return l.act.(l.W1*x .+ xnew .+ l.b)
end

# Register the custom layer with Flux's parameter-collection machinery so that
# Flux.setup, Flux.gradient, and Flux.Optimise.update! correctly find W1/W2/b
# as trainable parameters. Must be at top level: the macro emits global method
# definitions for Base.show, Functors.functor, etc., which Julia does not allow
# inside a function body.
Flux.@layer MyCustomConvolutionLayerModel

"""
    build_model(; nin::Int, nh::Int, nout::Int, dropout_p::Float64) -> GNNChain

Build a three-layer GNN classifier composed of `MyCustomConvolutionLayerModel`
stages, a global mean-pool readout, dropout regularization, and a dense classifier head.

# Keyword arguments
- `nin::Int`:           input node feature dimension (e.g., `7` for one-hot atom types in MUTAG).
- `nh::Int`:            hidden dimension shared across the three convolutional stages.
- `nout::Int`:          number of output classes for the classifier head.
- `dropout_p::Float64`: dropout probability applied to the pooled features (`0.0` disables dropout).

# Returns
- A [`GNNChain`](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GraphNeuralNetworks.jl/stable/GraphNeuralNetworks/api/basic/#GraphNeuralNetworks.GNNChain)
  with named stages `:input`, `:hidden`, `:output`, `:pool`, `:dropout`, `:dense`.
  The first three are message-passing iterations producing the node embedding matrix
  `H^(L) ∈ R^{N × nh}`; the remaining three form the task-specific readout that turns
  `H^(L)` into per-graph class logits.

Used in programmatic settings such as the Task 3 hyperparameter sweep, where a fresh
model is needed for each configuration. The notebook's Task 2 builds the same
architecture inline in a `let` block so students can read the wiring.
"""
function build_model(; nin::Int, nh::Int, nout::Int, dropout_p::Float64)
    return GNNChain(
        # Three message-passing iterations: nin -> nh -> nh -> nh.
        # The first two use ReLU; the third uses identity so the pooled features can
        # take negative values without being clipped before pooling.
        input   = MyCustomConvolutionLayerModel(nin => nh, relu),
        hidden  = MyCustomConvolutionLayerModel(nh  => nh, relu),
        output  = MyCustomConvolutionLayerModel(nh  => nh),

        # Graph-level readout: mean over each graph's nodes. Collapses the
        # variable-shape (nh, num_nodes_in_graph) into a fixed (nh,) per graph.
        pool    = GlobalPool(mean),

        # Regularize the pooled features so the dense head doesn't overfit.
        dropout = Dropout(dropout_p),

        # Classifier head: project pooled features to per-class logits.
        # Logits go straight into Flux.logitcrossentropy (no softmax here).
        dense   = Dense(nh, nout),
    )
end

"""
    train_one_config(training_graphs, ytrain, testing_graphs, ytest;
                     nin, nout, nh, dropout_p, batchsize, η, epochs)
        -> (; train_acc, test_acc, train_loss, test_loss)

Train a fresh GNN on `(training_graphs, ytrain)` with the given hyperparameters and
return its final train/test loss and accuracy. Self-contained: builds its own
`DataLoader` instances and its own model, so unlike [`train!`](src/Compute.jl) it
does not depend on global state.

# Positional arguments
- `training_graphs::Vector{GNNGraph}`: training graphs.
- `ytrain::AbstractMatrix`:            one-hot training labels of shape
                                       `(nout, length(training_graphs))`.
- `testing_graphs::Vector{GNNGraph}`:  held-out graphs.
- `ytest::AbstractMatrix`:             one-hot test labels of shape
                                       `(nout, length(testing_graphs))`.

# Keyword arguments
- `nin::Int`:           input node feature dimension.
- `nout::Int`:          number of output classes.
- `nh::Int`:            hidden dimension.
- `dropout_p::Float64`: dropout probability for the pooled features.
- `batchsize::Int`:     mini-batch size for the training loader. The test
                        batchsize is fixed at `10` since test-time batching
                        does not affect optimization.
- `η::Float64`:         Adam learning rate.
- `epochs::Int`:        number of full passes over the training set.

# Returns
- A `NamedTuple` of four numbers: `train_acc`, `test_acc` (percent), `train_loss`,
  `test_loss`.

# Reproducibility
The caller should `Random.seed!(...)` before calling this function for deterministic
results, since both data shuffling and Glorot-uniform initialization draw from
the global RNG.
"""
function train_one_config(training_graphs, ytrain, testing_graphs, ytest;
                          nin::Int, nout::Int,
                          nh::Int, dropout_p::Float64,
                          batchsize::Int, η::Float64, epochs::Int)

    # Build local DataLoaders so we don't mutate the notebook's global
    # train_loader / test_loader. shuffle=true on the training loader is critical
    # for Adam to escape the order of the dataset; shuffle=false on test gives
    # deterministic evaluation across configurations.
    local_train_loader = DataLoader((training_graphs, ytrain),
                                    batchsize=batchsize, shuffle=true,  collate=true)
    local_test_loader  = DataLoader((testing_graphs,  ytest),
                                    batchsize=10,        shuffle=false, collate=true)

    # CPU training; switch to Flux.gpu for GPU.
    device = Flux.cpu

    # Build a fresh model and optimizer state for this configuration. Each call
    # of build_model gives newly Glorot-initialized weights, so different
    # configurations do not share parameters.
    model     = build_model(nin=nin, nh=nh, nout=nout, dropout_p=dropout_p) |> device
    opt_state = Flux.setup(Adam(η), model)

    # Plain training loop with no progress reporting (the sweep runs many
    # configurations and prints a summary table at the end).
    for _ in 1:epochs
        for (g, y) in local_train_loader
            g, y = g |> device, y |> device
            grads = Flux.gradient(model) do model
                ŷ = model(g, g.ndata.x)
                logitcrossentropy(ŷ, y)
            end
            Flux.Optimise.update!(opt_state, model, grads[1])
        end
    end

    # Final-epoch evaluation on both splits. Note that Dropout's training-time
    # randomness is automatically turned off here because eval_loss_accuracy
    # does not call Flux.trainmode!; Flux defaults to test mode outside gradient.
    train_eval = eval_loss_accuracy(model, local_train_loader, device)
    test_eval  = eval_loss_accuracy(model, local_test_loader,  device)

    return (train_acc  = train_eval.acc,
            test_acc   = test_eval.acc,
            train_loss = train_eval.loss,
            test_loss  = test_eval.loss)
end
