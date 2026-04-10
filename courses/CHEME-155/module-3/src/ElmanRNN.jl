"""
    mutable struct MyElmanRNNModel

Holds the parameters of an Elman RNN with input dimension `d_in`, hidden dimension `h`,
and output dimension `d_out`.

### Fields
- `U::Matrix{Float64}`: recurrent weight matrix, size `h x h`.
- `W::Matrix{Float64}`: input weight matrix, size `h x d_in`.
- `V::Matrix{Float64}`: output weight matrix, size `d_out x h`.
- `bh::Vector{Float64}`: hidden bias vector, length `h`.
- `by::Vector{Float64}`: output bias vector, length `d_out`.
"""
mutable struct MyElmanRNNModel
    U::Matrix{Float64}   # h x h (recurrent weights)
    W::Matrix{Float64}   # h x d_in (input weights)
    V::Matrix{Float64}   # d_out x h (output weights)
    bh::Vector{Float64}  # h (hidden bias)
    by::Vector{Float64}  # d_out (output bias)
end

# register with Flux so that Flux.withgradient and Flux.setup can traverse parameters -
Flux.@layer MyElmanRNNModel

"""
    build_elman_rnn(d_in::Int, h::Int, d_out::Int; seed::Int = 42) -> MyElmanRNNModel

Initialize an Elman RNN with Xavier-scaled random weights and zero biases.

### Arguments
- `d_in::Int`: input dimension.
- `h::Int`: hidden state dimension.
- `d_out::Int`: output dimension.
- `seed::Int`: random seed for reproducibility (default: 42).

### Returns
- `MyElmanRNNModel`: initialized model.
"""
function build_elman_rnn(d_in::Int, h::Int, d_out::Int; seed::Int = 42)::MyElmanRNNModel

    rng = Random.MersenneTwister(seed);

    # Xavier initialization: scale ~ sqrt(2 / (fan_in + fan_out)) -
    U = randn(rng, h, h) .* sqrt(2.0 / (h + h));
    W = randn(rng, h, d_in) .* sqrt(2.0 / (h + d_in));
    V = randn(rng, d_out, h) .* sqrt(2.0 / (d_out + h));
    bh = zeros(Float64, h);
    by = zeros(Float64, d_out);

    return MyElmanRNNModel(U, W, V, bh, by)
end

"""
    forward_step(model::MyElmanRNNModel, x_t::Vector{Float64},
        h_prev::Vector{Float64}) -> Tuple{Vector{Float64}, Vector{Float64}}

Compute one time step of the Elman RNN.

### Arguments
- `model::MyElmanRNNModel`: the RNN model.
- `x_t::Vector{Float64}`: input vector at time `t`, length `d_in`.
- `h_prev::Vector{Float64}`: hidden state from time `t-1`, length `h`.

### Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: tuple of (y_t, h_t) where `y_t` is the output
   in `(0, 1)` (sigmoid-activated) and `h_t` is the updated hidden state.
"""
function forward_step(model::MyElmanRNNModel, x_t::Vector{Float64},
    h_prev::Vector{Float64})::Tuple{Vector{Float64}, Vector{Float64}}

    # hidden state update: h_t = tanh(U * h_{t-1} + W * x_t + bh) -
    h_t = tanh.(model.U * h_prev .+ model.W * x_t .+ model.bh);

    # output: y_t = sigmoid(V * h_t + by)
    # sigmoid bounds output to (0, 1), which is necessary because targets are normalized to [0, 1].
    # a linear output layer is unbounded: during teacher forcing, the model never sees its own
    # out-of-range predictions, so small errors compound immediately during autoregressive rollout.
    y_t = NNlib.sigmoid.(model.V * h_t .+ model.by);

    return (y_t, h_t)
end

"""
    forward_sequence(model::MyElmanRNNModel, X::Matrix{Float64},
        condition::Float64) -> Tuple{Matrix{Float64}, Matrix{Float64}}

Run the Elman RNN over a full input sequence using teacher forcing. At each time step,
the input is `[x_t; condition]` where `x_t` is the true value at time `t`.

### Arguments
- `model::MyElmanRNNModel`: the RNN model.
- `X::Matrix{Float64}`: input sequence of shape `(T, 1)` where `T` is the sequence length.
- `condition::Float64`: conditioning parameter (e.g., normalized TF concentration).

### Returns
- `Tuple{Matrix{Float64}, Matrix{Float64}}`: tuple of (predictions, hidden_states) where
   predictions has shape `(T-1, d_out)` and hidden_states has shape `(T, h)`.
"""
function forward_sequence(model::MyElmanRNNModel, X::Matrix{Float64},
    condition::Float64)

    # initialize hidden state -
    T = size(X, 1);
    h_t = zeros(Float64, length(model.bh)); # initial hidden state
    all_preds = Vector{Float64}();           # accumulate scalar predictions via vcat

    # forward pass with teacher forcing -
    # note: we use vcat (creates new vector) instead of push! or setindex! because
    # Zygote cannot differentiate through in-place array mutations -
    for t in 1:(T - 1)
        x_t = [X[t, 1]; condition]; # concatenate current value and condition parameter
        y_t, h_t = forward_step(model, x_t, h_t);
        all_preds = vcat(all_preds, y_t); # functional append: creates a new vector each step
    end

    # reshape flat vector into (T-1) x d_out matrix -
    predictions = reshape(all_preds, :, length(model.by));

    return (predictions, nothing)
end

"""
    predict_sequence(model::MyElmanRNNModel, x_0::Float64, condition::Float64,
        T::Int) -> Vector{Float64}

Generate a full predicted sequence using autoregressive rollout. Starting from `x_0`,
the model feeds its own predictions back as input at each time step.

### Arguments
- `model::MyElmanRNNModel`: the RNN model.
- `x_0::Float64`: initial value (normalized).
- `condition::Float64`: conditioning parameter (e.g., normalized TF concentration).
- `T::Int`: number of time steps to predict.

### Returns
- `Vector{Float64}`: predicted sequence of length `T` (including the initial value).
"""
function predict_sequence(model::MyElmanRNNModel, x_0::Float64, condition::Float64,
    T::Int)::Vector{Float64}

    # initialize -
    predictions = zeros(Float64, T);
    predictions[1] = x_0;
    h_t = zeros(Float64, length(model.bh));
    x_current = x_0;

    # autoregressive rollout -
    for t in 1:(T - 1)
        x_t = [x_current; condition];
        y_t, h_t = forward_step(model, x_t, h_t);
        x_current = y_t[1];
        predictions[t + 1] = x_current;
    end

    return predictions
end

"""
    count_parameters(model::MyElmanRNNModel) -> Int

Count the total number of trainable parameters in the Elman RNN.

### Arguments
- `model::MyElmanRNNModel`: the RNN model.

### Returns
- `Int`: total parameter count.
"""
function count_parameters(model::MyElmanRNNModel)::Int
    return length(model.U) + length(model.W) + length(model.V) + length(model.bh) + length(model.by)
end
