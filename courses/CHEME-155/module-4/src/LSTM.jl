"""
    mutable struct MyLSTMModel

Holds the parameters of an LSTM network with input dimension `d_in`, hidden dimension `h`,
and output dimension `d_out`. The LSTM has four gates (forget, input, candidate, output)
and an output projection layer.

### Fields
- `Wf, Uf, bf`: forget gate weights and bias.
- `Wi, Ui, bi`: input gate weights and bias.
- `Wc, Uc, bc`: candidate (cell input) weights and bias.
- `Wo, Uo, bo`: output gate weights and bias.
- `Vy::Matrix{Float64}`: output projection matrix, size `d_out x h`.
- `by::Vector{Float64}`: output bias vector, length `d_out`.
"""
mutable struct MyLSTMModel

    # forget gate -
    Wf::Matrix{Float64}     # h x d_in
    Uf::Matrix{Float64}     # h x h
    bf::Vector{Float64}     # h

    # input gate -
    Wi::Matrix{Float64}     # h x d_in
    Ui::Matrix{Float64}     # h x h
    bi::Vector{Float64}     # h

    # candidate (cell input) -
    Wc::Matrix{Float64}     # h x d_in
    Uc::Matrix{Float64}     # h x h
    bc::Vector{Float64}     # h

    # output gate -
    Wo::Matrix{Float64}     # h x d_in
    Uo::Matrix{Float64}     # h x h
    bo::Vector{Float64}     # h

    # output projection -
    Vy::Matrix{Float64}     # d_out x h
    by::Vector{Float64}     # d_out
end

# register with Flux so that Flux.withgradient and Flux.setup can traverse parameters -
Flux.@layer MyLSTMModel

"""
    build_lstm(d_in::Int, h::Int, d_out::Int; seed::Int = 42) -> MyLSTMModel

Initialize an LSTM with Xavier-scaled random weights. The forget gate bias is initialized
to 1.0 to encourage remembering at the start of training. All other biases are zero.

### Arguments
- `d_in::Int`: input dimension.
- `h::Int`: hidden state dimension.
- `d_out::Int`: output dimension.
- `seed::Int`: random seed for reproducibility (default: 42).

### Returns
- `MyLSTMModel`: initialized model.
"""
function build_lstm(d_in::Int, h::Int, d_out::Int; seed::Int = 42)::MyLSTMModel

    rng = Random.MersenneTwister(seed);

    # Xavier initialization helper -
    xavier(rng, rows, cols) = randn(rng, rows, cols) .* sqrt(2.0 / (rows + cols));

    # forget gate (bias = 1.0 to encourage remembering) -
    Wf = xavier(rng, h, d_in);
    Uf = xavier(rng, h, h);
    bf = ones(Float64, h);

    # input gate -
    Wi = xavier(rng, h, d_in);
    Ui = xavier(rng, h, h);
    bi = zeros(Float64, h);

    # candidate -
    Wc = xavier(rng, h, d_in);
    Uc = xavier(rng, h, h);
    bc = zeros(Float64, h);

    # output gate -
    Wo = xavier(rng, h, d_in);
    Uo = xavier(rng, h, h);
    bo = zeros(Float64, h);

    # output projection -
    Vy = xavier(rng, d_out, h);
    by = zeros(Float64, d_out);

    return MyLSTMModel(Wf, Uf, bf, Wi, Ui, bi, Wc, Uc, bc, Wo, Uo, bo, Vy, by);
end

"""
    forward_step(model::MyLSTMModel, x_t::Vector{Float64},
        h_prev::Vector{Float64}, c_prev::Vector{Float64}) -> Tuple

Compute one time step of the LSTM.

### Arguments
- `model::MyLSTMModel`: the LSTM model.
- `x_t::Vector{Float64}`: input vector at time `t`, length `d_in`.
- `h_prev::Vector{Float64}`: hidden state from time `t-1`, length `h`.
- `c_prev::Vector{Float64}`: cell state from time `t-1`, length `h`.

### Returns
- `Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}`: tuple of (y_t, h_t, c_t)
   where `y_t` is the output in (0, 1) (sigmoid-activated), `h_t` is the updated hidden state,
   and `c_t` is the updated cell state.
"""
function forward_step(model::MyLSTMModel, x_t::Vector{Float64},
    h_prev::Vector{Float64}, c_prev::Vector{Float64})::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}

    # forget gate: what to discard from cell state -
    f_t = NNlib.sigmoid.(model.Wf * x_t .+ model.Uf * h_prev .+ model.bf);

    # input gate: what new information to store -
    i_t = NNlib.sigmoid.(model.Wi * x_t .+ model.Ui * h_prev .+ model.bi);

    # candidate cell state -
    c_tilde = tanh.(model.Wc * x_t .+ model.Uc * h_prev .+ model.bc);

    # cell state update: forget old + add new -
    c_t = f_t .* c_prev .+ i_t .* c_tilde;

    # output gate: what to expose from cell state -
    o_t = NNlib.sigmoid.(model.Wo * x_t .+ model.Uo * h_prev .+ model.bo);

    # hidden state: gated cell state -
    h_t = o_t .* tanh.(c_t);

    # output: sigmoid bounds to (0, 1) because targets are normalized to [0, 1] -
    y_t = NNlib.sigmoid.(model.Vy * h_t .+ model.by);

    return (y_t, h_t, c_t);
end

"""
    forward_sequence(model::MyLSTMModel, X::Matrix{Float64},
        condition_vec::Vector{Float64}) -> Tuple

Run the LSTM over a full input sequence using teacher forcing. At each time step,
the input is `[X[t,:]; condition_vec]` where `X[t,:]` is the true state vector at time `t`.

### Arguments
- `model::MyLSTMModel`: the LSTM model.
- `X::Matrix{Float64}`: input sequence of shape `(T, n_states)`.
- `condition_vec::Vector{Float64}`: conditioning parameters (e.g., normalized feed policy).

### Returns
- `Tuple{Matrix{Float64}, Nothing}`: tuple of (predictions, nothing) where
   predictions has shape `(T-1, d_out)`.
"""
function forward_sequence(model::MyLSTMModel, X::Matrix{Float64},
    condition_vec::Vector{Float64})

    # initialize hidden and cell states -
    T = size(X, 1);
    h_t = zeros(Float64, length(model.bf));
    c_t = zeros(Float64, length(model.bf));
    all_preds = Vector{Float64}();

    # forward pass with teacher forcing -
    # note: we use vcat (creates new vector) instead of push! or setindex! because
    # Zygote cannot differentiate through in-place array mutations -
    for t in 1:(T - 1)
        x_t = vcat(X[t, :], condition_vec);
        y_t, h_t, c_t = forward_step(model, x_t, h_t, c_t);
        all_preds = vcat(all_preds, y_t);
    end

    # reshape flat vector into (T-1) x d_out matrix -
    d_out = length(model.by);
    predictions = reshape(all_preds, d_out, T - 1)';

    return (predictions, nothing);
end

"""
    predict_sequence(model::MyLSTMModel, x_0_vec::Vector{Float64},
        condition_vec::Vector{Float64}, T::Int) -> Matrix{Float64}

Generate a full predicted sequence using autoregressive rollout. Starting from `x_0_vec`,
the model feeds its own predictions back as input at each time step.

### Arguments
- `model::MyLSTMModel`: the LSTM model.
- `x_0_vec::Vector{Float64}`: initial state vector (normalized), length `n_states`.
- `condition_vec::Vector{Float64}`: conditioning parameters (e.g., normalized feed policy).
- `T::Int`: number of time steps to predict.

### Returns
- `Matrix{Float64}`: predicted sequence of shape `(T, n_states)` (including the initial state).
"""
function predict_sequence(model::MyLSTMModel, x_0_vec::Vector{Float64},
    condition_vec::Vector{Float64}, T::Int)::Matrix{Float64}

    # initialize -
    n_states = length(x_0_vec);
    predictions = zeros(Float64, T, n_states);
    predictions[1, :] = x_0_vec;
    h_t = zeros(Float64, length(model.bf));
    c_t = zeros(Float64, length(model.bf));
    x_current = x_0_vec;

    # autoregressive rollout -
    for t in 1:(T - 1)
        x_t = vcat(x_current, condition_vec);
        y_t, h_t, c_t = forward_step(model, x_t, h_t, c_t);
        x_current = y_t;
        predictions[t + 1, :] = x_current;
    end

    return predictions;
end

"""
    count_parameters(model::MyLSTMModel) -> Int

Count the total number of trainable parameters in the LSTM.

### Arguments
- `model::MyLSTMModel`: the LSTM model.

### Returns
- `Int`: total parameter count.
"""
function count_parameters(model::MyLSTMModel)::Int
    return (
        length(model.Wf) + length(model.Uf) + length(model.bf) +
        length(model.Wi) + length(model.Ui) + length(model.bi) +
        length(model.Wc) + length(model.Uc) + length(model.bc) +
        length(model.Wo) + length(model.Uo) + length(model.bo) +
        length(model.Vy) + length(model.by)
    );
end
