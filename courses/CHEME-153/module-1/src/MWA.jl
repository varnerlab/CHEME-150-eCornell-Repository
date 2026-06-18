# Local Multiplicative Weights Algorithm engine for the rain-forecasting activity.
# This extends the package generics build(...) and play(...) with methods for a
# generic prediction-from-expert-advice model (the form introduced in the lecture).

"""
    mutable struct MyMultiplicativeWeightsAlgorithmModel <: AbstractOnlineLearningModel

Holds a prediction-from-expert-advice problem solved with the Multiplicative Weights Algorithm.

### Fields
- `η::Float64`: learning rate, with `0 < η ≤ 1/2`.
- `n::Int64`: number of experts.
- `T::Int64`: number of rounds.
- `forecasts::Array{Int64,2}`: `T × n` matrix of expert predictions, each entry in `{-1, 1}`.
- `outcomes::Array{Int64,1}`: length-`T` vector of true outcomes, each entry in `{-1, 1}`.
- `weights::Array{Float64,2}`: `(T+1) × n` weight trajectory, populated by `play(...)`.
"""
mutable struct MyMultiplicativeWeightsAlgorithmModel <: AbstractOnlineLearningModel

    # parameters -
    η::Float64                  # learning rate
    n::Int64                    # number of experts
    T::Int64                    # number of rounds
    forecasts::Array{Int64,2}   # T × n expert predictions in {-1, 1}
    outcomes::Array{Int64,1}    # length T true outcomes in {-1, 1}
    weights::Array{Float64,2}   # (T+1) × n weight trajectory

    # default constructor -
    MyMultiplicativeWeightsAlgorithmModel() = new();
end

"""
    VLDataScienceMachineLearningPackage.build(modeltype::Type{MyMultiplicativeWeightsAlgorithmModel},
        data::NamedTuple) -> MyMultiplicativeWeightsAlgorithmModel

Build a `MyMultiplicativeWeightsAlgorithmModel`. The named tuple `data` must contain:
- `η::Float64`: learning rate.
- `forecasts::Array{Int64,2}`: `T × n` matrix of expert predictions in `{-1, 1}`.
- `outcomes::Array{Int64,1}`: length-`T` vector of true outcomes in `{-1, 1}`.

The number of experts `n` and rounds `T` are read from the shape of `forecasts`, and the
weight trajectory is initialized with `w_i^{(1)} = 1` for every expert.
"""
function VLDataScienceMachineLearningPackage.build(modeltype::Type{MyMultiplicativeWeightsAlgorithmModel},
    data::NamedTuple)::MyMultiplicativeWeightsAlgorithmModel

    # initialize -
    model = modeltype();

    # load data -
    model.η = data.η;
    model.forecasts = data.forecasts;
    model.outcomes = data.outcomes;
    model.T = size(data.forecasts, 1);
    model.n = size(data.forecasts, 2);
    model.weights = ones(Float64, model.T + 1, model.n); # w_i^{(1)} = 1 for all experts

    # return -
    return model;
end

"""
    VLDataScienceMachineLearningPackage.play(model::MyMultiplicativeWeightsAlgorithmModel) -> Dict{String,Any}

Run the Multiplicative Weights Algorithm for `T` rounds. Each round the model forms the
probability distribution `p^{(t)} = w^{(t)} / Σ w^{(t)}`, scores every expert with the cost
`m_i^{(t)} = -1` if correct and `+1` if incorrect, records the expected loss `p^{(t)} · m^{(t)}`,
and updates the weights multiplicatively as `w_i^{(t+1)} = w_i^{(t)}(1 - η m_i^{(t)})`. The
weights are renormalized each round to keep them bounded; this does not change the
probabilities, costs, or regret.

### Returns
A dictionary with keys:
- `"p"`: `T × n` matrix of per-round probability distributions.
- `"weights"`: `(T+1) × n` weight trajectory.
- `"cost"`: `T × n` matrix of expert costs in `{-1, 1}`.
- `"round_loss"`: length-`T` vector of expected losses `p^{(t)} · m^{(t)}`.
- `"algorithm_loss"`: total expected loss `Σ_t p^{(t)} · m^{(t)}`.
- `"expert_cumulative_loss"`: length-`n` vector of cumulative costs `Σ_t m_i^{(t)}` per expert.
- `"best_expert"`: index of the best expert in hindsight (minimum cumulative cost).
- `"best_expert_loss"`: cumulative cost of the best expert.
- `"regret"`: realized regret `algorithm_loss - best_expert_loss`.
"""
function VLDataScienceMachineLearningPackage.play(model::MyMultiplicativeWeightsAlgorithmModel)::Dict{String,Any}

    # initialize -
    η = model.η;
    n = model.n;
    T = model.T;
    forecasts = model.forecasts;
    outcomes = model.outcomes;
    weights = model.weights;            # (T+1) × n, row 1 = ones

    # storage -
    p = zeros(Float64, T, n);           # probability distribution each round
    cost = zeros(Float64, T, n);        # expert cost m_i^{(t)} ∈ {-1, 1}
    round_loss = zeros(Float64, T);     # expected loss p^{(t)} · m^{(t)}

    # main loop -
    for t ∈ 1:T

        # form the probability distribution from the current weights -
        p[t, :] = weights[t, :] ./ sum(weights[t, :]);

        # score every expert: -1 if correct, +1 if incorrect -
        for i ∈ 1:n
            cost[t, i] = (forecasts[t, i] == outcomes[t]) ? -1.0 : 1.0;
        end

        # expected loss of the algorithm this round -
        round_loss[t] = dot(p[t, :], cost[t, :]);

        # multiplicative weight update, then renormalize for numerical safety -
        for i ∈ 1:n
            weights[t+1, i] = weights[t, i]*(1 - η*cost[t, i]);
        end
        weights[t+1, :] ./= sum(weights[t+1, :]);
    end

    # cumulative costs and regret -
    expert_cumulative_loss = vec(sum(cost, dims = 1));   # length n
    algorithm_loss = sum(round_loss);
    best_expert = argmin(expert_cumulative_loss);
    best_expert_loss = expert_cumulative_loss[best_expert];
    regret = algorithm_loss - best_expert_loss;

    # package results -
    results = Dict{String,Any}();
    results["p"] = p;
    results["weights"] = weights;
    results["cost"] = cost;
    results["round_loss"] = round_loss;
    results["algorithm_loss"] = algorithm_loss;
    results["expert_cumulative_loss"] = expert_cumulative_loss;
    results["best_expert"] = best_expert;
    results["best_expert_loss"] = best_expert_loss;
    results["regret"] = regret;

    # return -
    return results;
end
