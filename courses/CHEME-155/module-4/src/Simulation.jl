"""
    simulate_fedbatch(p::MyFedBatchCHOParameters;
        u0::Vector{Float64} = [0.5, 0.5, 25.0, 4.0, 0.0, 0.0, 0.0],
        tspan::Tuple{Float64,Float64} = (0.0, 240.0),
        saveat::Float64 = 1.0) -> ODESolution

Simulate the fed-batch CHO bioreactor model with glucose-triggered square wave feeding.

### Arguments
- `p::MyFedBatchCHOParameters`: model parameters (includes feed policy).
- `u0::Vector{Float64}`: initial state [V (L), X (gDW/L), S_glc (mM), S_gln (mM), P (mg/L), Lac (mM), Amm (mM)].
- `tspan::Tuple{Float64,Float64}`: simulation time span in hours (default: 0 to 240, i.e., 10 days).
- `saveat::Float64`: time step for saving solution points in hours (default: 1.0).

### Returns
- `ODESolution`: solution object from OrdinaryDiffEq.
"""
function simulate_fedbatch(p::MyFedBatchCHOParameters;
    u0::Vector{Float64} = [0.5, 0.5, 25.0, 4.0, 0.0, 0.0, 0.0],
    tspan::Tuple{Float64,Float64} = (0.0, 240.0),
    saveat::Float64 = 1.0)

    # reset feed state before each simulation -
    p.feed_on = 0.0;

    # build ODE problem and feed callbacks -
    prob = ODEProblem(rhs!, u0, tspan, p);
    cbs = build_feed_callbacks(p);

    # solve with explicit Runge-Kutta method; reject steps that push any
    # concentration (states 2–7) below -1e-6 as a numerical safety net -
    sol = solve(prob, Tsit5();
        callback = cbs,
        saveat = saveat,
        abstol = 1e-8,
        reltol = 1e-6,
        maxiters = 1_000_000,
        isoutofdomain = (u, p, t) -> any(u[i] < -1e-6 for i in 2:7)
    );

    return sol;
end

"""
    generate_cho_dataset(conditions::Vector{Tuple{Float64,Float64,Float64}};
        u0::Vector{Float64} = [0.5, 0.5, 25.0, 4.0, 0.0, 0.0, 0.0],
        tspan::Tuple{Float64,Float64} = (0.0, 240.0),
        saveat::Float64 = 1.0) -> Tuple

Generate a dataset of CHO bioreactor simulations at different feed policy conditions.
Each condition is a tuple `(F_max, Glc_min, Glc_max)`.

### Arguments
- `conditions::Vector{Tuple{Float64,Float64,Float64}}`: vector of (F_max, Glc_min, Glc_max) tuples.
- `u0::Vector{Float64}`: initial state vector (default: V=0.5L, X=0.5gDW/L, S_glc=25mM, S_gln=4mM, P=0, Lac=0, Amm=0).
- `tspan::Tuple{Float64,Float64}`: simulation time span in hours (default: 0 to 240).
- `saveat::Float64`: time step for saving solution points in hours (default: 1.0).

### Returns
- `Tuple{Vector{Float64}, Vector{Matrix{Float64}}, Vector{Tuple{Float64,Float64,Float64}}}`:
  tuple of (time_vector, state_arrays, conditions) where each `state_arrays[i]` is a
  `(T x 7)` matrix (rows = time points, columns = states).
"""
function generate_cho_dataset(conditions::Vector{Tuple{Float64,Float64,Float64}};
    u0::Vector{Float64} = [0.5, 0.5, 25.0, 4.0, 0.0, 0.0, 0.0],
    tspan::Tuple{Float64,Float64} = (0.0, 240.0),
    saveat::Float64 = 1.0)

    # initialize -
    n_conditions = length(conditions);
    state_arrays = Vector{Matrix{Float64}}(undef, n_conditions);

    # build a common time grid so all conditions have the same number of time points.
    # the ODE solver with callbacks can return extra points at event times,
    # so we interpolate each solution onto this fixed grid -
    time_vector = collect(range(tspan[1], tspan[2], step = saveat));
    n_timepoints = length(time_vector);

    # simulate each condition -
    for (j, cond) in enumerate(conditions)

        # build parameters for this condition -
        p = build_default_parameters(; F_max = cond[1], Glc_min = cond[2], Glc_max = cond[3]);

        # simulate -
        sol = simulate_fedbatch(p; u0 = copy(u0), tspan = tspan, saveat = saveat);

        # interpolate solution onto the common time grid -
        state_matrix = zeros(Float64, n_timepoints, 7);
        for i in 1:n_timepoints
            state_matrix[i, :] = sol(time_vector[i]);
        end
        state_arrays[j] = state_matrix;
    end

    return (time_vector, state_arrays, conditions);
end

"""
    normalize_minmax_perstate(data_arrays::Vector{Matrix{Float64}},
        train_indices::Vector{Int}) -> Tuple

Apply per-state min-max normalization using bounds computed from training data only.
Each of the 7 states is normalized independently to [0, 1] using the global min and max
across all training curves for that state.

### Arguments
- `data_arrays::Vector{Matrix{Float64}}`: vector of `(T x 7)` state matrices.
- `train_indices::Vector{Int}`: indices of training curves (used to compute normalization bounds).

### Returns
- `Tuple{Vector{Matrix{Float64}}, Vector{Float64}, Vector{Float64}}`: tuple of
  (normalized_arrays, state_mins, state_maxs) where `state_mins` and `state_maxs` are
  vectors of length 7 holding the per-state normalization bounds.
"""
function normalize_minmax_perstate(data_arrays::Vector{Matrix{Float64}},
    train_indices::Vector{Int})::Tuple{Vector{Matrix{Float64}}, Vector{Float64}, Vector{Float64}}

    # compute per-state bounds from training data only -
    n_states = size(data_arrays[1], 2);
    state_mins = fill(Inf, n_states);
    state_maxs = fill(-Inf, n_states);
    for idx in train_indices
        for j in 1:n_states
            col_min = minimum(data_arrays[idx][:, j]);
            col_max = maximum(data_arrays[idx][:, j]);
            state_mins[j] = min(state_mins[j], col_min);
            state_maxs[j] = max(state_maxs[j], col_max);
        end
    end

    # normalize all curves using training bounds -
    n_curves = length(data_arrays);
    normalized_arrays = Vector{Matrix{Float64}}(undef, n_curves);
    for i in 1:n_curves
        normalized = similar(data_arrays[i]);
        for j in 1:n_states
            range_j = state_maxs[j] - state_mins[j];
            if range_j > 0.0
                normalized[:, j] = (data_arrays[i][:, j] .- state_mins[j]) ./ range_j;
            else
                normalized[:, j] .= 0.0;
            end
        end
        normalized_arrays[i] = normalized;
    end

    return (normalized_arrays, state_mins, state_maxs);
end

"""
    denormalize_minmax(data::Matrix{Float64}, state_mins::Vector{Float64},
        state_maxs::Vector{Float64}) -> Matrix{Float64}

Reverse per-state min-max normalization for a matrix.

### Arguments
- `data::Matrix{Float64}`: normalized matrix with values in [0, 1], shape `(T x n_states)`.
- `state_mins::Vector{Float64}`: per-state minimum values.
- `state_maxs::Vector{Float64}`: per-state maximum values.

### Returns
- `Matrix{Float64}`: data in original scale.
"""
function denormalize_minmax(data::Matrix{Float64}, state_mins::Vector{Float64},
    state_maxs::Vector{Float64})::Matrix{Float64}

    result = similar(data);
    for j in 1:length(state_mins)
        result[:, j] = data[:, j] .* (state_maxs[j] - state_mins[j]) .+ state_mins[j];
    end
    return result;
end

"""
    denormalize_minmax(data::Vector{Float64}, state_mins::Vector{Float64},
        state_maxs::Vector{Float64}) -> Vector{Float64}

Reverse per-state min-max normalization for a single time-step vector.

### Arguments
- `data::Vector{Float64}`: normalized vector of length `n_states`.
- `state_mins::Vector{Float64}`: per-state minimum values.
- `state_maxs::Vector{Float64}`: per-state maximum values.

### Returns
- `Vector{Float64}`: vector in original scale.
"""
function denormalize_minmax(data::Vector{Float64}, state_mins::Vector{Float64},
    state_maxs::Vector{Float64})::Vector{Float64}

    return data .* (state_maxs .- state_mins) .+ state_mins;
end
