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
    # concentration (states 2–7) below -1e-6 as a numerical safety net.
    # Qualify the ODE solve so our own solve(::MyMimoLegSHippoModel, ...) in
    # Compute.jl does not shadow it.
    sol = OrdinaryDiffEq.solve(prob, Tsit5();
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
    compute_feed_on(glc::AbstractVector{<:Real}; feed_on0::Real = 0.0,
        slope_tol::Real = 1.0e-6) -> Vector{Float64}

Reconstruct the feed-on indicator at each sampled time step from the sign of
the glucose slope between consecutive samples. When the pump is off glucose
is monotonically consumed, so a rise means the feed fired during the interval;
when the pump is on glucose rebounds toward the feed concentration, so a fall
means the feed turned off. Concretely:

- `feed_on[t] = 1` if `glc[t] - glc[t-1] >  slope_tol`,
- `feed_on[t] = 0` if `glc[t] - glc[t-1] < -slope_tol`,
- otherwise `feed_on[t] = feed_on[t-1]` (hysteresis).

The initial sample inherits `feed_on0` (default 0, matching the simulator's
"feed starts OFF" convention). This detector is robust to the sampling issue
that the raw hysteresis replay has: the ODE callback triggers feed-on exactly
when glucose touches `Glc_min` and glucose rebounds immediately, so discrete
samples often never fall below `Glc_min`, but the slope sign flip across the
event is unambiguous.
"""
function compute_feed_on(glc::AbstractVector{<:Real};
    feed_on0::Real = 0.0,
    slope_tol::Real = 1.0e-6)::Vector{Float64}

    T = length(glc)
    f = zeros(Float64, T)
    state = Float64(feed_on0)
    f[1] = state
    for t in 2:T
        dg = glc[t] - glc[t - 1]
        if dg >  slope_tol
            state = 1.0
        elseif dg < -slope_tol
            state = 0.0
        end
        f[t] = state
    end
    return f
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

"""
    apply_minmax_perstate(data_arrays::Vector{Matrix{Float64}},
        state_mins::Vector{Float64},
        state_maxs::Vector{Float64}) -> Vector{Matrix{Float64}}

Apply an existing per-state min-max scaler (produced by
`normalize_minmax_perstate`) to a new vector of trajectories. Used to
normalize the clean ground-truth arrays onto the same `[0, 1]` axis defined by
the noisy-training scaler, so normalized-MSE scoring compares predictions and
truth on a single comparable scale.

### Arguments
- `data_arrays::Vector{Matrix{Float64}}`: natural-unit trajectories to
  transform.
- `state_mins::Vector{Float64}`, `state_maxs::Vector{Float64}`: per-state
  bounds returned by `normalize_minmax_perstate` on the training set.

### Returns
- `Vector{Matrix{Float64}}`: normalized copies on the same `[0, 1]` axis as
  the input scaler.
"""
function apply_minmax_perstate(data_arrays::Vector{Matrix{Float64}},
    state_mins::Vector{Float64},
    state_maxs::Vector{Float64})::Vector{Matrix{Float64}}

    n_curves = length(data_arrays);
    n_states = length(state_mins);
    normalized_arrays = Vector{Matrix{Float64}}(undef, n_curves);
    for i in 1:n_curves
        m = data_arrays[i];
        out = similar(m);
        for j in 1:n_states
            r = state_maxs[j] - state_mins[j];
            if r > 0.0
                out[:, j] = (m[:, j] .- state_mins[j]) ./ r;
            else
                out[:, j] .= 0.0;
            end
        end
        normalized_arrays[i] = out;
    end
    return normalized_arrays;
end

"""
    add_measurement_noise(state_arrays_true::Vector{Matrix{Float64}},
        σ_rel::Union{Real,AbstractVector{<:Real}},
        train_indices::Vector{Int};
        rng::AbstractRNG = Random.default_rng(),
        exclude_indices::Vector{Int} = Int[]) -> Vector{Matrix{Float64}}

Return noisy copies of each trajectory in `state_arrays_true` by adding iid
Gaussian observation noise to every state at every time step. The per-state
standard deviation is range-relative, scaled to the state's dynamic range on
the training set only:

    σ_j = σ_rel * (max_j - min_j)          (scalar σ_rel)
    σ_j = σ_rel[j] * (max_j - min_j)       (per-state σ_rel)

This gives one knob (`σ_rel = 0.05` means "5% of each state's natural range")
that generalizes across states with very different magnitudes (gDW/L, mM,
mg/L). Passing `σ_rel = 0` returns exact copies of the input arrays.

### Arguments
- `state_arrays_true::Vector{Matrix{Float64}}`: clean ODE-generated natural-unit
  trajectories (the "truth" that a real experiment would never see directly).
- `σ_rel`: scalar relative noise level or per-state vector of length `n_states`.
- `train_indices::Vector{Int}`: indices used to compute the per-state range
  (mirrors the normalization convention: no test info leaks into the σ scale).

### Keyword arguments
- `rng::AbstractRNG`: RNG for the noise draws (reproducibility).
- `exclude_indices::Vector{Int}`: state indices to leave noise-free (for
  example, `[1]` to skip reactor volume since it comes from a flowmeter rather
  than a bio-assay).

### Returns
- `Vector{Matrix{Float64}}`: noisy natural-unit trajectories, same shape as
  the input.
"""
function add_measurement_noise(state_arrays_true::Vector{Matrix{Float64}},
    σ_rel::Union{Real,AbstractVector{<:Real}},
    train_indices::Vector{Int};
    rng::AbstractRNG = Random.default_rng(),
    exclude_indices::Vector{Int} = Int[])::Vector{Matrix{Float64}}

    n_states = size(state_arrays_true[1], 2);
    σ_vec = σ_rel isa Real ? fill(Float64(σ_rel), n_states) : Float64.(collect(σ_rel));
    length(σ_vec) == n_states || error("σ_rel vector length ($(length(σ_vec))) does not match n_states ($(n_states))");

    # compute per-state range on training set only (no leakage from the held-out runs)
    state_mins = fill(Inf, n_states);
    state_maxs = fill(-Inf, n_states);
    for idx in train_indices
        for j in 1:n_states
            col_min = minimum(state_arrays_true[idx][:, j]);
            col_max = maximum(state_arrays_true[idx][:, j]);
            state_mins[j] = min(state_mins[j], col_min);
            state_maxs[j] = max(state_maxs[j], col_max);
        end
    end
    σ_per_state = [σ_vec[j] * (state_maxs[j] - state_mins[j]) for j in 1:n_states];
    for j in exclude_indices
        σ_per_state[j] = 0.0;
    end

    # draw iid Gaussian noise per (condition, time, state) at the per-state σ_j
    n = length(state_arrays_true);
    out = Vector{Matrix{Float64}}(undef, n);
    for i in 1:n
        out[i] = copy(state_arrays_true[i]);
        T = size(out[i], 1);
        for j in 1:n_states
            if σ_per_state[j] > 0.0
                out[i][:, j] .+= σ_per_state[j] .* randn(rng, T);
            end
        end
    end
    return out;
end
