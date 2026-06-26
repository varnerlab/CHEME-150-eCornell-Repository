# ===== GRID WORLD (lava-world) ============================================================================== #
# PRIVATE METHODS BELOW HERE ================================================================================= #
function _world(model::MyRectangularGridWorldModel, s::Int, a::Int)::Tuple{Int,Float64}
    model, a # dummy default; replace with a problem-specific worldmodel kwarg in solve(...)
    return (s, 0.0);
end
# PRIVATE METHODS ABOVE HERE ================================================================================= #

"""
    solve(agent::MyQLearningAgentModel, environment::MyRectangularGridWorldModel;
        maxsteps::Int = 100, δ::Float64 = 0.02, worldmodel::Function = _world) -> MyQLearningAgentModel

Simulate the Q-Learning agent in a rectangular grid world for a maximum number of steps per starting state.

### Arguments
- `agent::MyQLearningAgentModel`: the Q-Learning agent model.
- `environment::MyRectangularGridWorldModel`: the environment model.
- `maxsteps::Int = 100`: the maximum number of steps to simulate from each starting state.
- `δ::Float64 = 0.02`: the convergence threshold on the change in `Q` between iterations.
- `worldmodel::Function = _world`: function `(environment, s, a) -> (s′, r)` that returns the next state and reward.

### Returns
- `MyQLearningAgentModel`: the updated Q-Learning agent model after simulation.
"""
function solve(agent::MyQLearningAgentModel, environment::MyRectangularGridWorldModel;
    maxsteps::Int = 100, δ::Float64 = 0.02, worldmodel::Function = _world)::MyQLearningAgentModel

    # initialize -
    actions = agent.actions;
    K = length(actions); # number of actions
    states = agent.states;
    Q₁ = agent.Q;
    γ = agent.γ;

    # simulation loop -
    for s ∈ states

        # initialize t -
        t = 1;
        has_converged = false;
        αₜ = copy(agent.α);
        while (has_converged == false)

            # compute the ϵ -
            ϵₜ = (1.0/(t^(1/3)))*(log(K*t))^(1/3);
            p = rand();

            aₜ = nothing;
            if p ≤ ϵₜ
                aₜ = rand(1:K); # generate a random action
            elseif p > ϵₜ
                aₜ = argmax(Q₁[s,:]); # select the greedy action, given state s
            end

            # compute new state and reward -
            s′, r = worldmodel(environment, s, aₜ);

            # use the update rule to update Q -
            Q₂ = copy(Q₁);
            Q₁[s,aₜ] += αₜ*(r + γ*maximum(Q₁[s′,:]) - Q₁[s,aₜ])

            # update stuff -
            s = s′; # state update
            t += 1; # time update
            αₜ = 0.99*αₜ; # update the learning rate

            # check if we have converged -
            if ((t > maxsteps) || norm(Q₂ - Q₁) < δ)
                has_converged = true;
            end
        end
    end

    agent.Q = Q₁; # update the model

    # return -
    return agent
end

# ===== ADAPTIVE DOSING ENVIRONMENT ========================================================================== #
# PRIVATE METHODS BELOW HERE ================================================================================= #
_Tof(m::MyAdaptiveDosingModel, i::Int)::Float64 = (i - 1)*m.step;
_Zof(m::MyAdaptiveDosingModel, j::Int)::Float64 = (j - 1)*m.step;
_snap(m::MyAdaptiveDosingModel, x::Float64)::Int = clamp(round(Int, x/m.step) + 1, 1, m.nlevels);
_isdead_ij(m::MyAdaptiveDosingModel, ::Int, j::Int)::Bool  = _Zof(m, j) ≥ m.Zlethal;
_iscured_ij(m::MyAdaptiveDosingModel, i::Int, j::Int)::Bool = (_Tof(m, i) ≤ m.Tcure) && (_isdead_ij(m, i, j) == false);
_isterminal_ij(m::MyAdaptiveDosingModel, i::Int, j::Int)::Bool = _isdead_ij(m, i, j) || _iscured_ij(m, i, j);
# PRIVATE METHODS ABOVE HERE ================================================================================= #

"""
    stateindex(model::MyAdaptiveDosingModel, T::Float64, Z::Float64) -> Int

Return the grid state index nearest to the continuous patient state `(T, Z)`.
"""
stateindex(model::MyAdaptiveDosingModel, T::Float64, Z::Float64)::Int =
    model.states[(_snap(model, T), _snap(model, Z))];

"""
    decode(model::MyAdaptiveDosingModel, s::Int) -> Tuple{Float64,Float64}

Return the continuous patient state `(T, Z)` for grid state index `s`.
"""
function decode(model::MyAdaptiveDosingModel, s::Int)::Tuple{Float64,Float64}
    (i, j) = model.coordinates[s];
    return (_Tof(model, i), _Zof(model, j));
end

"""
    isterminal(model::MyAdaptiveDosingModel, s::Int) -> Bool

Return `true` if state `s` is absorbing (cured or toxic death).
"""
function isterminal(model::MyAdaptiveDosingModel, s::Int)::Bool
    (i, j) = model.coordinates[s];
    return _isterminal_ij(model, i, j);
end

"""
    world(model::MyAdaptiveDosingModel, s::Int, a::Int) -> Tuple{Int,Float64}

The environment dynamics. Given state `s` and action `a` (a dose), return the next state
`s′` and the reward `r`. Tumor `T` follows logistic growth with dose-dependent log-kill,
toxicity `Z` follows first-order loading and clearance, and the continuous next state is
snapped to the nearest grid cell. Absorbing states return `(s, 0.0)`.
"""
function world(model::MyAdaptiveDosingModel, s::Int, a::Int)::Tuple{Int,Float64}

    (i, j) = model.coordinates[s];
    _isterminal_ij(model, i, j) && return (s, 0.0); # absorbing

    # current state and dose -
    T = _Tof(model, i);
    Z = _Zof(model, j);
    u = model.doses[a];

    # one-cycle dynamics -
    T′ = clamp(T + model.g*T*(1 - T) - model.k*u*T, 0.0, 1.0);
    Z′ = clamp(Z + model.a*u - model.c*Z,           0.0, 1.0);

    # snap to the grid and look up the next state -
    i′ = _snap(model, T′);
    j′ = _snap(model, Z′);
    s′ = model.states[(i′, j′)];

    # reward -
    r = 0.0;
    if (_isdead_ij(model, i′, j′) == true)
        r = model.Rdead;
    elseif (_iscured_ij(model, i′, j′) == true)
        r = model.Rcure;
    else
        r = -(model.wT*_Tof(model, i′) + model.wZ*_Zof(model, j′) + model.cost*u);
    end

    return (s′, r);
end

"""
    solve(model::MyAdaptiveDosingModel; γ::Float64 = 0.95, tol::Float64 = 1e-8,
        maxiter::Int = 20000) -> Array{Float64,2}

Compute the optimal action-value table `Q★` for the dosing environment by **value iteration**
(this is the model-based ground truth, since the dynamics in `world(...)` are known). Returns
a `nstates × nactions` matrix; the optimal policy is `policy(Q★)`.
"""
function solve(model::MyAdaptiveDosingModel; γ::Float64 = 0.95, tol::Float64 = 1e-8,
    maxiter::Int = 20000)::Array{Float64,2}

    nstates  = model.nlevels^2;
    nactions = length(model.doses);

    # precompute the transition and reward tables -
    S′ = Array{Int,2}(undef, nstates, nactions);
    R  = Array{Float64,2}(undef, nstates, nactions);
    terminal = falses(nstates);
    for s ∈ 1:nstates
        terminal[s] = isterminal(model, s);
        for a ∈ 1:nactions
            (s′, r) = world(model, s, a);
            S′[s,a] = s′;
            R[s,a]  = r;
        end
    end

    # value iteration on the state-value function U -
    U = zeros(nstates);
    for _ ∈ 1:maxiter
        Δ = 0.0;
        for s ∈ 1:nstates
            terminal[s] && continue;
            best = -Inf;
            for a ∈ 1:nactions
                q = R[s,a] + γ*U[S′[s,a]];
                best = max(best, q);
            end
            Δ = max(Δ, abs(best - U[s]));
            U[s] = best;
        end
        Δ < tol && break;
    end

    # assemble the optimal action-value table Q★ -
    Q = zeros(nstates, nactions);
    for s ∈ 1:nstates
        terminal[s] && continue;
        for a ∈ 1:nactions
            Q[s,a] = R[s,a] + γ*U[S′[s,a]];
        end
    end

    return Q;
end

"""
    solve(agent::MyQLearningAgentModel, model::MyAdaptiveDosingModel;
        episodes::Int = 200_000, maxsteps::Int = 80) -> MyQLearningAgentModel

Train the Q-learning agent on the dosing environment **model-free** (the agent only sees
sampled transitions `(s, a, r, s′)` from `world(...)`; it does not use the dynamics directly).
Each episode starts from a random non-terminal state and runs until an absorbing state or
`maxsteps`. The step size follows a `1/N(s,a)` (sample-average) schedule, which satisfies the
Robbins–Monro convergence conditions; exploration is ε-greedy with ε annealed over episodes.
Returns the agent with its learned `Q` table; the learned policy is `policy(agent.Q)`.
"""
function solve(agent::MyQLearningAgentModel, model::MyAdaptiveDosingModel;
    episodes::Int = 200_000, maxsteps::Int = 80)::MyQLearningAgentModel

    Q = agent.Q;
    γ = agent.γ;
    nstates  = model.nlevels^2;
    nactions = length(model.doses);
    N = zeros(Int, nstates, nactions); # visit counts for the 1/N step size

    # non-terminal start states -
    starts = [s for s ∈ 1:nstates if isterminal(model, s) == false];

    for ep ∈ 1:episodes
        ϵ = max(0.05, 1.0 - ep/(0.7*episodes)); # anneal exploration
        s = starts[rand(1:length(starts))];
        for _ ∈ 1:maxsteps
            isterminal(model, s) && break;

            # ε-greedy action -
            a = (rand() < ϵ) ? rand(1:nactions) : argmax(@view Q[s,:]);

            # sample the world -
            (s′, r) = world(model, s, a);

            # temporal-difference update with a 1/N step size -
            N[s,a] += 1;
            α = 1.0/N[s,a];
            target = isterminal(model, s′) ? r : r + γ*maximum(@view Q[s′,:]);
            Q[s,a] += α*(target - Q[s,a]);

            s = s′;
        end
    end

    agent.Q = Q;
    return agent;
end

"""
    simulate(model::MyAdaptiveDosingModel, π::Array{Int,1}, s0::Int; maxsteps::Int = 100)

Roll out policy `π` from start state `s0`. Returns a named tuple with the trajectory
vectors `T`, `Z`, `dose`, the `outcome` (`:cured`, `:dead`, or `:timeout`), and the number
of dosing `steps` taken.
"""
function simulate(model::MyAdaptiveDosingModel, π::Array{Int,1}, s0::Int; maxsteps::Int = 100)
    Ts = Float64[]; Zs = Float64[]; us = Float64[];
    s = s0;
    for t ∈ 1:maxsteps
        (i, j) = model.coordinates[s];
        push!(Ts, _Tof(model, i));
        push!(Zs, _Zof(model, j));
        if _isterminal_ij(model, i, j)
            push!(us, 0.0);
            return (T = Ts, Z = Zs, dose = us,
                outcome = (_iscured_ij(model, i, j) ? :cured : :dead), steps = t - 1);
        end
        a = π[s];
        push!(us, model.doses[a]);
        (s, _) = world(model, s, a);
    end
    return (T = Ts, Z = Zs, dose = us, outcome = :timeout, steps = maxsteps);
end

# ===== POLICY EXTRACTION (shared) =========================================================================== #
"""
    policy(Q_array::Array{Float64,2}) -> Array{Int,1}

Compute the greedy policy from a Q-value table: for each state (row), return the action
(column) with the largest Q-value. Works for both the value-iteration `Q★` and a learned `Q`.
"""
function policy(Q_array::Array{Float64,2})::Array{Int64,1}
    (NR, _) = size(Q_array);
    π_array = Array{Int64,1}(undef, NR);
    for s ∈ 1:NR
        π_array[s] = argmax(Q_array[s,:]);
    end
    return π_array;
end

"""
    mypolicy(Q_array::Array{Float64,2}) -> Array{Int,1}

Alias for [`policy`](@ref).
"""
mypolicy(Q_array::Array{Float64,2})::Array{Int64,1} = policy(Q_array);
