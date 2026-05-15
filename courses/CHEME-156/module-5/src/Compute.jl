# PRIVATE METHODS BELOW HERE ================================================================================= #
function _world(model::MyRectangularGridWorldModel, s::Int, a::Int)::Tuple{Int,Float64}
    model, a # dummy default; replace with a problem-specific worldmodel kwarg in solve(...)
    return (s, 0.0);
end

@inline function _is_lava(x::Real, y::Real, context::MyDQNworldContextModel)::Bool
    r = context.lava_radius;
    @inbounds for (lx, ly) ∈ context.lava
        if (x - lx)^2 + (y - ly)^2 ≤ r^2
            return true;
        end
    end
    return false;
end

@inline function _is_charger(x::Real, y::Real, context::MyDQNworldContextModel)::Bool
    cx, cy = context.charger;
    return (x - cx)^2 + (y - cy)^2 ≤ context.charger_radius^2;
end

"""
    _dqn_update!(M, T, optstate, loss_fn, replaybuffer, B, γ, actions)

One DQN gradient step. Samples B transitions from the replay buffer, builds
the bootstrap target `y_i = r_i + γ (1 − done_i) max_a' Q'(s′_i, a')` from the
target network T, and minimizes the MSE between `[Q_θ(s_i)]_{a_i}` and `y_i`
on the action that was actually taken.
"""
function _dqn_update!(M, T, optstate, loss_fn,
        replaybuffer::CircularBuffer, B::Int, γ::Float32,
        actions::Dict{Int64, Vector{Float32}})

    minibatch = rand(replaybuffer, B);

    # build a stable mapping: action vector → index
    action_index = Dict{Vector{Float32}, Int}(v => k for (k, v) ∈ actions);

    # stack the minibatch into matrices for one batched forward pass
    n = length(minibatch);
    nin = length(minibatch[1][1]);
    S = Array{Float32}(undef, nin, n);
    Sp = Array{Float32}(undef, nin, n);
    rs = Array{Float32}(undef, n);
    dones = Array{Float32}(undef, n);
    a_idx = Array{Int}(undef, n);
    @inbounds for j ∈ 1:n
        s, a, r, s′, d = minibatch[j];
        S[:, j] .= s;
        Sp[:, j] .= s′;
        rs[j] = r;
        dones[j] = d ? 1.0f0 : 0.0f0;
        a_idx[j] = action_index[a];
    end

    # bootstrap target from the (frozen) target network
    Qp = T(Sp);                                   # |𝒜| × n
    maxQp = vec(maximum(Qp; dims = 1));           # n
    y = rs .+ γ .* (1.0f0 .- dones) .* maxQp;     # n

    # gradient step on M, evaluating loss only on the taken action
    grads = Flux.gradient(M) do m
        Q = m(S);                                 # |𝒜| × n
        # gather Q[a_idx[j], j] for j = 1..n
        idx = CartesianIndex.(a_idx, 1:n);
        ŷ = Q[idx];
        loss_fn(ŷ, y);
    end
    Flux.update!(optstate, M, grads[1]);

    return nothing;
end
# PRIVATE METHODS ABOVE HERE ================================================================================= #

# PUBLIC METHODS BELOW HERE ================================================================================== #
"""
    solve(agent::MyQLearningAgentModel, environment::MyRectangularGridWorldModel;
        maxsteps::Int = 100, δ::Float64 = 0.02, worldmodel::Function = _world) -> MyQLearningAgentModel

Simulate the Q-Learning agent in the given environment for a maximum number of steps per starting state.

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

"""
    policy(Q_array::Array{Float64,2}) -> Array{Int,1}

Compute the greedy policy from a Q-value table.

### Arguments
- `Q_array::Array{Float64,2}`: the Q-value table with rows indexed by state and columns by action.

### Returns
- `Array{Int,1}`: the policy mapping each state to the action with the largest Q-value.
"""
function policy(Q_array::Array{Float64,2})::Array{Int64,1}

    (NR, _) = size(Q_array);

    π_array = Array{Int64,1}(undef, NR)
    for s ∈ 1:NR
        π_array[s] = argmax(Q_array[s,:]);
    end

    return π_array;
end

# ----------------------------------------------------------------------------
# Continuous LavaWorld dynamics + DQN training loop.
# ----------------------------------------------------------------------------

"""
    initial_state(context::MyDQNworldContextModel; rng=Random.GLOBAL_RNG) -> Vector{Float32}

Sample a random non-terminal initial state s = (x, y, vx, vy) on the board. We
draw (x, y) uniformly over the interior, reject draws that land inside any
lava disk or the charger disk, and start with zero velocity.
"""
function initial_state(context::MyDQNworldContextModel; rng = Random.GLOBAL_RNG)::Vector{Float32}

    while true
        x = rand(rng) * context.ncols;
        y = rand(rng) * context.nrows;
        if !_is_lava(x, y, context) && !_is_charger(x, y, context)
            return Float32[x, y, 0.0, 0.0];
        end
    end
end

"""
    world(s, a, context) -> (s′, r, done)

One Euler step of the continuous LavaWorld. The state is `s = (x, y, vx, vy)`
and the action `a = (Δvx, Δvy)` is a thrust vector that adds to the velocity.

Dynamics:
* The thrust is perturbed by Gaussian actuator noise: Δv ← Δv + σ · Z.
* Velocity update: v ← (1 − drag) · v + Δv, then clipped componentwise to ±vmax.
* Position update: p ← p + dt · v, then clipped to the board.
* Walls: hitting the boundary clips position and zeroes the corresponding velocity.

Terminal conditions and rewards:
* If s′ lands in any lava disk → r = lava_reward, done = true.
* If s′ lands in the charger disk → r = charger_reward, done = true.
* Otherwise → r = step_cost, done = false.

Returns the next state `s′::Vector{Float32}`, the scalar reward
`r::Float32`, and a `done::Bool` flag used by the DQN target.
"""
function world(s::Vector{Float32}, a::Vector{Float32},
        context::MyDQNworldContextModel)::Tuple{Vector{Float32}, Float32, Bool}

    x, y, vx, vy = s[1], s[2], s[3], s[4];

    # actuator noise + drag
    Δvx = a[1] + context.σ * Float32(rand(context.Z));
    Δvy = a[2] + context.σ * Float32(rand(context.Z));
    vx_new = (1.0f0 - context.drag) * vx + Δvx;
    vy_new = (1.0f0 - context.drag) * vy + Δvy;

    # cap speed
    vx_new = clamp(vx_new, -context.vmax, context.vmax);
    vy_new = clamp(vy_new, -context.vmax, context.vmax);

    # integrate position
    x_new = x + context.dt * vx_new;
    y_new = y + context.dt * vy_new;

    # walls: clip and zero out the offending velocity component
    if x_new < 0.0f0
        x_new = 0.0f0; vx_new = 0.0f0;
    elseif x_new > Float32(context.ncols)
        x_new = Float32(context.ncols); vx_new = 0.0f0;
    end
    if y_new < 0.0f0
        y_new = 0.0f0; vy_new = 0.0f0;
    elseif y_new > Float32(context.nrows)
        y_new = Float32(context.nrows); vy_new = 0.0f0;
    end

    s′ = Float32[x_new, y_new, vx_new, vy_new];

    # terminal check (terminals ignore shaping by convention — the bootstrap is
    # zeroed by the done flag downstream, and shaping a terminal reward muddles
    # the meaning of "lava cost" / "charger reward" for the lab plots).
    if _is_lava(x_new, y_new, context)
        return s′, context.lava_reward, true;
    elseif _is_charger(x_new, y_new, context)
        return s′, context.charger_reward, true;
    end

    # potential-based shaping: F = w * (d_old - d_new), where d is the L2
    # distance from the agent to the charger. Policy-invariant with γ ≈ 1
    # (Ng, Harada, & Russell 1999), but gives the agent a dense gradient
    # signal toward the goal instead of waiting for the sparse terminal reward.
    r = context.step_cost;
    if context.shape_weight > 0.0f0
        cx, cy = context.charger;
        d_old = sqrt((x - cx)^2 + (y - cy)^2);
        d_new = sqrt((x_new - cx)^2 + (y_new - cy)^2);
        r += context.shape_weight * (d_old - d_new);
    end
    return s′, r, false;
end

"""
    learn(agent, worldmodel; context, ...) -> (agent, history)

Train `agent::MyDQNLearningAgentModel` against the supplied `worldmodel(s, a, context)`
function. Implements the DQN algorithm exactly as written in the L16a lecture
notes: ε-greedy exploration with a t^(−1/3) schedule, fixed-size circular replay
buffer, mini-batch gradient steps once a warm-up threshold has been reached,
periodic target-network sync, and a done-flag that zeroes the bootstrap on
terminal transitions.

### Returns
    - agent::MyDQNLearningAgentModel: the input agent, with replay buffer and
      networks updated in place.
    - history::NamedTuple: episode-level diagnostics with fields
        * episode_returns::Vector{Float32}
        * episode_lengths::Vector{Int}
        * terminal_kind::Vector{Symbol}   (:charger, :lava, or :timeout per episode)
"""
function learn(agent::MyDQNLearningAgentModel, worldmodel::Function;
        context::MyDQNworldContextModel,
        maxnumberofsteps::Int = 200,
        numberofepisodes::Int = 200,
        γ::Float32 = 0.99f0,
        α::Float32 = 1.0f-3,
        maxreplaybuffersize::Int = 10_000,
        warmupsize::Int = 1_000,
        trainfreq::Int = 1,
        parameterupdatefreq::Int = 200,
        minibatchsize::Int = 64,
        ε_floor::Float32 = 0.05f0,
        rng = Random.GLOBAL_RNG,
        verbose::Bool = true)

    # unpack agent fields
    actions = agent.actions;
    K = length(actions);
    M = agent.mainnetwork;
    T = agent.targetnetwork;

    # circular replay buffer holding (s, a, r, s′, done) tuples
    replaybuffer = CircularBuffer{Tuple{Vector{Float32}, Vector{Float32}, Float32, Vector{Float32}, Bool}}(maxreplaybuffersize);

    # Adam optimizer on the main network
    optstate = Flux.setup(Flux.Adam(α), M);
    loss_fn(ŷ, y) = Flux.Losses.mse(ŷ, y; agg = mean);

    # episode-level diagnostics
    episode_returns = Float32[];
    episode_lengths = Int[];
    terminal_kind = Symbol[];

    global_step = 0;

    for ep ∈ 1:numberofepisodes
        s = initial_state(context; rng = rng);
        ep_return = 0.0f0;
        ep_length = 0;
        ep_terminal = :timeout;

        for _ ∈ 1:maxnumberofsteps
            global_step += 1;

            # ε-greedy with t^(−1/3) decay (per L16a lecture), floor at ε_floor
            ϵₜ = max(ε_floor, min(1.0f0, (1.0f0 / (Float32(global_step)^(1.0f0/3.0f0))) *
                Float32((log(K * global_step + 1.0))^(1.0/3.0))));

            local action_idx::Int;
            if rand(rng) ≤ ϵₜ
                action_idx = rand(rng, 1:K);
            else
                action_idx = argmax(M(s));
            end
            a = actions[action_idx];

            # step the world
            s′, r, done = worldmodel(s, a, context);
            ep_return += r;
            ep_length += 1;

            # store transition
            push!(replaybuffer, (s, a, r, s′, done));

            # gradient step (only after warm-up, and at trainfreq cadence)
            if length(replaybuffer) ≥ warmupsize && rem(global_step, trainfreq) == 0
                _dqn_update!(M, T, optstate, loss_fn, replaybuffer,
                    minibatchsize, γ, actions);
            end

            # periodic target sync
            if rem(global_step, parameterupdatefreq) == 0
                Flux.loadmodel!(T, Flux.state(M));
            end

            # advance, or terminate the episode
            if done
                ep_terminal = (r ≥ context.charger_reward) ? :charger : :lava;
                break;
            end
            s = s′;
        end

        push!(episode_returns, ep_return);
        push!(episode_lengths, ep_length);
        push!(terminal_kind, ep_terminal);

        if verbose && (ep == 1 || rem(ep, max(1, numberofepisodes ÷ 10)) == 0 || ep == numberofepisodes)
            @info "DQN episode" episode=ep return_=round(ep_return; digits=2) length=ep_length terminal=ep_terminal buffer=length(replaybuffer);
        end
    end

    agent.mainnetwork = M;
    agent.targetnetwork = T;
    agent.replaybuffer = replaybuffer;

    history = (episode_returns = episode_returns,
        episode_lengths = episode_lengths,
        terminal_kind = terminal_kind);

    return agent, history;
end

"""
    greedy_policy_field(agent, context; nx, ny, v=(0,0)) -> (X, Y, A, Q)

Evaluate the trained agent on a regular grid of positions (X, Y) at fixed
velocity `v` and return the greedy action index `A[i,j]` and max Q-value
`Q[i,j]` at each grid point. Used by the visualization cells in the notebook.
"""
function greedy_policy_field(agent::MyDQNLearningAgentModel,
        context::MyDQNworldContextModel; nx::Int = 30, ny::Int = 30,
        v::Tuple{Float32, Float32} = (0.0f0, 0.0f0))

    xs = range(0.0f0, Float32(context.ncols); length = nx);
    ys = range(0.0f0, Float32(context.nrows); length = ny);
    A = Array{Int}(undef, nx, ny);
    Q = Array{Float32}(undef, nx, ny);

    @inbounds for i ∈ 1:nx, j ∈ 1:ny
        s = Float32[xs[i], ys[j], v[1], v[2]];
        q = agent.mainnetwork(s);
        A[i, j] = argmax(q);
        Q[i, j] = maximum(q);
    end
    return collect(xs), collect(ys), A, Q;
end

"""
    save_agent(path, agent, history)

Serialize the trained agent's main and target network parameter states plus
the per-episode `history::NamedTuple` to a JLD2 file at `path`. We do not
serialize the replay buffer (it is large and not needed for inference).
"""
function save_agent(path::AbstractString, agent::MyDQNLearningAgentModel, history::NamedTuple)
    main_state = Flux.state(agent.mainnetwork);
    target_state = Flux.state(agent.targetnetwork);
    JLD2.jldsave(path; main_state = main_state, target_state = target_state, history = history);
    return path;
end

"""
    load_agent!(path, agent) -> (agent, history)

Load main- and target-network parameter states from a JLD2 file written by
`save_agent` into the existing `agent`'s networks (architecture must match).
Returns the updated agent together with the cached `history` NamedTuple.
"""
function load_agent!(path::AbstractString, agent::MyDQNLearningAgentModel)
    data = JLD2.load(path);
    Flux.loadmodel!(agent.mainnetwork, data["main_state"]);
    Flux.loadmodel!(agent.targetnetwork, data["target_state"]);
    return agent, data["history"];
end

"""
    rollout(agent, context; max_steps, greedy=true, rng) -> NamedTuple

Run a single episode using the trained agent. With `greedy=true` (default),
the agent always picks `argmax_a Q(s, a)`; otherwise it follows a small fixed
ε. Returns the trajectory (positions, velocities, actions, rewards) plus the
terminal kind.
"""
function rollout(agent::MyDQNLearningAgentModel, worldmodel::Function;
        context::MyDQNworldContextModel,
        max_steps::Int = 400,
        greedy::Bool = true,
        ε::Float32 = 0.05f0,
        s0::Union{Nothing, Vector{Float32}} = nothing,
        rng = Random.GLOBAL_RNG)

    s = isnothing(s0) ? initial_state(context; rng = rng) : copy(s0);

    xs = Float32[s[1]];
    ys = Float32[s[2]];
    vxs = Float32[s[3]];
    vys = Float32[s[4]];
    aidx = Int[];
    rs = Float32[];
    terminal = :timeout;

    actions = agent.actions;
    K = length(actions);

    for _ ∈ 1:max_steps
        if !greedy && rand(rng) ≤ ε
            ai = rand(rng, 1:K);
        else
            ai = argmax(agent.mainnetwork(s));
        end
        a = actions[ai];

        s′, r, done = worldmodel(s, a, context);
        push!(xs, s′[1]); push!(ys, s′[2]);
        push!(vxs, s′[3]); push!(vys, s′[4]);
        push!(aidx, ai); push!(rs, r);

        if done
            terminal = (r ≥ context.charger_reward) ? :charger : :lava;
            break;
        end
        s = s′;
    end

    return (xs = xs, ys = ys, vxs = vxs, vys = vys,
        actions = aidx, rewards = rs, terminal = terminal,
        total_return = sum(rs));
end
# PUBLIC METHODS ABOVE HERE ================================================================================== #
