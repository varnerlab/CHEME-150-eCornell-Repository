# Abstract types -
abstract type AbstractWorldModel end
abstract type AbstractOnlineLearningModel end
abstract type AbstractNeuralNetwork end
abstract type AbstractLearningModel end
abstract type AbstractWorldContextModel end

"""
    mutable struct MyRectangularGridWorldModel <: AbstractWorldModel

A mutable struct that defines a rectangular grid world model.

### Fields
- `number_of_rows::Int`: number of rows in the grid
- `number_of_cols::Int`: number of columns in the grid
- `coordinates::Dict{Int,Tuple{Int,Int}}`: dictionary of state to coordinate mapping
- `states::Dict{Tuple{Int,Int},Int}`: dictionary of coordinate to state mapping
- `moves::Dict{Int,Tuple{Int,Int}}`: dictionary of action to move mapping
- `rewards::Dict{Int,Float64}`: dictionary of state to reward mapping
"""
mutable struct MyRectangularGridWorldModel <: AbstractWorldModel

    # data -
    number_of_rows::Int
    number_of_cols::Int
    coordinates::Dict{Int,Tuple{Int,Int}}
    states::Dict{Tuple{Int,Int},Int}
    moves::Dict{Int,Tuple{Int,Int}}
    rewards::Dict{Int,Float64}

    # constructor -
    MyRectangularGridWorldModel() = new();
end

"""
    mutable struct MyQLearningAgentModel <: AbstractOnlineLearningModel

A mutable type for the Q-Learning Agent model.

### Fields
- `states::Array{Int,1}`: array of states
- `actions::Array{Int,1}`: array of actions
- `γ::Float64`: discount factor
- `α::Float64`: learning rate
- `Q::Array{Float64,2}`: Q-value table
"""
mutable struct MyQLearningAgentModel <: AbstractOnlineLearningModel

    # data -
    states::Array{Int,1}
    actions::Array{Int,1}
    γ::Float64
    α::Float64
    Q::Array{Float64,2}

    # constructor -
    MyQLearningAgentModel() = new();
end

"""
    struct MyFluxNeuralNetworkModel <: AbstractNeuralNetwork

A thin wrapper around a Flux `Chain` so that `Flux.@layer` can register the
trainable submodules by name. Used to build both the main and target Q-networks.

### Fields
    - chain::Chain: the underlying Flux model.
"""
struct MyFluxNeuralNetworkModel <: AbstractNeuralNetwork
    chain::Chain
end

"""
    mutable struct MyDQNLearningAgentModel <: AbstractLearningModel

DQN agent state shared across the training loop.

### Fields
    - actions::Dict{Int64, Vector{Float32}}: action index → 4-d thrust vector applied to (vx, vy)
    - mainnetwork::Chain: main Q-network (the one we update at every step)
    - targetnetwork::Chain: delayed target Q-network used to compute bootstrap targets
    - number_of_actions::Int64: |𝒜|, equals length(actions)
    - number_of_inputs::Int64: dimension of the state vector consumed by the networks
    - replaybuffer::CircularBuffer{...}: stored transitions (s, a, r, s′, done)
"""
mutable struct MyDQNLearningAgentModel <: AbstractLearningModel

    actions::Dict{Int64, Vector{Float32}}
    mainnetwork::Chain
    targetnetwork::Chain
    number_of_actions::Int64
    number_of_inputs::Int64
    replaybuffer::CircularBuffer{Tuple{Vector{Float32}, Vector{Float32}, Float32, Vector{Float32}, Bool}}

    MyDQNLearningAgentModel() = new();
end

"""
    mutable struct MyDQNworldContextModel <: AbstractWorldContextModel

Continuous LavaWorld parameters consumed by the `world(...)` function. Holds
the board geometry, the lava and charger locations, and the dynamics knobs
(thrust gain, drag, step cost, terminal rewards, noise).

### Fields
    - nrows::Int: number of grid rows (board is x ∈ [0, ncols], y ∈ [0, nrows])
    - ncols::Int: number of grid columns
    - lava::Vector{Tuple{Float32,Float32}}: (x, y) centers of lava pits
    - charger::Tuple{Float32,Float32}: (x, y) of the goal/charger
    - lava_radius::Float32: radius of each lava pit (any visit inside terminates with lava penalty)
    - charger_radius::Float32: radius of the charger (visit inside terminates with charger reward)
    - thrust::Float32: thrust gain per action (Δv per step in chosen direction)
    - drag::Float32: linear-drag coefficient applied to velocity at every step (∈ [0, 1])
    - dt::Float32: integration step size
    - step_cost::Float32: per-step reward (negative; encourages reaching the goal quickly)
    - lava_reward::Float32: terminal reward for hitting any lava pit
    - charger_reward::Float32: terminal reward for hitting the charger
    - σ::Float32: actuator noise std added to the thrust each step
    - vmax::Float32: cap on the magnitude of vx, vy
    - Z::Normal{Float32}: standard Normal used to draw actuator noise
    - shape_weight::Float32: gain on potential-based shaping toward the charger
      (0 disables; otherwise the per-step reward gets `w*(d_old - γ*d_new)`,
      which is policy-invariant per Ng et al. 1999 with γ from training)
"""
mutable struct MyDQNworldContextModel <: AbstractWorldContextModel

    nrows::Int
    ncols::Int
    lava::Vector{Tuple{Float32, Float32}}
    charger::Tuple{Float32, Float32}
    lava_radius::Float32
    charger_radius::Float32
    thrust::Float32
    drag::Float32
    dt::Float32
    step_cost::Float32
    lava_reward::Float32
    charger_reward::Float32
    σ::Float32
    vmax::Float32
    Z::Normal{Float32}
    shape_weight::Float32

    MyDQNworldContextModel() = new();
end
