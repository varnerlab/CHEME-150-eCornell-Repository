# Abstract types -
abstract type AbstractWorldModel end
abstract type AbstractOnlineLearningModel end

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
    mutable struct MyAdaptiveDosingModel <: AbstractWorldModel

A mutable type for the adaptive drug-dosing environment. The patient state is a point
`(T, Z)` on a square grid, where `T` is the (normalized) tumor burden and `Z` is the
(normalized) accumulated toxicity. Each cycle the agent picks a dose `u ∈ doses`; the
state then evolves by logistic tumor growth with dose-dependent log-kill, and first-order
toxicity loading and clearance.

### Fields
- `nlevels::Int`: number of grid levels per axis (`T` and `Z` each take values `0, step, …, 1`)
- `step::Float64`: grid spacing
- `coordinates::Dict{Int,Tuple{Int,Int}}`: state index → `(iT, iZ)` grid indices
- `states::Dict{Tuple{Int,Int},Int}`: `(iT, iZ)` grid indices → state index
- `doses::Array{Float64,1}`: dose magnitude `u` for each action
- `g::Float64`: tumor logistic growth rate
- `k::Float64`: tumor kill efficacy at full dose
- `a::Float64`: toxicity loading per unit dose
- `c::Float64`: toxicity clearance fraction per cycle
- `Tcure::Float64`: cured (absorbing) if `T ≤ Tcure`
- `Zlethal::Float64`: toxic death (absorbing) if `Z ≥ Zlethal`
- `wT::Float64`: step-penalty weight on tumor burden
- `wZ::Float64`: step-penalty weight on toxicity
- `cost::Float64`: step-penalty weight per unit dose
- `Rcure::Float64`: terminal reward for reaching the cured state
- `Rdead::Float64`: terminal reward for reaching the toxic-death state
"""
mutable struct MyAdaptiveDosingModel <: AbstractWorldModel

    # state grid -
    nlevels::Int
    step::Float64
    coordinates::Dict{Int,Tuple{Int,Int}}
    states::Dict{Tuple{Int,Int},Int}

    # actions -
    doses::Array{Float64,1}

    # dynamics parameters -
    g::Float64
    k::Float64
    a::Float64
    c::Float64

    # absorbing thresholds -
    Tcure::Float64
    Zlethal::Float64

    # reward parameters -
    wT::Float64
    wZ::Float64
    cost::Float64
    Rcure::Float64
    Rdead::Float64

    # constructor -
    MyAdaptiveDosingModel() = new();
end
