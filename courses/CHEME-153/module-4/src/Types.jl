# Abstract types -
abstract type AbstractWorldModel end
abstract type AbstractOnlineLearningModel end

# Tabular MDP environments with intrinsic, known dynamics. Any subtype provides
# `world(model, s, a) -> (s′, r)`, `isterminal(model, s)`, and `_nactions(model)`; the
# generic `solve(...)` (value iteration) and `solve(agent, ...)` (Q-learning) work on all of them.
abstract type AbstractTabularMDP <: AbstractWorldModel end

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
    mutable struct MyAdaptiveDosingModel <: AbstractTabularMDP

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
mutable struct MyAdaptiveDosingModel <: AbstractTabularMDP

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

"""
    mutable struct MyFedBatchBioreactorModel <: AbstractTabularMDP

A mutable type for the fed-batch CHO bioreactor environment. The culture state is a point
`(X, L, V)` on a cubic grid: `X` is the (normalized) viable biomass, `L` is the (normalized)
lactate by-product, and `V` is the (normalized) reactor volume. Each cycle the agent picks a
feed rate `f ∈ feeds`; the feed adds volume `ΔV = vfeed·f` and sets the dilution rate
`D = ΔV/V`. Biomass grows by Monod kinetics in the feed with lactate inhibition (minus first-
order death and dilution); lactate is produced by overflow (feed × biomass) and removed by
consumption and dilution; volume rises with feed. The batch ends when the reactor fills
(`V ≥ Vmax`, **harvest**), and the reward is the product harvested then (`qP·X·V`); the culture
crashes (absorbing, penalized) if lactate reaches `Lcrash` before harvest.

### Fields
- `nlevels::Int`: number of grid levels per axis (`X`, `L`, `V` each take values `0, step, …, 1`)
- `step::Float64`: grid spacing
- `coordinates::Dict{Int,Tuple{Int,Int,Int}}`: state index → `(iX, iL, iV)` grid indices
- `states::Dict{Tuple{Int,Int,Int},Int}`: `(iX, iL, iV)` grid indices → state index
- `feeds::Array{Float64,1}`: feed-rate fraction for each action
- `mumax::Float64`: maximum specific growth rate
- `Kf::Float64`: feed half-saturation constant (Monod in feed)
- `KIL::Float64`: lactate inhibition constant
- `kd::Float64`: first-order cell death rate
- `yL::Float64`: lactate yield per unit feed × biomass (overflow)
- `kL::Float64`: lactate consumption/clearance fraction per cycle
- `qP::Float64`: specific productivity (harvest reward is `qP·X·V`)
- `vfeed::Float64`: reactor volume added per cycle at full feed (`ΔV = vfeed·f`)
- `V0::Float64`: initial (inoculation) volume
- `Vmax::Float64`: harvest volume (absorbing) when `V ≥ Vmax`
- `Lcrash::Float64`: culture crash (absorbing) if `L ≥ Lcrash`
- `Rcrash::Float64`: terminal penalty for a crashed batch
"""
mutable struct MyFedBatchBioreactorModel <: AbstractTabularMDP

    # state grid -
    nlevels::Int
    step::Float64
    coordinates::Dict{Int,Tuple{Int,Int,Int}}
    states::Dict{Tuple{Int,Int,Int},Int}

    # actions -
    feeds::Array{Float64,1}

    # dynamics parameters -
    mumax::Float64
    Kf::Float64
    KIL::Float64
    kd::Float64
    yL::Float64
    kL::Float64
    qP::Float64

    # volume / fed-batch parameters -
    vfeed::Float64
    V0::Float64
    Vmax::Float64

    # absorbing threshold and penalty -
    Lcrash::Float64
    Rcrash::Float64

    # constructor -
    MyFedBatchBioreactorModel() = new();
end
