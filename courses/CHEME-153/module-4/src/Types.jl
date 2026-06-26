# Abstract types -
abstract type AbstractWorldModel end
abstract type AbstractOnlineLearningModel end
abstract type MyAbstractContextModel end

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
    mutable struct MyExperimentalDrugCocktailContext <: MyAbstractContextModel

A mutable type for the experimental context of the drug combination design problem.

### Fields
- `K::Int64`: number of drug types
- `m::Int64`: number of concentration levels per drug type
- `γ::Array{Float64,1}`: effectiveness parameters
- `B::Float64`: total budget in USD
- `cost::Dict{Int, Float64}`: maps drug type to cost per mg/kg
- `levels::Dict{Int, NamedTuple}`: maps drug type to its concentration levels in mg/kg
- `W::Float64`: weight of the patient in kg
"""
mutable struct MyExperimentalDrugCocktailContext <: MyAbstractContextModel

    # data -
    K::Int64                      # number of drug types
    m::Int64                      # number of concentration levels per drug type
    γ::Array{Float64,1}           # effectiveness parameters
    B::Float64                    # total budget in USD
    cost::Dict{Int, Float64}      # maps drug type to cost per mg/kg
    levels::Dict{Int, NamedTuple} # maps drug type to concentration levels in mg/kg
    W::Float64                    # weight of the patient in kg

    # constructor -
    MyExperimentalDrugCocktailContext() = new(); # create new *empty* instance
end
