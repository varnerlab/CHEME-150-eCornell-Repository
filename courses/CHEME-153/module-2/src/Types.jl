# Abstract supertype for the multiplicative-weights linear programming problems -
abstract type AbstractMWALinearProgramProblem end

"""
    mutable struct MyLinearProgramFeasibilityProblem <: AbstractMWALinearProgramProblem

Holds the data for a linear programming feasibility problem solved with the
multiplicative weights update algorithm. We look for a vector on the simplex
`x ∈ Δ_m = {x ≥ 0 : sum(x) = τ}` that satisfies the packing constraints `A x ≤ b`.

### Fields
- `A::Array{Float64,2}`: constraint matrix (`n × m`) for the constraints `A x ≤ b`.
- `b::Array{Float64,1}`: right-hand side vector (`n`).
- `τ::Float64`: simplex scale, i.e., the sum of the decision variables `sum(x) = τ`.
- `η::Float64`: learning rate. Must satisfy `0 < η ≤ 1/ρ` to keep the weights non-negative.
- `ϵ::Float64`: tolerance. The solver adds a `2ϵ` slack to the constraints.
- `T::Int64`: maximum number of iterations.
- `ρ::Float64`: coefficient bound, `|a_ij| ≤ ρ` for all `i,j`.
"""
mutable struct MyLinearProgramFeasibilityProblem <: AbstractMWALinearProgramProblem

    # data -
    A::Array{Float64,2}     # constraint matrix (n × m)
    b::Array{Float64,1}     # right-hand side vector (n)
    τ::Float64              # simplex scale: sum(x) = τ
    η::Float64              # learning rate
    ϵ::Float64              # tolerance
    T::Int64               # maximum number of iterations
    ρ::Float64             # coefficient bound: |a_ij| ≤ ρ

    # constructor -
    MyLinearProgramFeasibilityProblem() = new();
end

"""
    mutable struct MyLinearProgramOptimizationProblem <: AbstractMWALinearProgramProblem

Holds the data for a linear program solved by wrapping the feasibility solver in a
binary search on the objective. We maximize `c'x` over the set
`{x ∈ Δ_m : A x ≤ b}` by adding the cut `c'x ≥ v` and searching for the largest
feasible value of `v`.

### Fields
- `A::Array{Float64,2}`: constraint matrix (`n × m`) for the constraints `A x ≤ b`.
- `b::Array{Float64,1}`: right-hand side vector (`n`).
- `c::Array{Float64,1}`: objective coefficient vector (`m`). We maximize `c'x`.
- `τ::Float64`: simplex scale, i.e., the sum of the decision variables `sum(x) = τ`.
- `η::Float64`: learning rate passed to each feasibility call.
- `ϵ::Float64`: tolerance passed to each feasibility call.
- `T::Int64`: maximum number of iterations per feasibility call.
- `ρ::Float64`: coefficient bound, computed over both `A` and `c`.
"""
mutable struct MyLinearProgramOptimizationProblem <: AbstractMWALinearProgramProblem

    # data -
    A::Array{Float64,2}     # constraint matrix (n × m)
    b::Array{Float64,1}     # right-hand side vector (n)
    c::Array{Float64,1}     # objective coefficient vector (m)
    τ::Float64              # simplex scale: sum(x) = τ
    η::Float64              # learning rate
    ϵ::Float64              # tolerance
    T::Int64               # maximum number of iterations per feasibility call
    ρ::Float64             # coefficient bound

    # constructor -
    MyLinearProgramOptimizationProblem() = new();
end
