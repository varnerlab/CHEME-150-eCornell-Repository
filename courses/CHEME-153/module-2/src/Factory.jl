# We add methods to the build(...) generic exported by VLDataScienceMachineLearningPackage
# so the notebooks keep the familiar build(Type, data) idiom used across the course.

"""
    build(modeltype::Type{MyLinearProgramFeasibilityProblem}, data::NamedTuple) -> MyLinearProgramFeasibilityProblem

Builds a `MyLinearProgramFeasibilityProblem` instance from the data in a `NamedTuple`.

### Required keys
- `A::Array{Float64,2}`: constraint matrix (`n × m`).
- `b::Array{Float64,1}`: right-hand side vector (`n`).

### Optional keys (defaults shown)
- `τ::Float64 = 1.0`: simplex scale `sum(x) = τ`.
- `ρ::Float64 = maximum(abs.(A))`: coefficient bound `|a_ij| ≤ ρ`.
- `ϵ::Float64 = 1e-2`: tolerance.
- `T::Int64 = ceil(log(m)/ϵ^2)`: maximum number of iterations (`m` is the number of variables).
- `η::Float64 = 1/(2ρ)`: learning rate. The default sits inside `(0, 1/ρ]`.
"""
function VLDataScienceMachineLearningPackage.build(modeltype::Type{MyLinearProgramFeasibilityProblem},
    data::NamedTuple)::MyLinearProgramFeasibilityProblem

    # initialize -
    model = modeltype();
    A = data.A;
    b = data.b;
    m = size(A, 2); # number of decision variables (experts)

    # fill in optional values (or use the defaults) -
    τ = haskey(data, :τ) ? data.τ : 1.0;
    ρ = haskey(data, :ρ) ? data.ρ : maximum(abs.(A));
    ϵ = haskey(data, :ϵ) ? data.ϵ : 1e-2;
    T = haskey(data, :T) ? data.T : Int(ceil(log(m) / ϵ^2));
    η = haskey(data, :η) ? data.η : 1.0 / (2.0 * ρ);

    # set the fields -
    model.A = A;
    model.b = b;
    model.τ = τ;
    model.η = η;
    model.ϵ = ϵ;
    model.T = T;
    model.ρ = ρ;

    # return -
    return model;
end

"""
    build(modeltype::Type{MyLinearProgramOptimizationProblem}, data::NamedTuple) -> MyLinearProgramOptimizationProblem

Builds a `MyLinearProgramOptimizationProblem` instance from the data in a `NamedTuple`.

### Required keys
- `A::Array{Float64,2}`: constraint matrix (`n × m`).
- `b::Array{Float64,1}`: right-hand side vector (`n`).
- `c::Array{Float64,1}`: objective coefficient vector (`m`). We maximize `c'x`.

### Optional keys (defaults shown)
- `τ::Float64 = 1.0`: simplex scale `sum(x) = τ`.
- `ρ::Float64 = max(maximum(abs.(A)), maximum(abs.(c)))`: coefficient bound, taken over `A` and `c`.
- `ϵ::Float64 = 1e-2`: tolerance per feasibility call.
- `T::Int64 = ceil(log(m)/ϵ^2)`: maximum iterations per feasibility call.
- `η::Float64 = 1/(2ρ)`: learning rate.
"""
function VLDataScienceMachineLearningPackage.build(modeltype::Type{MyLinearProgramOptimizationProblem},
    data::NamedTuple)::MyLinearProgramOptimizationProblem

    # initialize -
    model = modeltype();
    A = data.A;
    b = data.b;
    c = data.c;
    m = size(A, 2); # number of decision variables (experts)

    # the objective row -c enters the augmented problem, so ρ must bound c as well -
    ρ = haskey(data, :ρ) ? data.ρ : max(maximum(abs.(A)), maximum(abs.(c)));
    τ = haskey(data, :τ) ? data.τ : 1.0;
    ϵ = haskey(data, :ϵ) ? data.ϵ : 1e-2;
    T = haskey(data, :T) ? data.T : Int(ceil(log(m) / ϵ^2));
    η = haskey(data, :η) ? data.η : 1.0 / (2.0 * ρ);

    # set the fields -
    model.A = A;
    model.b = b;
    model.c = c;
    model.τ = τ;
    model.η = η;
    model.ϵ = ϵ;
    model.T = T;
    model.ρ = ρ;

    # return -
    return model;
end
