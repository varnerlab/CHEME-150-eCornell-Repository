# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "src");

# load external packages -
using Pkg
Pkg.activate("."); # activate local Project.toml environment
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# using statements -
using OrdinaryDiffEq
using Statistics
using Random
using JLD2
using LinearAlgebra
using Plots
using NNlib
using Flux
using PrettyTables
using DataFrames
using IJulia

# load source files -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Parameters.jl"));
include(joinpath(_PATH_TO_SRC, "Kinetics.jl"));
include(joinpath(_PATH_TO_SRC, "MassBalances.jl"));
include(joinpath(_PATH_TO_SRC, "Simulation.jl"));
include(joinpath(_PATH_TO_SRC, "LSTM.jl"));
