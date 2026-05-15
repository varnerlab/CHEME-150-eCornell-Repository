# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_FIGS = joinpath(_ROOT, "figs");

!isdir(_PATH_TO_FIGS) && mkpath(_PATH_TO_FIGS);

using Pkg;
Pkg.activate(_ROOT);
# Always reconcile the manifest with Project.toml. `instantiate` is a no-op
# when nothing has changed, but it installs any new entries (e.g., JLD2)
# that were added to Project.toml after the manifest was first written.
Pkg.resolve(); Pkg.instantiate();

# load external packages -
using Statistics
using StatsBase
using LinearAlgebra
using Random
using DataFrames
using PrettyTables
using Plots
using Colors
using Distributions
using DataStructures
using Flux
using NNlib
using JLD2

# set the random seed for reproducibility -
Random.seed!(42);

# load local source files -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Factory.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));
