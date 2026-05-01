# setup paths -
const _ROOT = @__DIR__;
const _PATH_TO_DATA = joinpath(_ROOT, "data");
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_IMAGES = joinpath(_ROOT, "images");

# if we are missing any packages, install them -
using Pkg;
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load external packages -
using Flux
using Flux: DataLoader
using Flux: logitcrossentropy, onecold, onehotbatch
using GraphNeuralNetworks
using LinearAlgebra
using MLDatasets
using Random
using Statistics
using JLD2
using PrettyTables
using DataFrames
using Plots
using GraphRecipes
import GraphMakie
import CairoMakie as Makie
using Graphs

# setup random seed
Random.seed!(1);

# load my codes -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Factory.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));