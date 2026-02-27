# setup paths -
const _ROOT = pwd();
const _PATH_TO_SRC = joinpath(_ROOT, "src");

# check do we have a Manifest.toml file?
using Pkg;
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.add(path="https://github.com/varnerlab/VLDataScienceMachineLearningPackage.jl.git")
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load external packages -
using VLDataScienceMachineLearningPackage
using StatsPlots
using Plots
using Colors
using Statistics
using LinearAlgebra
using Distances
using PrettyTables
using Random
using KernelFunctions
using DataFrames

# set the random seed for reproducibility
Random.seed!(1234);