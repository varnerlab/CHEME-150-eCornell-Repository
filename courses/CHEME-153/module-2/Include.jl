# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "src");

# check: do we need to download any packages?
using Pkg
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.add(path="https://github.com/varnerlab/VLDataScienceMachineLearningPackage.jl.git")
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# render plots without a display (headless-safe for batch execution) -
ENV["GKSwstype"] = "100";

# load the required packages -
using VLDataScienceMachineLearningPackage
using Distributions
using Plots
using StatsPlots
using Colors
using LinearAlgebra
using Statistics
using DataFrames
using PrettyTables
using Random

# load our local codes for this module (LP via multiplicative weights) -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Factory.jl"));
include(joinpath(_PATH_TO_SRC, "Solve.jl"));

Random.seed!(1234); # set the random seed for reproducibility
