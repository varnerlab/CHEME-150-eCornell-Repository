# setup paths -
const _ROOT = pwd();
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_DATA = joinpath(_ROOT, "data");

# check do we have a Manifest.toml file?
using Pkg;
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.add(path="https://github.com/varnerlab/VLDataScienceMachineLearningPackage.jl.git")
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load external packages
using VLDataScienceMachineLearningPackage
using CSV
using JLD2
using FileIO
using DataFrames
using Statistics
using LinearAlgebra
using KernelFunctions
using NNlib
using HTTP
using JSON
using ColorVectorSpace
using PrettyTables
using ImageMagick
using Images
using ImageIO
using Plots
using Colors

# include my codes
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Factory.jl"));
include(joinpath(_PATH_TO_SRC, "Network.jl"));
include(joinpath(_PATH_TO_SRC, "Handler.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));
include(joinpath(_PATH_TO_SRC, "Eigendecomposition.jl"));