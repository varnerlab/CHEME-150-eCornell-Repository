"""
    load_ohlc_dataset(path::AbstractString) -> Dict{String, DataFrame}

Load an OHLC jld2 file produced from the Polygon-sourced SP500 dataset.
The file is expected to contain a single top-level key `"dataset"` mapping
to a `Dict{String, DataFrame}` keyed by ticker.
"""
function load_ohlc_dataset(path::AbstractString)
    isfile(path) || error("dataset not found: $(path)")
    d = load(path)
    haskey(d, "dataset") || error("expected key 'dataset' in $(path)")
    return d["dataset"]::Dict{String, DataFrame}
end

"""
    save_model_checkpoint(path::AbstractString, model::MySisoLegSHippoModel)

Save a trained model to `path` as a jld2 file under the key `"model"`.
"""
function save_model_checkpoint(path::AbstractString, model::MySisoLegSHippoModel)
    jldsave(path; model = model)
    return path
end

"""
    load_model_checkpoint(path::AbstractString) -> MySisoLegSHippoModel

Load a trained model checkpoint previously written by `save_model_checkpoint`.
"""
function load_model_checkpoint(path::AbstractString)
    isfile(path) || error("checkpoint not found: $(path)")
    d = load(path)
    haskey(d, "model") || error("expected key 'model' in $(path)")
    return d["model"]::MySisoLegSHippoModel
end

"""
    save_observer_zoo(path::AbstractString, zoo::Dict{String, Matrix{Float64}}; h::Int, Δt::Float64)

Save a dictionary of ticker-specific readout matrices under the key `"zoo"`,
together with the hidden dimension `h` and sampling step `Δt` that the shared
LegS filter was built with. Since every entry of `zoo` shares the same
`(A, B)`, we only persist the per-ticker `C` row vectors.
"""
function save_observer_zoo(path::AbstractString, zoo::Dict{String, Matrix{Float64}}; h::Int, Δt::Float64)
    jldsave(path; zoo = zoo, h = h, Δt = Δt)
    return path
end

"""
    load_observer_zoo(path::AbstractString) -> (zoo, h, Δt)

Load an observer zoo previously written by `save_observer_zoo`. Returns the
per-ticker readout dictionary along with the hidden dimension and sampling
step recorded at save time.
"""
function load_observer_zoo(path::AbstractString)
    isfile(path) || error("zoo not found: $(path)")
    d = load(path)
    haskey(d, "zoo") || error("expected key 'zoo' in $(path)")
    haskey(d, "h")   || error("expected key 'h' in $(path)")
    haskey(d, "Δt")  || error("expected key 'Δt' in $(path)")
    return (d["zoo"]::Dict{String, Matrix{Float64}}, d["h"]::Int, d["Δt"]::Float64)
end

"""
    save_model_checkpoint(path::AbstractString, model::MyMimoLegSHippoModel)

Save a trained MIMO LegS model to `path` as a jld2 file under the key
`"model"`. The function returns `path` so it can be threaded into longer
workflows.
"""
function save_model_checkpoint(path::AbstractString, model::MyMimoLegSHippoModel)
    jldsave(path; model = model)
    return path
end

"""
    load_model_checkpoint(::Type{MyMimoLegSHippoModel}, path::AbstractString) -> MyMimoLegSHippoModel

Load a MIMO LegS model checkpoint previously written by
`save_model_checkpoint`. The file must exist and contain a `"model"` entry with
type `MyMimoLegSHippoModel`.
"""
function load_model_checkpoint(::Type{MyMimoLegSHippoModel}, path::AbstractString)
    isfile(path) || error("checkpoint not found: $(path)")
    d = load(path)
    haskey(d, "model") || error("expected key 'model' in $(path)")
    return d["model"]::MyMimoLegSHippoModel
end
