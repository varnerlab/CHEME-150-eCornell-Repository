abstract type AbstractBiggEndpointModel end

struct MyBiggModelsEndpointModel <: AbstractBiggEndpointModel

    # methods -
    MyBiggModelsEndpointModel() = new();
end

mutable struct MyBiggModelsDownloadModelEndpointModel <: AbstractBiggEndpointModel

    # data -
    bigg_id::String

    # methods -
    MyBiggModelsDownloadModelEndpointModel() = new();
end