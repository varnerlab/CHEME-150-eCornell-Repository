


# --- PUBLIC METHODS BELOW HERE -------------------------------------------------------------------------------- #
function build(base::String, model::MyBiggModelsEndpointModel; apiversion::String = "v2")::String
    
    # TODO: implement this function, and remove the throw statement
    # throw(ArgumentError("build(base::String, model::MyWeatherGridPointEndpointModel) not implemented yet!"));

    # build the URL string -
    url_string = "$(base)/api/$(apiversion)/models";

    # return the URL string -
    return url_string;
end

function build(base::String, model::MyBiggModelsDownloadModelEndpointModel; apiversion::String = "v2")::String

    # get data -
    bigg_id = model.bigg_id;

    # build the URL string -
    url_string = "$(base)/api/$(apiversion)/models/$(bigg_id)/download";

    # return the URL string -
    return url_string;
end
# --- PUBLIC METHODS ABOVE HERE -------------------------------------------------------------------------------- #