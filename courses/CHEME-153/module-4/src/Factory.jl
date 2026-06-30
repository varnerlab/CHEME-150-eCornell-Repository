"""
    build(modeltype::Type{MyRectangularGridWorldModel}, data::NamedTuple) -> MyRectangularGridWorldModel

Builds a `MyRectangularGridWorldModel` from data in a `NamedTuple`.

The `data` `NamedTuple` must contain the following keys:
- `nrows::Int`: number of rows in the grid
- `ncols::Int`: number of columns in the grid
- `rewards::Dict{Tuple{Int,Int},Float64}`: dictionary of coordinate to reward mapping
- `defaultreward::Float64`: default reward value (optional, default -1.0)
"""
function build(modeltype::Type{MyRectangularGridWorldModel}, data::NamedTuple)::MyRectangularGridWorldModel

    # initialize an empty model -
    model = modeltype()

    # get the data -
    nrows = data[:nrows]
    ncols = data[:ncols]
    rewards = data[:rewards]
    defaultreward = haskey(data, :defaultreward) == false ? -1.0 : data[:defaultreward]

    # setup storage
    rewards_dict = Dict{Int,Float64}()
    coordinates = Dict{Int,Tuple{Int,Int}}()
    states = Dict{Tuple{Int,Int},Int}()
    moves = Dict{Int,Tuple{Int,Int}}()

    # build the coordinate, state, and reward maps
    position_index = 1;
    for i ∈ 1:nrows
        for j ∈ 1:ncols

            # capture this coordinate
            coordinate = (i,j);

            # set -
            coordinates[position_index] = coordinate;
            states[coordinate] = position_index;

            if (haskey(rewards, coordinate) == true)
                rewards_dict[position_index] = rewards[coordinate];
            else
                rewards_dict[position_index] = defaultreward;
            end

            # update position_index -
            position_index += 1;
        end
    end

    # setup the moves dictionary -
    moves[1] = (-1,0)   # a = 1 up
    moves[2] = (1,0)    # a = 2 down
    moves[3] = (0,-1)   # a = 3 left
    moves[4] = (0,1)    # a = 4 right

    # add items to the model -
    model.rewards = rewards_dict
    model.coordinates = coordinates
    model.states = states;
    model.moves = moves;
    model.number_of_rows = nrows
    model.number_of_cols = ncols

    # return -
    return model
end

"""
    build(modeltype::Type{MyQLearningAgentModel}, data::NamedTuple) -> MyQLearningAgentModel

Builds a `MyQLearningAgentModel` from data in a `NamedTuple`.

The `data` `NamedTuple` must contain the following keys:
- `states::Array{Int,1}`: the state space
- `actions::Array{Int,1}`: the action space
- `α::Float64`: the learning rate
- `γ::Float64`: the discount factor
- `Q::Array{Float64,2}`: the initial Q-value table
"""
function build(modeltype::Type{MyQLearningAgentModel}, data::NamedTuple)::MyQLearningAgentModel

    # initialize -
    model = modeltype();

    # if we have options, populate the fields from the NamedTuple -
    if (isempty(data) == false)

        for key ∈ fieldnames(modeltype)

            field_name_string = string(key)

            if (haskey(data, key) == false)
                throw(ArgumentError("NamedTuple is missing: $(field_name_string)"))
            end

            value = data[key]
            setproperty!(model, key, value)
        end
    end

    # return -
    return model
end

"""
    build(modeltype::Type{MyAdaptiveDosingModel}, data::NamedTuple) -> MyAdaptiveDosingModel

Builds a `MyAdaptiveDosingModel` from data in a `NamedTuple`, including the
state-grid index maps `coordinates` and `states`.

The `data` `NamedTuple` must contain the following keys:
- `nlevels::Int`: number of grid levels per axis
- `doses::Array{Float64,1}`: dose magnitude for each action
- `g, k, a, c::Float64`: tumor growth, kill, toxicity loading, and clearance parameters
- `Tcure, Zlethal::Float64`: cured and toxic-death thresholds
- `wT, wZ, cost, Rcure, Rdead::Float64`: reward parameters
"""
function build(modeltype::Type{MyAdaptiveDosingModel}, data::NamedTuple)::MyAdaptiveDosingModel

    # initialize an empty model -
    model = modeltype();

    # grid bookkeeping -
    nlevels = data[:nlevels];
    model.nlevels = nlevels;
    model.step = 1.0/(nlevels - 1);   # levels are 0, step, …, 1

    # build the (iT, iZ) <-> state index maps -
    coordinates = Dict{Int,Tuple{Int,Int}}();
    states = Dict{Tuple{Int,Int},Int}();
    s = 1;
    for iT ∈ 1:nlevels
        for iZ ∈ 1:nlevels
            coordinates[s] = (iT, iZ);
            states[(iT, iZ)] = s;
            s += 1;
        end
    end
    model.coordinates = coordinates;
    model.states = states;

    # actions and parameters -
    model.doses = data[:doses];
    model.g = data[:g];
    model.k = data[:k];
    model.a = data[:a];
    model.c = data[:c];
    model.Tcure = data[:Tcure];
    model.Zlethal = data[:Zlethal];
    model.wT = data[:wT];
    model.wZ = data[:wZ];
    model.cost = data[:cost];
    model.Rcure = data[:Rcure];
    model.Rdead = data[:Rdead];

    # return -
    return model
end

"""
    build(modeltype::Type{MyFedBatchBioreactorModel}, data::NamedTuple) -> MyFedBatchBioreactorModel

Builds a `MyFedBatchBioreactorModel` from data in a `NamedTuple`, including the state-grid
index maps `coordinates` and `states`.

The `data` `NamedTuple` must contain the following keys:
- `nlevels::Int`: number of grid levels per axis
- `feeds::Array{Float64,1}`: feed-rate fraction for each action
- `mumax, Kf, KIL, kd, yL, kL, qP::Float64`: growth, feed, inhibition, death, lactate, and productivity parameters
- `vfeed, V0, Vmax::Float64`: volume added per cycle at full feed, initial volume, and harvest volume
- `Lcrash, Rcrash::Float64`: culture-crash threshold and penalty
"""
function build(modeltype::Type{MyFedBatchBioreactorModel}, data::NamedTuple)::MyFedBatchBioreactorModel

    # initialize an empty model -
    model = modeltype();

    # grid bookkeeping -
    nlevels = data[:nlevels];
    model.nlevels = nlevels;
    model.step = 1.0/(nlevels - 1);   # levels are 0, step, …, 1

    # build the (iX, iL, iV) <-> state index maps -
    coordinates = Dict{Int,Tuple{Int,Int,Int}}();
    states = Dict{Tuple{Int,Int,Int},Int}();
    s = 1;
    for iX ∈ 1:nlevels
        for iL ∈ 1:nlevels
            for iV ∈ 1:nlevels
                coordinates[s] = (iX, iL, iV);
                states[(iX, iL, iV)] = s;
                s += 1;
            end
        end
    end
    model.coordinates = coordinates;
    model.states = states;

    # actions and parameters -
    model.feeds = data[:feeds];
    model.mumax = data[:mumax];
    model.Kf = data[:Kf];
    model.KIL = data[:KIL];
    model.kd = data[:kd];
    model.yL = data[:yL];
    model.kL = data[:kL];
    model.qP = data[:qP];
    model.vfeed = data[:vfeed];
    model.V0 = data[:V0];
    model.Vmax = data[:Vmax];
    model.Lcrash = data[:Lcrash];
    model.Rcrash = data[:Rcrash];

    # return -
    return model
end
