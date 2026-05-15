"""
    build(modeltype::Type{MyRectangularGridWorldModel}, data::NamedTuple) -> MyRectangularGridWorldModel

Builds a `MyRectangularGridWorldModel` from data in a `NamedTuple`.

### Arguments
- `modeltype::Type{MyRectangularGridWorldModel}`: the model type to build
- `data::NamedTuple`: the data to use to build the model

The `data` `NamedTuple` must contain the following keys:
- `nrows::Int`: number of rows in the grid
- `ncols::Int`: number of columns in the grid
- `rewards::Dict{Tuple{Int,Int},Float64}`: dictionary of state to reward mapping
- `defaultreward::Float64`: default reward value (optional, default -1.0)

### Returns
- `MyRectangularGridWorldModel`: a populated rectangular grid world model
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

### Arguments
- `modeltype::Type{MyQLearningAgentModel}`: the model type to build
- `data::NamedTuple`: the data to use to build the model

The `data` `NamedTuple` must contain the following keys:
- `states::Array{Int,1}`: the state space
- `actions::Array{Int,1}`: the action space
- `α::Float64`: the learning rate
- `γ::Float64`: the discount factor
- `Q::Array{Float64,2}`: the initial Q-value table

### Returns
- `MyQLearningAgentModel`: a populated Q-learning agent model
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
    build(modeltype::Type{MyDQNworldContextModel}, data::NamedTuple) -> MyDQNworldContextModel

Construct a `MyDQNworldContextModel` from a NamedTuple of parameters. All fields
of the type must be supplied by the caller; no defaults are filled in here.
"""
function build(modeltype::Type{MyDQNworldContextModel}, data::NamedTuple)::MyDQNworldContextModel

    model = modeltype();
    model.nrows = data.nrows;
    model.ncols = data.ncols;
    model.lava = data.lava;
    model.charger = data.charger;
    model.lava_radius = data.lava_radius;
    model.charger_radius = data.charger_radius;
    model.thrust = data.thrust;
    model.drag = data.drag;
    model.dt = data.dt;
    model.step_cost = data.step_cost;
    model.lava_reward = data.lava_reward;
    model.charger_reward = data.charger_reward;
    model.σ = data.σ;
    model.vmax = data.vmax;
    model.Z = data.Z;
    model.shape_weight = data.shape_weight;

    return model;
end

"""
    build(modeltype::Type{MyDQNLearningAgentModel}, data::NamedTuple) -> MyDQNLearningAgentModel

Construct a `MyDQNLearningAgentModel`. The action set is built here from a
discrete thrust dictionary so the learn loop only sees indices: action 1 is
"+x thrust", action 2 is "−x thrust", action 3 is "+y thrust", action 4 is
"−y thrust", and action 5 is "coast" (no thrust). The thrust magnitude is
`data.thrust`. State dimension is fixed at 4 (x, y, vx, vy).

### Required NamedTuple fields
    - mainnetwork::Chain
    - targetnetwork::Chain
    - thrust::Float32  (magnitude of the velocity change per directed thrust)
    - number_of_inputs::Int64  (state dimension; 4 for (x, y, vx, vy))
"""
function build(modeltype::Type{MyDQNLearningAgentModel}, data::NamedTuple)::MyDQNLearningAgentModel

    model = modeltype();

    Δv = data.thrust;
    actions = Dict{Int64, Vector{Float32}}(
        1 => Float32[ Δv,  0.0],
        2 => Float32[-Δv,  0.0],
        3 => Float32[ 0.0,  Δv],
        4 => Float32[ 0.0, -Δv],
        5 => Float32[ 0.0,  0.0],
    );

    model.mainnetwork = data.mainnetwork;
    model.targetnetwork = data.targetnetwork;
    model.actions = actions;
    model.number_of_actions = length(actions);
    model.number_of_inputs = data.number_of_inputs;

    return model;
end
