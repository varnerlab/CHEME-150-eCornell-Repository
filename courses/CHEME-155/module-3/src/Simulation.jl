"""
    generate_thrombin_dataset(tf_values::Vector{Float64};
        tspan::Tuple{Float64,Float64} = (0.0, 1200.0), saveat::Float64 = 5.0) -> Tuple

Generate a dataset of thrombin generation curves by simulating the Hockin-Mann 2002 coagulation
model at each tissue factor (TF) concentration in `tf_values`.

### Arguments
- `tf_values::Vector{Float64}`: vector of TF concentrations in molar (M).
- `tspan::Tuple{Float64,Float64}`: simulation time span in seconds (default: 0 to 1200).
- `saveat::Float64`: time step for saving solution points in seconds (default: 5.0).

### Returns
- `Tuple{Vector{Float64}, Matrix{Float64}, Vector{Float64}}`: tuple of (time_vector, thrombin_matrix, tf_values)
   where `thrombin_matrix` has rows as time points and columns as different TF concentrations.
   Thrombin values are in nanomolar (nM).
"""
function generate_thrombin_dataset(tf_values::Vector{Float64};
    tspan::Tuple{Float64,Float64} = (0.0, 1200.0), saveat::Float64 = 5.0)::Tuple{Vector{Float64}, Matrix{Float64}, Vector{Float64}}

    # initialize -
    n_curves = length(tf_values);
    sol_first = HockinMannModel.simulate(HockinMann2002; TF_concentration=tf_values[1], tspan=tspan, saveat=saveat);
    time_vector = sol_first.t;
    n_timepoints = length(time_vector);
    thrombin_matrix = zeros(Float64, n_timepoints, n_curves);

    # simulate each TF concentration -
    for (j, tf) in enumerate(tf_values)
        sol = HockinMannModel.simulate(HockinMann2002; TF_concentration=tf, tspan=tspan, saveat=saveat);
        thrombin_matrix[:, j] = HockinMannModel.total_thrombin(HockinMann2002, sol) .* 1e9; # convert M to nM
    end

    return (time_vector, thrombin_matrix, tf_values)
end

"""
    generate_hemophilia_dataset(fviii_percent_values::Vector{Float64};
        TF::Float64 = 5e-12, tspan::Tuple{Float64,Float64} = (0.0, 1200.0),
        saveat::Float64 = 5.0) -> Tuple

Generate a dataset of thrombin generation curves by simulating the Hockin-Mann 2002 coagulation
model at each Factor VIII level in `fviii_percent_values` (as percent of nominal).

### Arguments
- `fviii_percent_values::Vector{Float64}`: vector of FVIII levels in percent nominal (0 to 100).
- `TF::Float64`: tissue factor concentration in molar (default: 5 pM).
- `tspan::Tuple{Float64,Float64}`: simulation time span in seconds (default: 0 to 1200).
- `saveat::Float64`: time step for saving solution points in seconds (default: 5.0).

### Returns
- `Tuple{Vector{Float64}, Matrix{Float64}, Vector{Float64}}`: tuple of (time_vector, thrombin_matrix, fviii_percent_values)
   where `thrombin_matrix` has rows as time points and columns as different FVIII levels.
   Thrombin values are in nanomolar (nM).
"""
function generate_hemophilia_dataset(fviii_percent_values::Vector{Float64};
    TF::Float64 = 5e-12, tspan::Tuple{Float64,Float64} = (0.0, 1200.0),
    saveat::Float64 = 5.0)::Tuple{Vector{Float64}, Matrix{Float64}, Vector{Float64}}

    # initialize -
    n_curves = length(fviii_percent_values);
    fviii_molar_first = HockinMannModel.percent_nominal_to_molar(fviii_percent_values[1], :VIII);
    u0_first = HockinMannModel.patient_initial_conditions(HockinMann2002; TF=TF, VIII=fviii_molar_first);
    sol_first = HockinMannModel.simulate(HockinMann2002; u0=u0_first, tspan=tspan, saveat=saveat);
    time_vector = sol_first.t;
    n_timepoints = length(time_vector);
    thrombin_matrix = zeros(Float64, n_timepoints, n_curves);

    # simulate each FVIII level -
    for (j, fviii_pct) in enumerate(fviii_percent_values)
        fviii_molar = HockinMannModel.percent_nominal_to_molar(fviii_pct, :VIII);
        u0 = HockinMannModel.patient_initial_conditions(HockinMann2002; TF=TF, VIII=fviii_molar);
        sol = HockinMannModel.simulate(HockinMann2002; u0=u0, tspan=tspan, saveat=saveat);
        thrombin_matrix[:, j] = HockinMannModel.total_thrombin(HockinMann2002, sol) .* 1e9; # convert M to nM
    end

    return (time_vector, thrombin_matrix, fviii_percent_values)
end

"""
    normalize_minmax(data::Matrix{Float64}) -> Tuple{Matrix{Float64}, Float64, Float64}

Apply min-max normalization to scale all values in `data` to the interval [0, 1].

### Arguments
- `data::Matrix{Float64}`: matrix of values to normalize.

### Returns
- `Tuple{Matrix{Float64}, Float64, Float64}`: tuple of (normalized_data, data_min, data_max).
"""
function normalize_minmax(data::Matrix{Float64})::Tuple{Matrix{Float64}, Float64, Float64}
    data_min = minimum(data);
    data_max = maximum(data);
    normalized = (data .- data_min) ./ (data_max - data_min);
    return (normalized, data_min, data_max)
end

"""
    normalize_minmax_percurve(data::Matrix{Float64}) -> Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}}

Apply min-max normalization independently to each column (curve) of `data`, scaling each
column to the interval [0, 1] using that column's own minimum and maximum.

Normalizing per-curve rather than globally ensures that:
- Test-curve statistics do not pollute the normalization applied to training data.
- The peak of every curve is mapped to 1.0, giving the peak region proportional weight
  in the MSE loss regardless of its absolute thrombin level.

### Arguments
- `data::Matrix{Float64}`: matrix of shape `(T, n_curves)` where each column is one curve.

### Returns
- `Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}}`: tuple of
  (normalized_data, curve_mins, curve_maxs) where `curve_mins` and `curve_maxs` are
  vectors of length `n_curves` holding the per-column normalization bounds.
"""
function normalize_minmax_percurve(data::Matrix{Float64})::Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}}
    n_curves = size(data, 2);
    normalized = similar(data);
    curve_mins = Vector{Float64}(undef, n_curves);
    curve_maxs = Vector{Float64}(undef, n_curves);
    for j in 1:n_curves
        col_min = minimum(data[:, j]);
        col_max = maximum(data[:, j]);
        curve_mins[j] = col_min;
        curve_maxs[j] = col_max;
        normalized[:, j] = (data[:, j] .- col_min) ./ (col_max - col_min);
    end
    return (normalized, curve_mins, curve_maxs)
end

"""
    denormalize_minmax(data::Matrix{Float64}, data_min::Float64, data_max::Float64) -> Matrix{Float64}

Reverse min-max normalization to recover original-scale values.

### Arguments
- `data::Matrix{Float64}`: normalized matrix with values in [0, 1].
- `data_min::Float64`: minimum value from the original data.
- `data_max::Float64`: maximum value from the original data.

### Returns
- `Matrix{Float64}`: data in original scale.
"""
function denormalize_minmax(data::Matrix{Float64}, data_min::Float64, data_max::Float64)::Matrix{Float64}
    return data .* (data_max - data_min) .+ data_min
end

"""
    denormalize_minmax(data::Vector{Float64}, data_min::Float64, data_max::Float64) -> Vector{Float64}

Reverse min-max normalization for a vector.

### Arguments
- `data::Vector{Float64}`: normalized vector with values in [0, 1].
- `data_min::Float64`: minimum value from the original data.
- `data_max::Float64`: maximum value from the original data.

### Returns
- `Vector{Float64}`: vector in original scale.
"""
function denormalize_minmax(data::Vector{Float64}, data_min::Float64, data_max::Float64)::Vector{Float64}
    return data .* (data_max - data_min) .+ data_min
end
