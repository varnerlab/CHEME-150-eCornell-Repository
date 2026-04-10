"""
    build_default_parameters(; F_max, Glc_min, Glc_max, kwargs...) -> MyFedBatchCHOParameters

Construct a `MyFedBatchCHOParameters` instance with default mid-range kinetic values
from the Hockin-Mann CHO fed-batch model literature. The feed policy parameters
(`F_max`, `Glc_min`, `Glc_max`) are required keyword arguments because they serve
as conditioning inputs to the LSTM.

### Arguments (keyword)
- `F_max::Float64`: maximum feed rate (L/h). Required.
- `Glc_min::Float64`: glucose threshold to turn feed ON (mM). Required.
- `Glc_max::Float64`: glucose threshold to turn feed OFF (mM). Required.
- All other keyword arguments override default kinetic parameter values.

### Returns
- `MyFedBatchCHOParameters`: initialized parameter struct with `feed_on = 0.0`.
"""
function build_default_parameters(; F_max::Float64, Glc_min::Float64, Glc_max::Float64,
    mu_max::Float64 = 0.029,        # Xing et al. (2010) Table 2: 0.029 1/h
    K_glc::Float64 = 0.10,          # Xing et al. (2010) Table 3: ~0.084 mM (rounded up slightly)
    K_gln::Float64 = 0.05,          # Xing et al. (2010) Table 3: ~0.047 mM
    K_I_lac::Float64 = 43.0,        # Xing et al. (2010) Table 3: 43.0 mM
    K_I_amm::Float64 = 6.5,         # Xing et al. (2010) Table 3: 6.51 mM
    k_d::Float64 = 0.016,           # Xing et al. (2010) Table 2: 0.016 1/h (maximum death rate)
    KD_lac::Float64 = 45.8,         # Xing et al. (2010) Table 3: 45.8 mM (lactate half-sat. for death)
    KD_amm::Float64 = 6.5,          # Xing et al. (2010) Table 3: 6.51 mM (ammonia half-sat. for death)
    alpha_P::Float64 = 100.0,       # growth-associated productivity (mg/gDW); at mu_max=0.029: 100*0.029=2.9 mg/gDW/h
    beta_P::Float64 = 5.0,          # non-growth-associated productivity (mg/gDW/h); Luedeking-Piret beta
    Y_X_glc::Float64 = 0.070,       # biomass yield on glucose (gDW/mmol); Xing et al. (2010) Table 2 basis
    Y_X_gln::Float64 = 0.210,       # biomass yield on glutamine (gDW/mmol)
    Y_P_glc::Float64 = 16.7,        # product yield on glucose (mg/mmol)
    Y_P_gln::Float64 = 33.3,        # product yield on glutamine (mg/mmol)
    Y_lac_glc::Float64 = 1.23,      # Xing et al. (2010) Table 2: 1.23 mmol/mmol
    Y_amm_gln::Float64 = 0.67,      # Xing et al. (2010) Table 2: 0.67 mmol/mmol
    S_glc_f::Float64 = 500.0,       # feed glucose concentration (mM)
    S_gln_f::Float64 = 167.0,       # feed glutamine concentration (mM); matched to 3:1 glc:gln consumption stoichiometry
    )::MyFedBatchCHOParameters

    return MyFedBatchCHOParameters(
        mu_max, K_glc, K_gln, K_I_lac, K_I_amm, k_d,
        KD_lac, KD_amm,
        alpha_P, beta_P,
        Y_X_glc, Y_X_gln, Y_P_glc, Y_P_gln, Y_lac_glc, Y_amm_gln,
        S_glc_f, S_gln_f,
        F_max, Glc_min, Glc_max,
        0.0  # feed starts OFF
    );
end
