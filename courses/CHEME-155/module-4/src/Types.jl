"""
    mutable struct MyFedBatchCHOParameters

Holds all kinetic parameters, yield coefficients, feed concentrations, and feed state
for a fed-batch CHO antibody production model.

### Fields
- `mu_max::Float64`: maximum specific growth rate (1/h).
- `K_glc::Float64`: Monod constant for glucose (mM).
- `K_gln::Float64`: Monod constant for glutamine (mM).
- `K_I_lac::Float64`: inhibition constant for lactate (mM).
- `K_I_amm::Float64`: inhibition constant for ammonia (mM).
- `k_d::Float64`: maximum specific death rate (1/h).
- `KD_lac::Float64`: half-saturation constant for lactate in the death rate model (mM).
- `KD_amm::Float64`: half-saturation constant for ammonia in the death rate model (mM).
- `alpha_P::Float64`: growth-associated productivity coefficient (mg/gDW); Luedeking–Piret α.
- `beta_P::Float64`: non-growth-associated productivity coefficient (mg/gDW/h); Luedeking–Piret β.
- `Y_X_glc::Float64`: biomass yield on glucose (gDW/mmol).
- `Y_X_gln::Float64`: biomass yield on glutamine (gDW/mmol).
- `Y_P_glc::Float64`: product yield on glucose (mg/mmol).
- `Y_P_gln::Float64`: product yield on glutamine (mg/mmol).
- `Y_lac_glc::Float64`: lactate yield on glucose (mmol/mmol).
- `Y_amm_gln::Float64`: ammonia yield on glutamine (mmol/mmol).
- `S_glc_f::Float64`: glucose concentration in feed (mM).
- `S_gln_f::Float64`: glutamine concentration in feed (mM).
- `F_max::Float64`: maximum feed rate (L/h).
- `Glc_min::Float64`: glucose threshold to turn feed ON (mM).
- `Glc_max::Float64`: glucose threshold to turn feed OFF (mM).
- `feed_on::Float64`: feed state (0.0 = off, 1.0 = on), mutated by callbacks.
"""
mutable struct MyFedBatchCHOParameters

    # growth kinetics -
    mu_max::Float64
    K_glc::Float64
    K_gln::Float64
    K_I_lac::Float64
    K_I_amm::Float64
    k_d::Float64
    KD_lac::Float64    # half-saturation for lactate in death rate (mM)
    KD_amm::Float64    # half-saturation for ammonia in death rate (mM)

    # product formation (Luedeking-Piret: q_P = alpha_P * mu + beta_P) -
    alpha_P::Float64   # growth-associated coefficient (mg/gDW)
    beta_P::Float64    # non-growth-associated coefficient (mg/gDW/h)

    # yield coefficients -
    Y_X_glc::Float64
    Y_X_gln::Float64
    Y_P_glc::Float64
    Y_P_gln::Float64
    Y_lac_glc::Float64
    Y_amm_gln::Float64

    # feed concentrations -
    S_glc_f::Float64
    S_gln_f::Float64

    # feed policy -
    F_max::Float64
    Glc_min::Float64
    Glc_max::Float64
    feed_on::Float64
end
