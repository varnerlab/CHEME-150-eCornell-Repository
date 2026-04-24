"""
    growth_rate(S_glc::Float64, S_gln::Float64, Lac::Float64, Amm::Float64,
        p::MyFedBatchCHOParameters) -> Float64

Compute the specific growth rate using Monod kinetics with dual by-product inhibition.

### Arguments
- `S_glc::Float64`: glucose concentration (mM).
- `S_gln::Float64`: glutamine concentration (mM).
- `Lac::Float64`: lactate concentration (mM).
- `Amm::Float64`: ammonia concentration (mM).
- `p::MyFedBatchCHOParameters`: model parameters.

### Returns
- `Float64`: specific growth rate (1/h).
"""
function growth_rate(S_glc::Float64, S_gln::Float64, Lac::Float64, Amm::Float64,
    p::MyFedBatchCHOParameters)::Float64

    # Monod terms for substrates -
    monod_glc = S_glc / (p.K_glc + S_glc);
    monod_gln = S_gln / (p.K_gln + S_gln);

    # inhibition terms for by-products -
    inhib_lac = p.K_I_lac / (p.K_I_lac + Lac);
    inhib_amm = p.K_I_amm / (p.K_I_amm + Amm);

    return p.mu_max * monod_glc * monod_gln * inhib_lac * inhib_amm;
end

"""
    death_rate(Lac::Float64, Amm::Float64, p::MyFedBatchCHOParameters) -> Float64

Compute the specific cell death rate as a product of two Monod-style terms in lactate
and ammonia. The death rate is near zero when by-products are low and approaches `k_d`
(the maximum death rate) as lactate and ammonia accumulate.

### Arguments
- `Lac::Float64`: lactate concentration (mM), clamped ≥ 0.
- `Amm::Float64`: ammonia concentration (mM), clamped ≥ 0.
- `p::MyFedBatchCHOParameters`: model parameters.

### Returns
- `Float64`: specific death rate (1/h).
"""
function death_rate(Lac::Float64, Amm::Float64, p::MyFedBatchCHOParameters)::Float64
    return p.k_d * (Lac / (p.KD_lac + Lac)) * (Amm / (p.KD_amm + Amm));
end

"""
    product_formation_rate(mu::Float64, p::MyFedBatchCHOParameters) -> Float64

Compute the specific antibody productivity using the Luedeking–Piret model:
`q_P = alpha_P * mu + beta_P`, where `alpha_P` is the growth-associated coefficient
(mg/gDW) and `beta_P` is the non-growth-associated coefficient (mg/gDW/h).

### Arguments
- `mu::Float64`: specific growth rate (1/h).
- `p::MyFedBatchCHOParameters`: model parameters.

### Returns
- `Float64`: specific antibody productivity (mg/gDW/h).
"""
function product_formation_rate(mu::Float64, p::MyFedBatchCHOParameters)::Float64
    return p.alpha_P * mu + p.beta_P;
end

"""
    substrate_uptake_glucose(mu::Float64, p::MyFedBatchCHOParameters) -> Float64

Compute the specific glucose uptake rate from growth demand only.
Product secretion (q_P) is decoupled from substrate stoichiometry, consistent with
the Xing et al. (2010) model where the B1/mAb mass balance does not appear in the
glucose balance. As mu -> 0 (substrate limited), q_glc -> 0 automatically, so glucose
cannot be driven negative by this term.

### Arguments
- `mu::Float64`: specific growth rate (1/h).
- `p::MyFedBatchCHOParameters`: model parameters.

### Returns
- `Float64`: specific glucose uptake rate (mmol/gDW/h).
"""
function substrate_uptake_glucose(mu::Float64, p::MyFedBatchCHOParameters)::Float64
    return mu / p.Y_X_glc;
end

"""
    substrate_uptake_glutamine(mu::Float64, p::MyFedBatchCHOParameters) -> Float64

Compute the specific glutamine uptake rate from growth demand only.
Product secretion (q_P) is decoupled from substrate stoichiometry, consistent with
the Xing et al. (2010) model where the B1/mAb mass balance does not appear in the
glutamine balance. As mu -> 0, q_gln -> 0 automatically.

### Arguments
- `mu::Float64`: specific growth rate (1/h).
- `p::MyFedBatchCHOParameters`: model parameters.

### Returns
- `Float64`: specific glutamine uptake rate (mmol/gDW/h).
"""
function substrate_uptake_glutamine(mu::Float64, p::MyFedBatchCHOParameters)::Float64
    return mu / p.Y_X_gln;
end

"""
    byproduct_formation_lactate(q_glc::Float64,
        p::MyFedBatchCHOParameters) -> Float64

Compute the specific lactate formation rate from glucose consumption.

### Arguments
- `q_glc::Float64`: specific glucose uptake rate (mmol/gDW/h).
- `p::MyFedBatchCHOParameters`: model parameters.

### Returns
- `Float64`: specific lactate formation rate (mmol/gDW/h).
"""
function byproduct_formation_lactate(q_glc::Float64,
    p::MyFedBatchCHOParameters)::Float64

    return p.Y_lac_glc * q_glc;
end

"""
    byproduct_formation_ammonia(q_gln::Float64,
        p::MyFedBatchCHOParameters) -> Float64

Compute the specific ammonia formation rate from glutamine consumption.

### Arguments
- `q_gln::Float64`: specific glutamine uptake rate (mmol/gDW/h).
- `p::MyFedBatchCHOParameters`: model parameters.

### Returns
- `Float64`: specific ammonia formation rate (mmol/gDW/h).
"""
function byproduct_formation_ammonia(q_gln::Float64,
    p::MyFedBatchCHOParameters)::Float64

    return p.Y_amm_gln * q_gln;
end
