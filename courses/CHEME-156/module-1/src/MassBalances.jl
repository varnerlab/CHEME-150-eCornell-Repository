"""
    rhs!(du, u, p::MyFedBatchCHOParameters, t)

Compute the right-hand side of the fed-batch CHO bioreactor ODE system.
The state vector is `u = [V, X, S_glc, S_gln, P, Lac, Amm]`.

### Arguments
- `du::Vector{Float64}`: derivative vector (modified in place).
- `u::Vector{Float64}`: state vector [V (L), X (gDW/L), S_glc (mM), S_gln (mM), P (mg/L), Lac (mM), Amm (mM)].
- `p::MyFedBatchCHOParameters`: model parameters (includes mutable `feed_on` field).
- `t::Float64`: current time (h).
"""
function rhs!(du, u, p::MyFedBatchCHOParameters, t)

    # unpack state vector -
    V, X, S_glc, S_gln, P, Lac, Amm = u;

    # clamp concentrations to prevent negative values from numerical noise -
    S_glc = max(S_glc, 0.0);
    S_gln = max(S_gln, 0.0);
    Lac = max(Lac, 0.0);
    Amm = max(Amm, 0.0);
    X = max(X, 0.0);

    # feed rate and dilution rate -
    F = p.feed_on * p.F_max;
    D = F / V;

    # kinetic rates -
    mu = growth_rate(S_glc, S_gln, Lac, Amm, p);
    mu_d = death_rate(Lac, Amm, p);
    q_P = product_formation_rate(mu, p);
    q_glc = substrate_uptake_glucose(mu, p);
    q_gln = substrate_uptake_glutamine(mu, p);
    q_lac = byproduct_formation_lactate(q_glc, p);
    q_amm = byproduct_formation_ammonia(q_gln, p);

    # mass balances -
    du[1] = F;                                      # dV/dt (L/h)
    du[2] = (mu - mu_d - D) * X;                    # dX/dt (gDW/L/h)
    du[3] = D * (p.S_glc_f - S_glc) - q_glc * X;   # dS_glc/dt (mM/h)
    du[4] = D * (p.S_gln_f - S_gln) - q_gln * X;   # dS_gln/dt (mM/h)
    du[5] = q_P * X - D * P;                        # dP/dt (mg/L/h)
    du[6] = q_lac * X - D * Lac;                     # dLac/dt (mM/h)
    du[7] = q_amm * X - D * Amm;                    # dAmm/dt (mM/h)

    return nothing;
end

"""
    build_feed_callbacks(p::MyFedBatchCHOParameters) -> CallbackSet

Construct a `CallbackSet` with two `ContinuousCallback`s that implement glucose-triggered
square wave feeding with hysteresis.

The feed turns ON when glucose drops below `p.Glc_min` and turns OFF when glucose
rises above `p.Glc_max`. Between these thresholds, the feed state is unchanged.

### Arguments
- `p::MyFedBatchCHOParameters`: model parameters (used to read `Glc_min` and `Glc_max`).

### Returns
- `CallbackSet`: pair of continuous callbacks for feed switching.
"""
function build_feed_callbacks(p::MyFedBatchCHOParameters)

    # callback 1: feed turns ON when glucose drops below Glc_min -
    # condition: S_glc - Glc_min. Downcrossing (positive to negative) means glucose fell below threshold.
    condition_on(u, t, integrator) = u[3] - integrator.p.Glc_min;
    function affect_on!(integrator)
        integrator.p.feed_on = 1.0;
    end
    cb_on = ContinuousCallback(condition_on, nothing, affect_on!);

    # callback 2: feed turns OFF when glucose rises above Glc_max -
    # condition: S_glc - Glc_max. Upcrossing (negative to positive) means glucose rose above threshold.
    condition_off(u, t, integrator) = u[3] - integrator.p.Glc_max;
    function affect_off!(integrator)
        integrator.p.feed_on = 0.0;
    end
    cb_off = ContinuousCallback(condition_off, affect_off!, nothing);

    return CallbackSet(cb_on, cb_off);
end
