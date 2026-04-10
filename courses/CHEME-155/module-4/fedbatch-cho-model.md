# Fed-Batch CHO Antibody Model Notes

This file summarizes the process-model discussion for fed-batch production of an antibody in CHO cells.

## 1. Core Dynamic Model

Use biomass concentration `X` in `gDW/L`, product concentration `P`, substrate concentrations `S_i`, by-product concentrations `B_j`, reactor volume `V`, feed rate `F`, and dilution rate:

```math
\frac{dV}{dt} = F, \qquad D = \frac{F}{V}
```

```math
\frac{dX}{dt} = (\mu - k_d - D)X
```

```math
\frac{dS_i}{dt} = D(S_{i,f} - S_i) - q_{S_i}X, \qquad i=1,\dots,n
```

```math
\frac{dP}{dt} = q_P X - DP
```

```math
\frac{dB_j}{dt} = q_{B_j} X - DB_j, \qquad j=1,\dots,m
```

For total biomass mass `M_X = XV` in `gDW`:

```math
\frac{dM_X}{dt} = (\mu - k_d) M_X
```

assuming sterile feed with no biomass in the feed.

## 2. Growth-Rate Model

A practical CHO fed-batch model is Monod kinetics with by-product inhibition:

```math
\mu = \mu_{\max}
\left(\frac{S_{glc}}{K_{glc}+S_{glc}}\right)
\left(\frac{S_{gln}}{K_{gln}+S_{gln}}\right)
\left(\frac{K_{I,lac}}{K_{I,lac}+Lac}\right)
\left(\frac{K_{I,amm}}{K_{I,amm}+Amm}\right)
```

Typical starting parameter ranges:

- `mu_max`: `0.025` to `0.050 1/h`
- `K_glc`: `1.0` to `2.25 mM`
- `K_gln`: `0.047` to `0.23 mM`
- `K_I,lac`: `43` to `52 mM`
- `K_I,amm`: `6.5` to `9.5 mM`

These are reasonable priors, not universal constants. They should be fit to clone- and process-specific data.

## 3. Typical Death Constant

First-order cell death:

```math
\text{death rate} = k_d X
```

Typical values:

- `k_d`: `0.003` to `0.007 1/h`
- Common fitted values: about `0.004` to `0.0066 1/h`

## 4. Typical Substrates and By-Products

### Minimal CHO fed-batch model

- Substrates: glucose, glutamine
- By-products: lactate, ammonia

This is the standard low-order model for many CHO fed-batch analyses.

### Slightly richer model

- Substrates: glucose, glutamine, amino-acid lump
- By-products: lactate, ammonia

Possible additional states if the data justify them:

- Substrates: oxygen, selected amino acids
- By-products: `CO2/HCO3-`, alanine, pyruvate, succinate

## 5. Yield Structure

The substrate uptake can be written in a yield-based form:

```math
q_{S_i} =
\frac{\mu}{Y_{X/S_i}}
+ \frac{q_P}{Y_{P/S_i}}
+ \sum_j \frac{q_{B_j}}{Y_{B_j/S_i}}
```

with consistent units for all yield coefficients.

Typical order-of-magnitude yield priors from the discussion:

- `Y_X/glc`: about `(0.14 to 0.41) x 10^9 cells / mmol glucose`
- `Y_X/gln`: about `(0.57 to 1.03) x 10^9 cells / mmol glutamine`
- `Y_lac/glc`: `0.7` to `1.6 mmol/mmol`
- `Y_amm/gln`: `0.67` to `0.74 mmol/mmol`

For product, many models use a specific productivity `q_P` rather than a constant yield:

- `q_P`: often around `20` to `50 pg/cell/day`

## 6. Switching from Cell Number to Biomass (`gDW`)

Using biomass concentration `X` in `gDW/L` is straightforward. The same mass balances apply; only the units of the yields change.

If `m_DW` is the dry mass per cell in `gDW/cell`, then:

```math
Y_{X/S_i}^{(gDW/mmol)} =
Y_{X/S_i}^{(10^9\ cells/mmol)} \times m_{DW} \times 10^9
```

A rough CHO dry-mass conversion is:

- `m_DW`: about `2 x 10^-10` to `4 x 10^-10 gDW/cell`
- Therefore `10^9 cells` is roughly `0.2` to `0.4 gDW`

## 7. Amino-Acid Lump Model

Treat an amino-acid lump as another substrate:

```math
\frac{dS_{AA}}{dt} = D(S_{AA,f} - S_{AA}) - q_{AA}X
```

Use a Pirt-like uptake model:

```math
q_{AA} = \frac{\mu}{Y_{X/AA}} + \frac{q_P}{Y_{P/AA}} + m_{AA}
```

Optionally, if the data support it:

```math
q_{AA} =
\frac{\mu}{Y_{X/AA}}
+ \frac{q_P}{Y_{P/AA}}
+ \sum_j \frac{q_{B_j}}{Y_{B_j/AA}}
+ m_{AA}
```

AA limitation can also appear in the growth law:

```math
\mu = \mu_{\max}
\left(\frac{S_{AA}}{K_{AA}+S_{AA}}\right)
\times (\text{other substrate terms})
\times (\text{inhibition terms})
```

Practical choices for the lump basis:

- `g total AA/L`
- `mmol N-equivalent/L`

If ammonia is modeled, AA uptake can contribute:

```math
q_{NH4} = Y_{NH4/gln} q_{gln} + Y_{NH4/AA} q_{AA}
```

## 8. If the Process Uses Glutamate Instead of Glutamine

If the process feed uses glutamate, replace or augment the glutamine state:

```math
\frac{dS_{glt}}{dt} = D(S_{glt,f} - S_{glt}) - q_{glt}X
```

Then modify the growth model accordingly:

```math
\mu = \mu_{\max}(T)
\left(\frac{S_{glc}}{K_{glc}+S_{glc}}\right)
\left(\frac{S_{glt}}{K_{glt}+S_{glt}}\right)
\left(\frac{K_{I,lac}}{K_{I,lac}+Lac}\right)
\left(\frac{K_{I,amm}}{K_{I,amm}+Amm}\right)
```

Ammonia formation can be re-expressed as:

```math
q_{NH4} \approx Y_{NH4/glt} q_{glt} + Y_{NH4/AA} q_{AA}
```

In practice:

- glutamate often yields less direct ammonia burden than glutamine-driven metabolism
- the exact effect should be identified from process data
- do not assume the same yield values as the glutamine process

## 9. Temperature Ramps

Temperature ramps do not change the balance structure; they change the kinetic parameters.

Make rate parameters temperature-dependent:

- `mu_max = mu_max(T)`
- `k_d = k_d(T)`
- `q_P = q_P(T)`
- optionally maintenance and yields: `m_i(T)`, `Y(T)`

If temperature is tightly controlled, treat `T(t)` as a known input equal to the setpoint profile.

If the temperature tracks the setpoint dynamically:

```math
\frac{dT}{dt} = \frac{T_{sp}(t)-T}{\tau_T}
```

### Practical recommendation

Start with a two-phase model:

- phase 1: pre-shift parameters
- phase 2: post-shift parameters

Only move to smooth temperature-dependent functions if the residuals indicate that the two-phase model is not adequate.

## 10. Recommended First-Pass Model

For a practical first implementation:

- Biomass: `X` in `gDW/L`
- Product: `P`
- Volume: `V`
- Substrates: glucose, glutamate or glutamine, AA-lump
- By-products: lactate, ammonia
- Inputs: feed rate `F(t)`, feed concentrations `S_f(t)`, temperature setpoint `T_sp(t)`

Recommended workflow:

1. Start with a minimal mechanistic model.
2. Fit `mu_max`, Monod constants, inhibition constants, `k_d`, and a small set of yield coefficients.
3. Use phase-dependent parameters if there is a temperature shift.
4. Add extra substrates or by-products only if the experimental residuals justify them.

## 11. Modeling Caution

The proposed equations are useful as a low-order fed-batch process model, but CHO metabolism is clone- and media-specific. Treat all quoted parameters as initial priors for estimation, not as fixed literature constants.
