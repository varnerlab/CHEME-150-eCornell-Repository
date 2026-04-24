"""
    build_legS_matrices(h::Int) -> (A, B)

Return the continuous-time HiPPO-LegS state and input matrices for hidden
dimension `h`, using the sign convention from Gu et al. (2020) in which the
LTI system is the stable form `dx/dt = A x + B u`. The entries are

    A[i,k] = -√((2i+1)(2k+1))  if i > k
           = -(i + 1)           if i == k
           =  0                 if i < k

    B[i,1] =  √(2i+1)

with 1-based indexing. The minus signs on `A` make the continuous-time
eigenvalues negative (stable), matching the convention used in the S4
codebase.
"""
function build_legS_matrices(h::Int)
    A = zeros(Float64, h, h)
    B = zeros(Float64, h, 1)
    for i in 1:h
        for k in 1:h
            if i > k
                A[i, k] = -sqrt((2i + 1) * (2k + 1))
            elseif i == k
                A[i, k] = -(i + 1)
            end
        end
        B[i, 1] = sqrt(2i + 1)
    end
    return (A, B)
end

"""
    discretize(A, B; Δt, method=:bilinear) -> (Ā, B̄)

Bilinear (Tustin) discretization of the continuous pair `(A, B)` at step `Δt`.
Valid for MIMO of arbitrary shape; `A` must be square. Returns discrete
matrices of the same shapes as the inputs:

    Ā = (I - Δt/2 A)⁻¹ (I + Δt/2 A)
    B̄ = (I - Δt/2 A)⁻¹ (Δt B)
"""
function discretize(A::AbstractMatrix, B::AbstractMatrix; Δt::Float64, method::Symbol=:bilinear)
    method == :bilinear || error("only :bilinear is supported, got $(method)")
    Δt > 0 || error("Δt must be positive")
    H, W = size(A)
    H == W || error("A must be square, got size $(size(A))")
    size(B, 1) == H || error("B has $(size(B, 1)) rows, expected $(H)")
    I_H = Matrix{Float64}(I, H, H)
    M   = I_H - (Δt / 2) .* A
    Ā   = M \ (I_H + (Δt / 2) .* A)
    B̄   = M \ (Δt .* B)
    return (Ā, B̄)
end

"""
    build(::Type{MySisoLegSHippoModel}; h, Δt, C=nothing, D=0.0, x₀=nothing) -> MySisoLegSHippoModel

Build a LegS HiPPO model instance. Constructs the continuous `(A, B)` and
the bilinear-discretized `(Ā, B̄)`. If `C` is not supplied, a zero row
vector of length `h` is used; `fit_C!` overwrites it.
"""
function build(::Type{MySisoLegSHippoModel};
    h::Int,
    Δt::Float64,
    C::Union{Nothing, AbstractMatrix, AbstractVector} = nothing,
    D::Float64 = 0.0,
    x₀::Union{Nothing, AbstractVector} = nothing,
)
    (A, B) = build_legS_matrices(h)
    (Ā, B̄) = discretize(A, B; Δt = Δt)
    Cmat = if C === nothing
        zeros(Float64, 1, h)
    elseif C isa AbstractVector
        reshape(collect(Float64, C), 1, h)
    else
        Matrix{Float64}(C)
    end
    x0 = x₀ === nothing ? zeros(Float64, h) : collect(Float64, x₀)
    return MySisoLegSHippoModel(h, Δt, A, B, Ā, B̄, Cmat, D, x0)
end

"""
    rollout(model::MySisoLegSHippoModel, u::AbstractVector) -> X

Run the discrete-time recursion `x_{t+1} = Ā x_t + B̄ u_t` with `x_0 = model.x₀`
and return the hidden-state matrix `X ∈ ℝ^{T×h}` whose `t`-th row is `x_t`
for `t = 1, …, T`. No readout is applied.
"""
function rollout(model::MySisoLegSHippoModel, u::AbstractVector)
    T = length(u)
    h = model.h
    X = zeros(Float64, T, h)
    x = copy(model.x₀)
    @inbounds for t in 1:T
        x = model.Ā * x + model.B̄ * u[t]
        X[t, :] = x
    end
    return X
end

"""
    solve(model::MySisoLegSHippoModel, u::AbstractVector) -> (X, y)

Roll the model forward on input `u` and apply the current readout `C` (and
scalar `D`) to produce output `y ∈ ℝ^T`. Returns the hidden-state matrix
`X ∈ ℝ^{T×h}` and `y`.
"""
function solve(model::MySisoLegSHippoModel, u::AbstractVector)
    X = rollout(model, u)
    y = (X * model.C')[:, 1] .+ model.D .* u
    return (X, y)
end

"""
    fit_C!(model::MySisoLegSHippoModel, u::AbstractVector, y::AbstractVector; λ=1e-4) -> model

Fit the readout `C` by ridge regression on the hidden-state rollout of `u`
against targets `y`. Solves

    Ĉᵀ = (XᵀX + λ I)⁻¹ Xᵀ y

and updates `model.C` in place. Returns the model.
"""
function fit_C!(model::MySisoLegSHippoModel, u::AbstractVector, y::AbstractVector; λ::Float64 = 1.0e-4)
    length(u) == length(y) || error("length(u) = $(length(u)) must equal length(y) = $(length(y))")
    X = rollout(model, u)
    h = model.h
    G = X' * X + λ .* Matrix{Float64}(I, h, h)
    rhs = X' * collect(Float64, y)
    c = G \ rhs
    model.C = reshape(c, 1, h)
    return model
end

"""
    predict(model::MySisoLegSHippoModel, u::AbstractVector; warmup::Int=0) -> (X, y)

Roll the model forward on an out-of-sample input `u` using the currently
trained `C`. If `warmup > 0`, the first `warmup` samples of `X` and `y` are
dropped to let the hidden state forget `x₀ = 0`.
"""
function predict(model::MySisoLegSHippoModel, u::AbstractVector; warmup::Int = 0)
    warmup >= 0 || error("warmup must be nonnegative")
    (X, y) = solve(model, u)
    if warmup > 0
        warmup < length(u) || error("warmup = $(warmup) must be smaller than length(u) = $(length(u))")
        return (X[(warmup + 1):end, :], y[(warmup + 1):end])
    end
    return (X, y)
end

"""
    log_growth_rate(prices::AbstractVector; Δt::Float64) -> g

Annualized log growth rate sequence `gₜ = (1/Δt) · log(Pₜ / Pₜ₋₁)` for a
price series, returned as a vector of length `length(prices) - 1`.
"""
function log_growth_rate(prices::AbstractVector; Δt::Float64)
    length(prices) >= 2 || error("need at least two prices")
    return (1.0 / Δt) .* diff(log.(collect(Float64, prices)))
end

"""
    static_alpha_beta(u::AbstractVector, y::AbstractVector) -> (α, β)

Ordinary-least-squares fit of the single-index-model regression
`yₜ = α + β uₜ + εₜ` on the full paired series `(u, y)`. Returns the scalar
intercept `α` and slope `β`, which the caller can then freeze and apply to
any future input via `α .+ β .* u_new`. The intercept is left free (this is
the single-index model, not CAPM).
"""
function static_alpha_beta(u::AbstractVector, y::AbstractVector)
    length(u) == length(y) || error("length(u) = $(length(u)) must equal length(y) = $(length(y))")
    length(u) >= 2 || error("need at least two paired samples")
    ū = mean(u)
    ȳ = mean(y)
    num = 0.0
    den = 0.0
    @inbounds for i in eachindex(u)
        du = u[i] - ū
        num += du * (y[i] - ȳ)
        den += du * du
    end
    den > 0 || error("zero variance in u; cannot fit a slope")
    β = num / den
    α = ȳ - β * ū
    return (α, β)
end

"""
    rolling_beta(u::AbstractVector, y::AbstractVector; window::Int) -> (α, β, ŷ)

Rolling-window ordinary-least-squares estimate of the single-index-model regression
`yₜ = α + β uₜ + εₜ` using a lookback window of size `window`. The intercept `α`
is left free (unlike CAPM, which pins `α = 0` on excess returns). For every
`t ≥ window + 1`, refit `(α_t, β_t)` on the preceding `window` samples
`(u_{t-window:t-1}, y_{t-window:t-1})` and form the prediction
`ŷ_t = α_t + β_t · u_t`. The first `window` entries of `α`, `β`, and `ŷ`
are set to `NaN` because there is no full lookback available.
"""
function rolling_beta(u::AbstractVector, y::AbstractVector; window::Int)
    length(u) == length(y) || error("length(u) = $(length(u)) must equal length(y) = $(length(y))")
    window >= 2 || error("window must be at least 2")
    T = length(u)
    window < T || error("window = $(window) must be smaller than series length $(T)")
    α = fill(NaN, T)
    β = fill(NaN, T)
    ŷ = fill(NaN, T)
    @inbounds for t in (window + 1):T
        uw = @view u[(t - window):(t - 1)]
        yw = @view y[(t - window):(t - 1)]
        ū  = sum(uw) / window
        ȳ  = sum(yw) / window
        num = 0.0
        den = 0.0
        for i in 1:window
            du = uw[i] - ū
            num += du * (yw[i] - ȳ)
            den += du * du
        end
        β_t   = den > 0 ? num / den : 0.0
        α_t   = ȳ - β_t * ū
        α[t]  = α_t
        β[t]  = β_t
        ŷ[t]  = α_t + β_t * u[t]
    end
    return (α, β, ŷ)
end

"""
    _build_legS_single(h::Int) -> (A, b)

Return the single-channel LegS state matrix `A ∈ ℝ^{h×h}` and input vector
`b ∈ ℝ^{h}` in the 1-based sign convention from Gu et al. 2020 with the stable
form `dx/dt = A x + b u`. In Julia, `b` is returned as a length-`h` vector
rather than an explicit `h×1` matrix.
"""
function _build_legS_single(h::Int)
    h >= 1 || error("h must be positive")
    A = zeros(Float64, h, h)
    b = zeros(Float64, h)
    for i in 1:h
        for k in 1:h
            if i > k
                A[i, k] = -sqrt((2i + 1) * (2k + 1))
            elseif i == k
                A[i, k] = -(i + 1)
            end
        end
        b[i] = sqrt(2i + 1)
    end
    return (A, b)
end

"""
    build_legS_matrices_mimo(h::Int, d_in::Int) -> (A, B)

Build the block-diagonal MIMO LegS state matrix `A ∈ ℝ^{(h·d_in)×(h·d_in)}` and
input matrix `B ∈ ℝ^{(h·d_in)×d_in}`. Each input channel drives one independent
`h`-dim LegS filter: `A` is block-diagonal with `d_in` copies of the single-
channel LegS block on its diagonal, and column `j` of `B` is the standard LegS
input vector placed in block `j` (zero elsewhere).
"""
function build_legS_matrices_mimo(h::Int, d_in::Int)
    h >= 1   || error("h must be positive")
    d_in >= 1 || error("d_in must be positive")
    (A_single, b_single) = _build_legS_single(h)
    H = h * d_in
    A = zeros(Float64, H, H)
    B = zeros(Float64, H, d_in)
    for j in 1:d_in
        rows = ((j - 1) * h + 1):(j * h)
        A[rows, rows] = A_single
        B[rows, j]    = b_single
    end
    return (A, B)
end

"""
    build(::Type{MyMimoLegSHippoModel}; h, d_in, d_out, Δt, C=nothing, D=nothing, x₀=nothing) -> MyMimoLegSHippoModel

Build a MIMO LegS HiPPO model. Constructs the block-diagonal continuous pair
`(A, B)` and its bilinear-discretized counterpart `(Ā, B̄)`. `C` defaults to a
zero matrix of shape `(d_out, h·d_in)` (which `fit_C!` overwrites), `D`
defaults to a zero matrix of shape `(d_out, d_in)`, and `x₀` defaults to
zeros. The returned model stores both the continuous operators `(A, B)` and the
discrete operators `(Ā, B̄)` so either representation can be inspected.
"""
function build(::Type{MyMimoLegSHippoModel};
    h::Int, d_in::Int, d_out::Int, Δt::Float64,
    C::Union{Nothing, AbstractMatrix} = nothing,
    D::Union{Nothing, AbstractMatrix} = nothing,
    x₀::Union{Nothing, AbstractVector} = nothing,
)
    d_out >= 1 || error("d_out must be positive")
    Δt > 0 || error("Δt must be positive")
    (A, B) = build_legS_matrices_mimo(h, d_in)
    (Ā, B̄) = discretize(A, B; Δt = Δt)
    H = h * d_in
    Cmat = C === nothing ? zeros(Float64, d_out, H) : Matrix{Float64}(C)
    Dmat = D === nothing ? zeros(Float64, d_out, d_in) : Matrix{Float64}(D)
    x0   = x₀ === nothing ? zeros(Float64, H) : collect(Float64, x₀)
    size(Cmat) == (d_out, H)     || error("C has shape $(size(Cmat)), expected ($(d_out), $(H))")
    size(Dmat) == (d_out, d_in)  || error("D has shape $(size(Dmat)), expected ($(d_out), $(d_in))")
    length(x0) == H              || error("x₀ has length $(length(x0)), expected $(H)")
    return MyMimoLegSHippoModel(h, d_in, d_out, Δt, A, B, Ā, B̄, Cmat, Dmat, x0)
end

"""
    rollout(model::MyMimoLegSHippoModel, U::AbstractMatrix) -> X

Run the discrete-time recursion `xₜ = Ā xₜ₋₁ + B̄ uₜ` with `x₀ = model.x₀` on
the input matrix `U ∈ ℝ^{T×d_in}` (row `t` is the input vector at time `t`)
and return the hidden-state matrix `X ∈ ℝ^{T×(h·d_in)}` whose `t`-th row is
`xₜᵀ`. The readout is not applied, so `rollout` isolates the latent dynamics.
"""
function rollout(model::MyMimoLegSHippoModel, U::AbstractMatrix)
    T, d_in = size(U)
    d_in == model.d_in || error("U has $(d_in) columns, expected $(model.d_in)")
    H = model.h * model.d_in
    X = zeros(Float64, T, H)
    x = copy(model.x₀)
    @inbounds for t in 1:T
        u = @view U[t, :]
        x = model.Ā * x + model.B̄ * u
        X[t, :] = x
    end
    return X
end

"""
    solve(model::MyMimoLegSHippoModel, U::AbstractMatrix) -> (X, Y)

Roll the model forward on the input matrix `U ∈ ℝ^{T×d_in}` and apply the
current readout to produce `Y ∈ ℝ^{T×d_out}` with `Yₜ = C xₜ + D uₜ`. Returns
the hidden-state matrix `X` and the output matrix `Y`. Row `t` of `Y`
corresponds to the output vector at time step `t`.
"""
function solve(model::MyMimoLegSHippoModel, U::AbstractMatrix)
    X = rollout(model, U)
    Y = X * model.C' .+ U * model.D'
    return (X, Y)
end

"""
    fit_C!(model::MyMimoLegSHippoModel, U::AbstractMatrix, Y::AbstractMatrix; λ=1e-4) -> model

Fit the readout `C` by closed-form ridge regression on the hidden-state
rollout of `U` against targets `Y`. With `D` fixed the loss decouples across
output channels, so the normal equations reduce to a single multi-output
solve

    Cᵀ = (XᵀX + λ I)⁻¹ Xᵀ (Y - U Dᵀ)

of size `(h·d_in)`. `U ∈ ℝ^{T×d_in}` and `Y ∈ ℝ^{T×d_out}` must have the same
number of rows. The function mutates `model.C` and returns `model` for
convenience.
"""
function fit_C!(model::MyMimoLegSHippoModel, U::AbstractMatrix, Y::AbstractMatrix; λ::Float64 = 1.0e-4)
    λ >= 0 || error("λ must be nonnegative")
    T_u, d_in = size(U)
    T_y, d_out = size(Y)
    T_u == T_y         || error("U has $(T_u) rows, Y has $(T_y) rows; must match")
    d_in == model.d_in || error("U has $(d_in) columns, expected $(model.d_in)")
    d_out == model.d_out || error("Y has $(d_out) columns, expected $(model.d_out)")
    X = rollout(model, U)
    Y_res = Y .- U * model.D'
    H = model.h * model.d_in
    G = X' * X + λ .* Matrix{Float64}(I, H, H)
    rhs = X' * collect(Float64, Y_res)
    CT = G \ rhs
    model.C = Matrix{Float64}(CT')
    return model
end

"""
    predict(model::MyMimoLegSHippoModel, U::AbstractMatrix; warmup::Int=0) -> (X, Y)

Roll the model on a new input matrix `U` using the currently trained `C` and
`D`. If `warmup > 0`, the first `warmup` rows of `X` and `Y` are discarded so
the hidden state has time to forget `x₀`. This is useful when the initial
condition is artificial but the long-time forecast is what matters.
"""
function predict(model::MyMimoLegSHippoModel, U::AbstractMatrix; warmup::Int = 0)
    warmup >= 0 || error("warmup must be nonnegative")
    (X, Y) = solve(model, U)
    if warmup > 0
        warmup < size(U, 1) || error("warmup = $(warmup) must be smaller than length $(size(U, 1))")
        return (X[(warmup + 1):end, :], Y[(warmup + 1):end, :])
    end
    return (X, Y)
end

"""
    make_forecast_pair(U::AbstractMatrix, k::Int) -> (U_input, U_target)

Given a full input matrix `U ∈ ℝ^{T×d}` and a forecast horizon `k ≥ 1`, build
the aligned pair
  * `U_input  = U[1:T-k, :]`          (samples 1 through T-k)
  * `U_target = U[1+k:T, :]`          (samples 1+k through T)
so that a model fit on `(U_input, U_target)` learns to predict the input `k`
steps ahead, `uₜ ↦ uₜ₊ₖ`. Both returned matrices have `T - k` rows.
"""
function make_forecast_pair(U::AbstractMatrix, k::Int)
    k >= 1 || error("forecast horizon k must be at least 1, got $(k)")
    T = size(U, 1)
    k < T || error("forecast horizon k = $(k) must be smaller than length $(T)")
    U_input  = U[1:(T - k), :]
    U_target = U[(1 + k):T, :]
    return (U_input, U_target)
end
