# We add methods to the solve(...) generic exported by VLDataScienceMachineLearningPackage.

"""
    solve(problem::MyLinearProgramFeasibilityProblem) -> Dict{String,Any}

Solves the linear programming feasibility problem with the multiplicative weights update
algorithm. Treats each decision variable `x_i` as an expert with weight `w_i`. On each
round it forms the candidate `x_i = (τ/Φ) w_i` (a point on the simplex `Δ_m`), measures
the constraint residuals `r_k = (A x)_k - b_k - 2ϵ`, and for every violated constraint
`k` (those with `r_k > 0`) updates every weight with `w_j ← w_j (1 - η a_{kj})`.

### Arguments
- `problem::MyLinearProgramFeasibilityProblem`: the feasibility problem instance.

### Returns
A `Dict{String,Any}` with keys:
- `"x"::Array{Float64,1}`: the final candidate solution on the simplex.
- `"feasible"::Bool`: `true` if `A x ≤ b + 2ϵ` was reached within `T` iterations.
- `"iterations"::Int64`: the number of iterations performed.
- `"violation"::Array{Float64,1}`: the worst constraint violation `max_k((A x)_k - b_k)` per iteration.
- `"weights"::Array{Float64,2}`: the weight vector at the start of each iteration.
"""
function VLDataScienceMachineLearningPackage.solve(problem::MyLinearProgramFeasibilityProblem)::Dict{String,Any}

    # unpack the problem -
    A = problem.A;
    b = problem.b;
    τ = problem.τ;
    η = problem.η;
    ϵ = problem.ϵ;
    T = problem.T;
    (n, m) = size(A); # n constraints, m experts (decision variables)

    # initialize: every expert starts with weight 1 (deterministic) -
    w = ones(Float64, m);
    weights = zeros(Float64, T + 1, m); # weight history
    weights[1, :] .= w;
    violation = Float64[]; # worst (true) violation per iteration

    # main loop -
    converged = false;
    feasible = false;
    x = (τ / sum(w)) .* w;
    t = 1;
    while (converged == false)

        # candidate solution on the simplex: x ≥ 0 and sum(x) = τ -
        Φ = sum(w);
        x = (τ / Φ) .* w;

        # constraint residuals with the 2ϵ slack -
        Ax = A * x;
        r = Ax .- b .- 2.0 * ϵ;
        push!(violation, maximum(Ax .- b)); # store the worst true violation

        # convergence checks -
        if (maximum(r) ≤ 0.0)
            feasible = true;  # all constraints satisfied (within 2ϵ)
            converged = true;
            break;
        elseif (t ≥ T)
            feasible = false; # ran out of iterations; problem may be infeasible
            converged = true;
            break;
        end

        # weight update over the violated constraints V = {k : r_k > 0} -
        V = findall(rₖ -> rₖ > 0.0, r);
        for k ∈ V
            for j ∈ 1:m
                w[j] = w[j] * (1.0 - η * A[k, j]);
            end
        end

        # renormalize the weights to sum(w) = m. The candidate x depends only on the
        # ratio w/Φ, so this leaves the algorithm unchanged but avoids under/overflow.
        w .= (m / sum(w)) .* w;

        # store the updated weights and advance the counter -
        weights[t + 1, :] .= w;
        t += 1;
    end

    # package the results -
    results = Dict{String,Any}();
    results["x"] = x;
    results["feasible"] = feasible;
    results["iterations"] = t;
    results["violation"] = violation;
    results["weights"] = weights[1:t, :];

    # return -
    return results;
end

"""
    solve(problem::MyLinearProgramOptimizationProblem; maxsteps::Int64 = 60, tol::Float64 = 1e-4) -> Dict{String,Any}

Maximizes `c'x` over `{x ∈ Δ_m : A x ≤ b}` by binary search on the objective. For a trial
value `v`, the cut `c'x ≥ v` is added as the packing row `-c'x ≤ -v`, and the augmented
problem is handed to the feasibility solver. The search keeps the largest `v` for which the
feasibility solver returns a feasible point.

### Arguments
- `problem::MyLinearProgramOptimizationProblem`: the optimization problem instance.
- `maxsteps::Int64`: maximum number of bisection steps (keyword, default `60`).
- `tol::Float64`: stop when the search interval is narrower than this (keyword, default `1e-4`).

### Returns
A `Dict{String,Any}` with keys:
- `"x"::Array{Float64,1}`: the best feasible solution found.
- `"objective_value"::Float64`: the objective `c'x` at that solution.
- `"feasible"::Bool`: `true` if a base feasible point exists.
- `"iterations"::Int64`: the number of bisection steps performed.
"""
function VLDataScienceMachineLearningPackage.solve(problem::MyLinearProgramOptimizationProblem;
    maxsteps::Int64 = 60, tol::Float64 = 1e-4)::Dict{String,Any}

    # unpack the problem -
    A = problem.A;
    b = problem.b;
    c = problem.c;
    τ = problem.τ;
    η = problem.η;
    ϵ = problem.ϵ;
    T = problem.T;
    ρ = problem.ρ;
    (n, m) = size(A);

    # feasibility oracle for {x ∈ Δ(τ) : A x ≤ b, c'x ≥ v} -
    feasible_at = function(v::Float64)
        A_aug = vcat(A, reshape(-c, 1, m)); # add the cut -c'x ≤ -v
        b_aug = vcat(b, -v);
        sub = build(MyLinearProgramFeasibilityProblem, (
            A = A_aug, b = b_aug, τ = τ, η = η, ϵ = ϵ, T = T, ρ = ρ));
        res = solve(sub);
        return (res["feasible"], res["x"]);
    end

    # lower bound: the objective at a base feasible point (no objective cut) -
    base = build(MyLinearProgramFeasibilityProblem, (A = A, b = b, τ = τ, η = η, ϵ = ϵ, T = T, ρ = ρ));
    base_res = solve(base);
    results = Dict{String,Any}();
    if (base_res["feasible"] == false)
        results["feasible"] = false;
        results["x"] = base_res["x"];
        results["objective_value"] = NaN;
        results["iterations"] = 0;
        return results;
    end
    x_best = base_res["x"];
    v_lo = dot(c, x_best);    # known feasible objective value
    v_hi = τ * maximum(c);    # upper bound on c'x over the simplex

    # binary search on the objective cut v -
    steps = 0;
    while (((v_hi - v_lo) > tol) && (steps < maxsteps))
        v_mid = 0.5 * (v_lo + v_hi);
        (ok, x) = feasible_at(v_mid);
        if (ok == true)
            v_lo = v_mid;   # v_mid is achievable; raise the floor
            x_best = x;
        else
            v_hi = v_mid;   # v_mid is not achievable; lower the ceiling
        end
        steps += 1;
    end

    # package the results -
    results["feasible"] = true;
    results["x"] = x_best;
    results["objective_value"] = dot(c, x_best);
    results["iterations"] = steps;

    # return -
    return results;
end
