function _rref!(A::AbstractMatrix{T}, eps_val=nothing) where {T<:Union{AbstractFloat,Rational}}
    if eps_val === nothing
        eps_val = T <: Rational ? zero(T) : eps(norm(A, Inf))
    end
    nr, nc = size(A)
    i = j = 1
    while i <= nr && j <= nc
        (m, mi) = findmax(abs.(A[i:nr,j]))
        mi = mi+i - 1
        if m <= eps_val
            if eps_val > 0
                A[i:nr,j] .= zero(T)
            end
            j += 1
        else
            for k=j:nc
                A[i, k], A[mi, k] = A[mi, k], A[i, k]
            end
            d = A[i,j]
            for k = j:nc
                A[i,k] /= d
            end
            for k = 1:nr
                if k != i
                    d = A[k,j]
                    for l = j:nc
                        A[k,l] -= d*A[i,l]
                    end
                end
            end
            i += 1
            j += 1
        end
    end
    A
end

function _rref!(A::AbstractMatrix{T}, eps_val=nothing) where {T<:Integer}
    throw(ArgumentError("_rref! requires floating or rational element types; convert with float.(A) or rationalize.(A)."))
end

function _upper_triangular(matrix::Matrix)
    A = copy(matrix)
    (n, m) = size(A)
    for k = 1:min(n,m)
        for i = k+1:n
            l = A[i,k] / A[k,k]
            for j = k:m
                A[i,j] = A[i,j] - l*A[k,j]
            end
        end
    end
    return A
end

function _smallest_singular_vector(A::Matrix{Float64})::Array{Float64,1}
    F = svd(A)
    return F.Vt[end, :]
end

function _nullspace_vector(A::Matrix{Float64}; tol::Float64 = 1e-10)::Array{Float64,1}
    R = _rref!(copy(A), tol)
    n = size(R,2)
    nrows = size(R,1)
    pivot_cols = Int[]
    pivot_rows = Int[]
    for i in 1:nrows
        pivot_col = 0
        for j in 1:n
            if abs(R[i,j]) > tol
                pivot_col = j
                break
            end
        end
        if pivot_col != 0
            push!(pivot_cols, pivot_col)
            push!(pivot_rows, i)
        end
    end

    free_cols = setdiff(collect(1:n), pivot_cols)
    if isempty(free_cols)
        return _smallest_singular_vector(A)
    end

    x = zeros(n)
    free_col = free_cols[end]
    x[free_col] = 1.0
    for idx in eachindex(pivot_cols)
        row = pivot_rows[idx]
        pc = pivot_cols[idx]
        s = 0.0
        for j in free_cols
            s += R[row,j] * x[j]
        end
        x[pc] = -s
    end

    if norm(x) <= tol
        return _smallest_singular_vector(A)
    end

    return x
end


"""
    ⊗(a::Array{Float64,1},b::Array{Float64,1}) -> Array{Float64,2}

Compute the outer product of two vectors `a` and `b` and returns a matrix.

### Arguments
- `a::Array{Float64,1}`: a vector of length `m`.
- `b::Array{Float64,1}`: a vector of length `n`.

### Returns
- `Y::Array{Float64,2}`: a matrix of size `m x n` such that `Y[i,j] = a[i]*b[j]`.
"""
function ⊗(a::Array{Float64,1},b::Array{Float64,1})::Array{Float64,2}

    # initialize -
    m = length(a)
    n = length(b)
    Y = zeros(m,n)

    # main loop 
    for i ∈ 1:m
        for j ∈ 1:n
            Y[i,j] = a[i]*b[j]
        end
    end

    # return 
    return Y
end

"""
    qriteration(A::Array{Float64,2}; maxiter::Int64 = 10, tolerance::Float64 = 1e-9) -> Tuple

Computes the eigenvalues and eigenvectors of a real matrix `A` using the QR iteration method.

### Arguments
- `A::Array{Float64,2}`: a real matrix of size `n x n`.
- `maxiter::Int64`: the maximum number of iterations (default is `10`).
- `tolerance::Float64`: the tolerance for the stopping criterion (default is `1e-9`).

### Returns
- `Tuple`: a tuple of two elements: the first element is an array of eigenvalues and the second element is a dictionary of eigenvectors.
"""
function qriteration(A::Array{Float64,2}; 
    maxiter::Int64 = 10, tolerance::Float64 = 1e-9)::Tuple{Array{Float64,1}, Dict{Int64,Array{Float64,1}}}

    # initialize 
    number_of_rows = size(A,1);
    AD = A;
    ϵ = Inf
    eigenvectors = Dict{Int64,Array{Float64,1}}()

    # compute iteration zero -
    Fₒ = qr(AD)
    Qₒ = Matrix(Fₒ.Q);
    λ = diag(Qₒ)

    # Phase 1: compute eigenvalues
    for _ ∈ 1:maxiter
        
        # factor -
        F = qr(AD)
        Q = Matrix(F.Q);

        # update
        AD = transpose(Q)*AD*Q

        # check: should we continute to iterate?
        λ′ = diag(AD);
        ϵ = norm(λ - λ′)
        if (ϵ < tolerance)
            
            # update the values
            λ = λ′
            
            # we are done with phase 1
            break;
        else
            # have not hit the tolerance - go around some more
            λ = λ′
        end
    end

    # sort the eigenvalues -
    sort!(λ)

    # Phase 2: compute the eigenvectors
    for i ∈ 1:number_of_rows
        
        # setup the homogenous system -
        AH = A - λ[i]*Matrix{Float64}(I, number_of_rows, number_of_rows);
        
        # compute the unscaled eigenvector from the nullspace -
        EVUS = _nullspace_vector(AH; tol = tolerance)

        # compute a scaling normalizing parameter
        d = norm(EVUS);

        # compute the scaled eigenvector - not sure why the (-1)^(i+1)?
        if !(isfinite(d)) || d <= eps(Float64)
            EVUS = _smallest_singular_vector(AH)
            d = norm(EVUS)
        end
        if !(isfinite(d)) || d <= eps(Float64)
            eigenvectors[i] = zeros(number_of_rows)
        else
            eigenvectors[i] = ((-1)^(i+1))*(1/d)*EVUS
        end
    end

    # return -
    return (λ, eigenvectors)
end


"""
    poweriteration(A::Array{<:Number,2}, v::Array{<:Number,1}; 
        maxiter::Int = 100, ϵ::Float64 = 0.0001)

This function computes the dominant eigenvector and eigenvalue of a matrix using the power iteration method.

### Arguments
- `A::Array{<:Number,2}`: A square matrix of real numbers.
- `v::Array{<:Number,1}`: An initial guess for the eigenvector.
- `maxiter::Int = 100`: The maximum number of iterations (optional).
- `ϵ::Float64 = 0.0001`: The convergence criterion (optional).

### Output
- A named tuple containing the dominant eigenvector and eigenvalue.
"""
function poweriteration(A::Array{<:Number,2}, v::Array{<:Number,1}; 
    maxiter::Int = 100, ϵ::Float64 = 0.0001)::NamedTuple

    # initialize
    loopcount = 1;
    should_we_stop = false;

    while (should_we_stop == false)
        
        # compute the next iteration
        w = A * v;
        w = w / norm(w);

        # check if we should stop
        if (norm(w - v)^2 ≤ ϵ || loopcount ≥ maxiter)
            should_we_stop = true;
            println("Converged in $(loopcount) iterations"); # let the user know how many iterations it took
        else
            v = w; # update the vector
            loopcount = loopcount + 1; # update the loop count
        end
    end
    
    # compute the eigenvalue -
    λ = dot(A * v, v) / dot(v, v);

    # return the result
    return (vector = v, value = λ);
end