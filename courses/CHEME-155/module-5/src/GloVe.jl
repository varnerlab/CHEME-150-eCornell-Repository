"""
    build_weighted_cooccurrence_matrix(sentences::Array{String,1}, vocabulary::Dict{String, Int64};
        window_size::Int64 = 2) -> Array{Float64, 2}

Build a weighted word-word co-occurrence matrix from a corpus. For each center word and context word
within the window, adds a weight of `1/d` where `d` is the distance between them. Closer words
receive higher weight.

### Arguments
- `sentences::Array{String,1}`: array of sentence strings.
- `vocabulary::Dict{String, Int64}`: dictionary mapping words to vocabulary indices.
- `window_size::Int64`: number of tokens on each side to consider as context (default: 2).

### Returns
- `Array{Float64, 2}`: weighted co-occurrence matrix of shape `(vocab_size, vocab_size)`.
"""
function build_weighted_cooccurrence_matrix(sentences::Array{String,1}, vocabulary::Dict{String, Int64};
    window_size::Int64 = 2)::Array{Float64, 2}

    # initialize -
    vocab_size = length(vocabulary);
    X = zeros(Float64, vocab_size, vocab_size);

    # populate the co-occurrence matrix with distance weighting -
    for sentence in sentences
        augmented_sentence = "<bos> " * sentence * " <eos>";
        words = split(lowercase(augmented_sentence)) .|> String;
        word_indices = [get(vocabulary, word, vocabulary["<unk>"]) for word in words];

        for i ∈ eachindex(word_indices)
            center_index = word_indices[i];
            left = max(1, i - window_size);
            right = min(length(word_indices), i + window_size);
            for j ∈ left:right
                if j == i
                    continue
                end
                context_index = word_indices[j];
                distance = abs(i - j);
                X[center_index, context_index] += 1.0 / distance;
            end
        end
    end

    return X;
end

"""
    glove_weight(x::Float64; x_max::Float64 = 100.0, α::Float64 = 0.75) -> Float64

Compute the GloVe weighting function: `f(x) = min(1, (x / x_max)^α)`.
Prevents rare and common co-occurrences from dominating training.

### Arguments
- `x::Float64`: co-occurrence count.
- `x_max::Float64`: cutoff value (default: 100.0).
- `α::Float64`: scaling exponent (default: 0.75).

### Returns
- `Float64`: weight in the range `[0, 1]`.
"""
function glove_weight(x::Float64; x_max::Float64 = 100.0, α::Float64 = 0.75)::Float64
    return min(1.0, (x / x_max)^α);
end

"""
    train_glove(X::Array{Float64, 2}, vocab_size::Int64;
        d::Int64 = 5, η::Float64 = 0.05, num_epochs::Int64 = 500,
        x_max::Float64 = 100.0, α::Float64 = 0.75) -> Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

Train GloVe embeddings by minimizing the weighted least-squares objective over non-zero
co-occurrence entries. For each pair (i,j) where X[i,j] > 0, the objective is:
`f(X_ij) * (w_i' * w_ctx_j + b_i + b_ctx_j - log(X_ij))^2`.

### Arguments
- `X::Array{Float64, 2}`: co-occurrence matrix of shape `(vocab_size, vocab_size)`.
- `vocab_size::Int64`: size of the vocabulary.
- `d::Int64`: embedding dimension (default: 5).
- `η::Float64`: learning rate (default: 0.05).
- `num_epochs::Int64`: number of training epochs (default: 500).
- `x_max::Float64`: weighting function cutoff (default: 100.0).
- `α::Float64`: weighting function exponent (default: 0.75).

### Returns
- `Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}`:
  tuple of (W_word, W_context, b_word, b_context, loss_history) where W matrices are `(d, vocab_size)`.
"""
function train_glove(X::Array{Float64, 2}, vocab_size::Int64;
    d::Int64 = 5, η::Float64 = 0.05, num_epochs::Int64 = 500,
    x_max::Float64 = 100.0, α::Float64 = 0.75)::Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

    # initialize parameters -
    W = randn(d, vocab_size) * 0.1;
    W_ctx = randn(d, vocab_size) * 0.1;
    b = zeros(Float64, vocab_size);
    b_ctx = zeros(Float64, vocab_size);

    # collect non-zero entries -
    nonzero_pairs = Tuple{Int64, Int64}[];
    for i in 1:vocab_size
        for j in 1:vocab_size
            if X[i, j] > 0.0
                push!(nonzero_pairs, (i, j));
            end
        end
    end

    # training loop -
    loss_history = zeros(num_epochs);
    for epoch ∈ 1:num_epochs
        epoch_loss = 0.0;
        for (i, j) in nonzero_pairs

            # compute prediction and error -
            x_ij = X[i, j];
            f_ij = glove_weight(x_ij, x_max = x_max, α = α);
            ŷ = dot(W[:, i], W_ctx[:, j]) + b[i] + b_ctx[j];
            e = ŷ - log(x_ij);
            weighted_e = f_ij * e;

            # accumulate loss -
            epoch_loss += f_ij * e^2;

            # compute gradients and update -
            grad_w = 2.0 * weighted_e * W_ctx[:, j];
            grad_w_ctx = 2.0 * weighted_e * W[:, i];
            grad_b = 2.0 * weighted_e;
            grad_b_ctx = 2.0 * weighted_e;

            W[:, i] .-= η * grad_w;
            W_ctx[:, j] .-= η * grad_w_ctx;
            b[i] -= η * grad_b;
            b_ctx[j] -= η * grad_b_ctx;
        end
        loss_history[epoch] = epoch_loss / length(nonzero_pairs);
    end

    return (W, W_ctx, b, b_ctx, loss_history);
end
