"""
    build_cooccurrence_matrix(sentences, vocabulary; window_size=2) -> Array{Int64,2}

Build a symmetric word co-occurrence count matrix from a corpus.

For each word at position `i` in a sentence, all words within `window_size`
positions on either side contribute one count to entry `[w_idx, ctx_idx]`.
The matrix has size `vocab_size × vocab_size`. Unknown tokens are mapped to
`<unk>` if present in `vocabulary` and ignored otherwise.

### Arguments
- `sentences::Array{String,1}`: raw sentences; each sentence is lowercased and split on whitespace.
- `vocabulary::Dict{String,Int64}`: mapping from token to 1-based vocabulary index.
- `window_size::Int64`: half-width of the context window (default: 2).

### Returns
- `::Array{Int64,2}`: symmetric co-occurrence count matrix of size `vocab_size × vocab_size`.
"""
function build_cooccurrence_matrix(sentences::Array{String,1}, vocabulary::Dict{String,Int64}; window_size::Int64=2)::Array{Int64,2}

    # initialize -
    vocab_size = length(vocabulary);
    cooccurrence = zeros(Int64, vocab_size, vocab_size);

    # main loop: iterate over sentences -
    for sentence in sentences

        # tokenize and lowercase -
        words = split(lowercase(sentence));

        for (i, word) in enumerate(words)

            # look up the target word index; skip if unknown -
            w_idx = get(vocabulary, word, get(vocabulary, "<unk>", 0));
            if w_idx == 0
                continue
            end

            # define the context window bounds -
            lo = max(1, i - window_size);
            hi = min(length(words), i + window_size);

            # accumulate co-occurrence counts within the window -
            for j in lo:hi
                if j == i
                    continue
                end
                ctx = get(vocabulary, words[j], get(vocabulary, "<unk>", 0));
                if ctx > 0
                    cooccurrence[w_idx, ctx] += 1;
                end
            end
        end
    end

    # return -
    return cooccurrence;
end

"""
    build_pmi_matrices(cooccurrence_matrix) -> Tuple{Array{Float64,2}, Array{Float64,2}}

Compute the Pointwise Mutual Information (PMI) and Positive PMI (PPMI) matrices
from a co-occurrence count matrix.

PMI is defined as `log2(P(w, c) / (P(w) * P(c)))`, where joint and marginal
probabilities are estimated from `cooccurrence_matrix`. Entries with zero joint
count are set to `-Inf`. PPMI clamps negative PMI values to zero. Both output
matrices have the same size as the input (`vocab_size × vocab_size`).
Returns `(zeros, zeros)` matrices if the total co-occurrence count is zero.

### Arguments
- `cooccurrence_matrix::Array{Int64,2}`: symmetric count matrix from `build_cooccurrence_matrix`.

### Returns
- `pmi_matrix::Array{Float64,2}`: PMI matrix of size `vocab_size × vocab_size`.
- `ppmi_matrix::Array{Float64,2}`: PPMI matrix of size `vocab_size × vocab_size`.
"""
function build_pmi_matrices(cooccurrence_matrix::Array{Int64,2})

    # initialize -
    vocab_size = size(cooccurrence_matrix, 1);
    total = sum(cooccurrence_matrix);

    # guard against empty matrix -
    if total == 0
        return zeros(Float64, vocab_size, vocab_size), zeros(Float64, vocab_size, vocab_size)
    end

    # estimate joint and marginal probabilities -
    joint_prob = cooccurrence_matrix ./ total;
    marginal_w = sum(joint_prob, dims=2);
    marginal_c = sum(joint_prob, dims=1);

    # compute PMI for each (word, context) pair -
    pmi_matrix = fill(-Inf, vocab_size, vocab_size);
    for i in 1:vocab_size
        for j in 1:vocab_size
            if joint_prob[i,j] > 0 && marginal_w[i] > 0 && marginal_c[j] > 0
                pmi_matrix[i,j] = log2(joint_prob[i,j] / (marginal_w[i] * marginal_c[j]));
            end
        end
    end

    # compute PPMI by clamping negative values to zero -
    ppmi_matrix = max.(pmi_matrix, 0.0);

    # return -
    return pmi_matrix, ppmi_matrix;
end
