"""
    build_tf_matrix(bow_matrix) -> Array{Float64,2}

Compute the term frequency (TF) matrix from a bag-of-words count matrix.

Each entry `tf[i, j]` is the relative frequency of token `j` in sentence `i`:
`tf[i, j] = bow_matrix[i, j] / sum(bow_matrix[i, :])`. Rows with zero total
count are left as all zeros.

### Arguments
- `bow_matrix::Array{Int64,2}`: count matrix of size `num_sentences × vocab_size`
  produced by `build_bow_matrix`.

### Returns
- `::Array{Float64,2}`: term frequency matrix of size `num_sentences × vocab_size`.
"""
function build_tf_matrix(bow_matrix::Array{Int64,2})::Array{Float64,2}

    # initialize -
    num_sentences, vocab_size = size(bow_matrix);
    tf_matrix = zeros(Float64, num_sentences, vocab_size);

    # compute relative frequency for each sentence -
    for i in 1:num_sentences
        total = sum(bow_matrix[i, :]);
        if total > 0
            tf_matrix[i, :] = bow_matrix[i, :] ./ total;
        end
    end

    # return -
    return tf_matrix;
end

"""
    build_idf_dictionary(bow_matrix, vocabulary, num_sentences) -> Dict{String,Float64}

Compute the inverse document frequency (IDF) for each vocabulary token.

Uses smoothed IDF: `idf(w) = log((num_sentences + 1) / (df(w) + 1))`, where
`df(w)` is the number of sentences in which token `w` appears at least once.
The smoothing prevents division by zero for tokens not seen during fitting.

### Arguments
- `bow_matrix::Array{Int64,2}`: count matrix of size `num_sentences × vocab_size`.
- `vocabulary::Dict{String,Int64}`: mapping from token to 1-based vocabulary index.
- `num_sentences::Int64`: total number of sentences in the corpus.

### Returns
- `::Dict{String,Float64}`: mapping from token string to smoothed IDF value.
"""
function build_idf_dictionary(bow_matrix::Array{Int64,2}, vocabulary::Dict{String,Int64}, num_sentences::Int64)::Dict{String,Float64}

    # initialize -
    idf_dict = Dict{String, Float64}();
    inverse_vocab = Dict{Int64, String}(v => k for (k, v) in vocabulary);
    vocab_size = size(bow_matrix, 2);

    # compute smoothed IDF for each token -
    for j in 1:vocab_size
        df = sum(bow_matrix[:, j] .> 0);
        word = get(inverse_vocab, j, "<unk>");
        idf_dict[word] = log((num_sentences + 1) / (df + 1));
    end

    # return -
    return idf_dict;
end

"""
    build_tfidf_matrix(tf_matrix, idf_dict, inverse_vocabulary) -> Array{Float64,2}

Compute the TF-IDF matrix by element-wise multiplication of TF and IDF values.

Each entry `tfidf[i, j] = tf_matrix[i, j] * idf_dict[token_j]`. Tokens absent
from `idf_dict` are assigned an IDF of 0.

### Arguments
- `tf_matrix::Array{Float64,2}`: term frequency matrix from `build_tf_matrix`.
- `idf_dict::Dict{String,Float64}`: IDF values from `build_idf_dictionary`.
- `inverse_vocabulary::Dict{Int64,String}`: mapping from 1-based index to token string.

### Returns
- `::Array{Float64,2}`: TF-IDF matrix of size `num_sentences × vocab_size`.
"""
function build_tfidf_matrix(tf_matrix::Array{Float64,2}, idf_dict::Dict{String,Float64}, inverse_vocabulary::Dict{Int64,String})::Array{Float64,2}

    # initialize -
    num_sentences, vocab_size = size(tf_matrix);
    tfidf_matrix = zeros(Float64, num_sentences, vocab_size);

    # multiply TF by IDF for each token column -
    for j in 1:vocab_size
        word = get(inverse_vocabulary, j, "<unk>");
        idf = get(idf_dict, word, 0.0);
        tfidf_matrix[:, j] = tf_matrix[:, j] .* idf;
    end

    # return -
    return tfidf_matrix;
end
