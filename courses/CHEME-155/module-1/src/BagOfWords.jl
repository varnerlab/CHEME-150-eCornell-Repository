"""
    build_bow_matrix(sentences, vocabulary) -> Array{Int64,2}

Build a bag-of-words count matrix from a corpus.

Each row corresponds to a sentence and each column to a vocabulary token.
Entry `[i, j]` is the count of token `j` in sentence `i`. Each sentence is
lowercased, split on whitespace, and wrapped with `<bos>` and `<eos>` tokens
before counting. Unknown tokens are mapped to `<unk>` if present in `vocabulary`.

### Arguments
- `sentences::Array{String,1}`: raw sentences.
- `vocabulary::Dict{String,Int64}`: mapping from token to 1-based vocabulary index.

### Returns
- `::Array{Int64,2}`: count matrix of size `num_sentences × vocab_size`.
"""
function build_bow_matrix(sentences::Array{String,1}, vocabulary::Dict{String,Int64})::Array{Int64,2}

    # initialize -
    num_sentences = length(sentences);
    vocab_size = length(vocabulary);
    bow_matrix = zeros(Int64, num_sentences, vocab_size);

    # main loop: count tokens in each sentence -
    for (i, sentence) in enumerate(sentences)

        # tokenize, lowercase, and add boundary tokens -
        words = split(lowercase(sentence));
        augmented = ["<bos>"; words; "<eos>"];

        # accumulate counts -
        for word in augmented
            idx = get(vocabulary, word, get(vocabulary, "<unk>", 0));
            if idx > 0
                bow_matrix[i, idx] += 1;
            end
        end
    end

    # return -
    return bow_matrix;
end

"""
    hashing_vectorizer(features; length=10) -> Array{Int64,1}

Map a list of string features to a fixed-length count vector using the hashing trick.

Each feature string is hashed and mapped to an index in `[1, length]` via `mod1(abs(hash(f)), length)`.
The value at each index is the number of features that hash to that index.
Collisions (multiple features mapping to the same index) are summed.

### Arguments
- `features::Array{String,1}`: list of feature strings to vectorize.
- `length::Int64`: output vector length (default: 10).

### Returns
- `::Array{Int64,1}`: count vector of size `length`.
"""
function hashing_vectorizer(features::Array{String,1}; length::Int64=10)::Array{Int64,1}

    # initialize -
    v = zeros(Int64, length);

    # hash each feature and accumulate counts -
    for f in features
        idx = mod1(abs(hash(f)), length);
        v[idx] += 1;
    end

    # return -
    return v;
end
