# This file handles the dataset side of the lab: downloading the corpus,
# building a character-level vocabulary, and sampling supervised training
# examples for next-token prediction.

"""
    download_shakespeare(datapath) -> String

Download the Tiny Shakespeare corpus (~1.1 MB) used by Karpathy's char-rnn
demos. Caches the file as `datapath/input.txt` and returns the path. If the
file is already present, returns it without redownloading.
"""
function download_shakespeare(datapath::String)::String
    txtfile = joinpath(datapath, "input.txt")
    # Reuse the local copy if a previous run already downloaded the corpus.
    isfile(txtfile) && return txtfile

    !isdir(datapath) && mkpath(datapath)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    @info "Downloading Tiny Shakespeare (~1.1 MB) from char-rnn..." url=url
    Downloads.download(url, txtfile; timeout = 120.0)
    @info "Download complete" path=txtfile size=filesize(txtfile)
    return txtfile
end

"""
    CharVocabulary

Lookup tables for converting between characters and integer token ids. Holds
both directions of the map plus the vocabulary size.
"""
struct CharVocabulary
    char_to_id::Dict{Char, Int}  # encode: character -> token id
    id_to_char::Dict{Int, Char}  # decode: token id -> character
    vocab_size::Int              # number of unique characters in the corpus
end

"""
    build_vocab(text::String) -> CharVocabulary

Build a character vocabulary from a text corpus. Each unique character in the
text gets a 1-based integer id. Ids are assigned in sorted-character order so
the mapping is deterministic.
"""
function build_vocab(text::String)::CharVocabulary
    # Sorting makes the mapping deterministic, which is helpful when students
    # compare outputs across runs or between different machines.
    chars = sort(unique(collect(text)))
    char_to_id = Dict{Char, Int}(c => i for (i, c) in enumerate(chars))
    id_to_char = Dict{Int, Char}(i => c for (i, c) in enumerate(chars))
    return CharVocabulary(char_to_id, id_to_char, length(chars))
end

"""
    encode(text::String, vocab::CharVocabulary) -> Vector{Int}

Convert a text string to a vector of integer token ids using `vocab`. Any
character that is not in the vocabulary will throw a `KeyError`.
"""
function encode(text::String, vocab::CharVocabulary)::Vector{Int}
    # This is a pure table lookup; it does not do any padding or unknown-token
    # handling because the lab uses a closed character set from one corpus.
    return [vocab.char_to_id[c] for c in text]
end

"""
    decode(ids::AbstractVector{Int}, vocab::CharVocabulary) -> String

Convert a vector of integer token ids back into a text string using `vocab`.
"""
function decode(ids::AbstractVector{<:Integer}, vocab::CharVocabulary)::String
    # Convert back through `id_to_char` and rebuild a Julia `String`.
    return String([vocab.id_to_char[Int(i)] for i in ids])
end

"""
    sample_batch(data::Vector{Int}, batchsize::Int, ctx_len::Int; rng=Random.default_rng())
        -> Tuple{Matrix{Int}, Matrix{Int}}

Sample `batchsize` random contiguous chunks of length `ctx_len + 1` from the
encoded corpus `data`. Returns `(X, Y)` where each is a `(ctx_len, batchsize)`
matrix of token ids:

* `X[t, b]` is the input token at position `t` of example `b`
* `Y[t, b]` is the next-token target (i.e. `X[t+1, b]`)

This pairing is what next-token prediction needs: at every position the model
sees `X[1:t, b]` and is asked to predict `Y[t, b]`.
"""
function sample_batch(data::AbstractVector{<:Integer}, batchsize::Int, ctx_len::Int;
                       rng::AbstractRNG = Random.default_rng())
    n = length(data)
    @assert n > ctx_len "corpus is shorter than the requested context length"
    # Each start index defines one contiguous training window of length
    # `ctx_len + 1`: the first `ctx_len` tokens become inputs and the shifted
    # sequence becomes targets.
    starts = rand(rng, 1:(n - ctx_len), batchsize)
    X = Matrix{Int}(undef, ctx_len, batchsize)
    Y = Matrix{Int}(undef, ctx_len, batchsize)
    for b in 1:batchsize
        s = starts[b]
        # Fill one column per training example so the output layout matches the
        # `(time, batch)` convention used throughout the model code.
        @inbounds for t in 1:ctx_len
            X[t, b] = data[s + t - 1]
            Y[t, b] = data[s + t]
        end
    end
    return X, Y
end
