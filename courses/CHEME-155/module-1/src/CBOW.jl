"""
    build_cbow_pairs(sentences, vocabulary; window_size=2) -> Vector

Build context–target training pairs from a tokenized corpus for CBOW training.

For each target word at position `i` in a sentence, the context is the average of
one-hot vocabulary vectors for all words within `window_size` positions on either
side of `i`. Returns a vector of `(context_avg, target)` tuples where both
`context_avg` and `target` are `Float64` vectors of length `|vocabulary|`.

### Arguments
- `sentences::Array{String,1}`: raw sentences; each sentence is lowercased and split on whitespace.
- `vocabulary::Dict{String,Int64}`: mapping from token to 1-based vocabulary index.
- `window_size::Int64`: half-width of the context window (default: 2).

### Returns
- `::Vector`: a vector of `(context_avg, target)` tuples, one per valid target word.
"""
function build_cbow_pairs(sentences::Array{String,1}, vocabulary::Dict{String,Int64}; window_size::Int64=2)

    # initialize -
    vocab_size = length(vocabulary);
    training_pairs = [];

    # main loop: iterate over sentences -
    for sentence in sentences

        # tokenize and get sentence length -
        words = split(lowercase(sentence));
        n = length(words);

        for i in 1:n

            # look up target word index; skip if unknown -
            target_word = get(vocabulary, words[i], get(vocabulary, "<unk>", 0));
            if target_word == 0
                continue
            end

            # build one-hot target vector -
            target = zeros(Float64, vocab_size);
            target[target_word] = 1.0;

            # accumulate context word vectors in the window -
            context_sum = zeros(Float64, vocab_size);
            count = 0;
            for j in max(1, i - window_size):min(n, i + window_size)
                if j == i
                    continue
                end
                ctx_word = get(vocabulary, words[j], get(vocabulary, "<unk>", 0));
                if ctx_word > 0
                    context_sum[ctx_word] += 1.0;
                    count += 1;
                end
            end

            # store the averaged context vector and target -
            if count > 0
                push!(training_pairs, (context_sum ./ count, target));
            end
        end
    end

    # return -
    return training_pairs;
end

"""
    train_cbow(training_pairs, vocab_size; d_h=5, eta=0.01, num_epochs=500, verbose=true, print_every=100) -> Tuple

Train a CBOW model on `training_pairs` using stochastic gradient descent.

The model has a single hidden layer of size `d_h`. The input weight matrix
`W1 ∈ ℝ^{d_h × vocab_size}` maps context vectors to the hidden layer, and
`W2 ∈ ℝ^{vocab_size × d_h}` maps the hidden layer to the output distribution.
Training minimizes cross-entropy loss with a softmax output.

### Arguments
- `training_pairs`: vector of `(context_avg, target)` tuples produced by `build_cbow_pairs`.
- `vocab_size::Int64`: number of tokens in the vocabulary.
- `d_h::Int64`: embedding (hidden layer) dimension (default: 5).
- `eta::Float64`: SGD learning rate (default: 0.01).
- `num_epochs::Int64`: number of full passes over `training_pairs` (default: 500).
- `verbose::Bool`: print loss every `print_every` epochs when `true` (default: true).
- `print_every::Int64`: print interval in epochs when `verbose=true` (default: 100).

### Returns
- `W1::Matrix{Float64}`: input weight matrix of size `d_h × vocab_size`; columns are word embeddings.
- `W2::Matrix{Float64}`: output weight matrix of size `vocab_size × d_h`.
- `loss_history::Vector{Float64}`: average cross-entropy loss per epoch.
"""
function train_cbow(training_pairs, vocab_size::Int64;
    d_h::Int64=5, eta::Float64=0.01, num_epochs::Int64=500, verbose::Bool=true, print_every::Int64=100)

    # initialize weight matrices with small random values -
    W1 = randn(d_h, vocab_size) * 0.01;
    W2 = randn(vocab_size, d_h) * 0.01;
    loss_history = zeros(Float64, num_epochs);

    # main training loop -
    for epoch in 1:num_epochs

        total_loss = 0.0;

        for (x, y) in training_pairs

            # forward pass: hidden layer and output scores -
            h = W1 * x;
            u = W2 * h;

            # stable softmax -
            exp_u = exp.(u .- maximum(u));
            y_hat = exp_u ./ sum(exp_u);

            # cross-entropy loss -
            loss = -sum(y .* log.(y_hat .+ 1e-10));
            total_loss += loss;

            # backward pass: output layer gradient -
            delta_u = y_hat .- y;
            dW2 = delta_u * h';

            # backward pass: hidden layer gradient -
            delta_h = W2' * delta_u;
            dW1 = delta_h * x';

            # SGD weight update -
            W2 .-= eta .* dW2;
            W1 .-= eta .* dW1;
        end

        # record average loss for this epoch -
        loss_history[epoch] = total_loss / length(training_pairs);

        # print progress -
        if verbose && (epoch % print_every == 0 || epoch == 1)
            println("Epoch $(lpad(epoch, ndigits(num_epochs)))/$(num_epochs)  loss = $(round(loss_history[epoch]; digits=4))");
        end
    end

    # return -
    return W1, W2, loss_history;
end

"""
    nearest_neighbors(W1, vocabulary, word; top_k=5) -> Vector{Tuple{String,Float64}}

Return the `top_k` vocabulary words most similar to `word` by cosine similarity.

Similarity is computed between the embedding of `word` (column `vocabulary[word]`
of `W1`) and every other column of `W1`. The query word itself is excluded from
the results. Returns an empty vector if `word` is not in `vocabulary`.

### Arguments
- `W1::Matrix{Float64}`: input weight matrix of size `d_h × vocab_size` from `train_cbow`.
- `vocabulary::Dict{String,Int64}`: mapping from token to 1-based vocabulary index.
- `word::String`: query word.
- `top_k::Int`: number of nearest neighbors to return (default: 5).

### Returns
- `::Vector{Tuple{String,Float64}}`: list of `(word, cosine_similarity)` pairs, sorted descending.
"""
function nearest_neighbors(W1::Matrix{Float64}, vocabulary::Dict{String,Int64},
                            word::String; top_k::Int=5)

    # look up the query word index; return empty if not found -
    idx = get(vocabulary, word, 0);
    if idx == 0
        return Tuple{String,Float64}[]
    end

    # get the query embedding and its norm -
    v = W1[:, idx];
    v_norm = norm(v);

    # compute cosine similarity to every other word in the vocabulary -
    sims = Vector{Tuple{String,Float64}}();
    for (w, i) in vocabulary
        if i == idx
            continue
        end
        vi = W1[:, i];
        ni = norm(vi);
        sim = (v_norm > 1e-10 && ni > 1e-10) ? dot(v, vi) / (v_norm * ni) : 0.0;
        push!(sims, (w, sim));
    end

    # sort by similarity descending and return top_k -
    sort!(sims, by = x -> x[2], rev=true);
    return [(w, round(s; digits=4)) for (w, s) in sims[1:min(top_k, length(sims))]];
end

"""
    solve_analogy(W1, vocabulary, word_a, word_b, word_c; top_k=5, exclude_inputs=true) -> Vector{Tuple{String,Float64}}

Solve the word analogy `word_a : word_b :: word_c : ?` using the vector offset method.

Computes the target vector `v_b - v_a + v_c` and returns the `top_k` vocabulary
words whose embeddings have the highest cosine similarity to that target. The three
input words are excluded from the candidate set when `exclude_inputs=true`.
Returns an empty vector if any input word is missing from `vocabulary`.

### Arguments
- `W1::Matrix{Float64}`: input weight matrix of size `d_h × vocab_size` from `train_cbow`.
- `vocabulary::Dict{String,Int64}`: mapping from token to 1-based vocabulary index.
- `word_a::String`, `word_b::String`, `word_c::String`: the three input words of the analogy.
- `top_k::Int`: number of candidate answers to return (default: 5).
- `exclude_inputs::Bool`: whether to exclude `word_a`, `word_b`, `word_c` from results (default: true).

### Returns
- `::Vector{Tuple{String,Float64}}`: list of `(word, cosine_similarity)` pairs, sorted descending.
"""
function solve_analogy(W1::Matrix{Float64}, vocabulary::Dict{String,Int64},
                       word_a::String, word_b::String, word_c::String;
                       top_k::Int=5, exclude_inputs::Bool=true)

    # check that all three words are in the vocabulary -
    for w in [word_a, word_b, word_c]
        if !haskey(vocabulary, w)
            return Tuple{String,Float64}[]
        end
    end

    # retrieve embeddings for each input word using the full W1 columns -
    v_a = W1[:, vocabulary[word_a]];
    v_b = W1[:, vocabulary[word_b]];
    v_c = W1[:, vocabulary[word_c]];

    # compute the analogy target vector: v_b - v_a + v_c -
    v_target = v_b .- v_a .+ v_c;
    v_target_norm = norm(v_target);

    # set of input words to exclude from results -
    exclude_set = Set([word_a, word_b, word_c]);

    # compute cosine similarity of every candidate word's embedding to the target -
    sims = Vector{Tuple{String,Float64}}();
    for (w, i) in vocabulary
        if exclude_inputs && w ∈ exclude_set
            continue
        end
        vi = W1[:, i];
        ni = norm(vi);
        sim = (v_target_norm > 1e-10 && ni > 1e-10) ? dot(v_target, vi) / (v_target_norm * ni) : 0.0;
        push!(sims, (w, sim));
    end

    # sort by similarity descending and return top_k -
    sort!(sims, by = x -> x[2], rev=true);
    return [(w, round(s; digits=4)) for (w, s) in sims[1:min(top_k, length(sims))]];
end
