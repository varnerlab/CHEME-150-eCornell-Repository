function build_noise_distribution(sentences::Array{String,1}, vocabulary::Dict{String,Int64})::Array{Float64,1}
    vocab_size = length(vocabulary)
    counts = zeros(Float64, vocab_size)
    for sentence in sentences
        for word in split(lowercase(sentence))
            idx = get(vocabulary, word, get(vocabulary, "<unk>", 0))
            if idx > 0
                counts[idx] += 1.0
            end
        end
    end
    smoothed = counts .^ 0.75
    return smoothed ./ sum(smoothed)
end

function train_skipgram_ns(training_pairs, vocab_size::Int64, noise_distribution::Array{Float64,1};
    d_h::Int64=5, eta::Float64=0.01, num_epochs::Int64=500, k::Int64=5,
    verbose::Bool=true, print_every::Int64=10)

    W1 = randn(d_h, vocab_size) * 0.01
    W2 = randn(vocab_size, d_h) * 0.01
    loss_history = zeros(Float64, num_epochs)
    for epoch in 1:num_epochs
        total_loss = 0.0
        for (x, y) in training_pairs
            target_idx = argmax(x)
            context_idx = argmax(y)
            neg_indices = [StatsBase.sample(1:vocab_size, Weights(noise_distribution)) for _ in 1:k]
            h = W1[:, target_idx]
            score_pos = dot(W2[context_idx, :], h)
            loss = -log(sigmoid(score_pos) + 1e-10)
            grad_pos = (sigmoid(score_pos) - 1.0)
            W2[context_idx, :] .-= eta .* grad_pos .* h
            dh = grad_pos .* W2[context_idx, :]
            for neg_idx in neg_indices
                score_neg = dot(W2[neg_idx, :], h)
                loss += -log(sigmoid(-score_neg) + 1e-10)
                grad_neg = sigmoid(score_neg)
                W2[neg_idx, :] .-= eta .* grad_neg .* h
                dh .+= grad_neg .* W2[neg_idx, :]
            end
            W1[:, target_idx] .-= eta .* dh
            total_loss += loss
        end
        loss_history[epoch] = total_loss / length(training_pairs)
        if verbose && (epoch % print_every == 0 || epoch == 1)
            println("Epoch $(lpad(epoch, ndigits(num_epochs)))/$(num_epochs)  loss = $(round(loss_history[epoch]; digits=4))");
        end
    end
    return W1, W2, loss_history
end
