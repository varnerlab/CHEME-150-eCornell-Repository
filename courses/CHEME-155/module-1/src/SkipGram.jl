function build_skipgram_pairs(sentences::Array{String,1}, vocabulary::Dict{String,Int64}; window_size::Int64=2)
    vocab_size = length(vocabulary)
    training_pairs = []
    for sentence in sentences
        words = split(lowercase(sentence))
        n = length(words)
        for i in 1:n
            target_word = get(vocabulary, words[i], get(vocabulary, "<unk>", 0))
            if target_word == 0; continue; end
            target = zeros(Float64, vocab_size)
            target[target_word] = 1.0
            for j in max(1,i-window_size):min(n,i+window_size)
                if j == i; continue; end
                ctx_word = get(vocabulary, words[j], get(vocabulary, "<unk>", 0))
                if ctx_word > 0
                    context = zeros(Float64, vocab_size)
                    context[ctx_word] = 1.0
                    push!(training_pairs, (target, context))
                end
            end
        end
    end
    return training_pairs
end

function train_skipgram_softmax(training_pairs, vocab_size::Int64;
    d_h::Int64=5, eta::Float64=0.01, num_epochs::Int64=500,
    verbose::Bool=true, print_every::Int64=10)

    W1 = randn(d_h, vocab_size) * 0.01
    W2 = randn(vocab_size, d_h) * 0.01
    loss_history = zeros(Float64, num_epochs)
    for epoch in 1:num_epochs
        total_loss = 0.0
        for (x, y) in training_pairs
            h = W1 * x
            u = W2 * h
            exp_u = exp.(u .- maximum(u))
            y_hat = exp_u ./ sum(exp_u)
            loss = -sum(y .* log.(y_hat .+ 1e-10))
            total_loss += loss
            delta_u = y_hat .- y
            W2 .-= eta .* (delta_u * h')
            W1 .-= eta .* (W2' * delta_u * x')
        end
        loss_history[epoch] = total_loss / length(training_pairs)
        if verbose && (epoch % print_every == 0 || epoch == 1)
            println("Epoch $(lpad(epoch, ndigits(num_epochs)))/$(num_epochs)  loss = $(round(loss_history[epoch]; digits=4))");
        end
    end
    return W1, W2, loss_history
end
