# finetune classifier: encoder (pretraining tokens) + mean pool + head

struct Classifier{E<:Encoder,H}
    encoder::E
    head::H
end
Flux.@layer Classifier

Classifier(enc::Encoder, n_classes::Int; hidden_dim::Int = 128, drop_prob::Float64 = 0.18) =
    Classifier(enc, ft_head(embed_dim(enc), hidden_dim, n_classes, drop_prob))

features(m::Classifier, x) = pooled(m.encoder, x)
(m::Classifier)(x) = m.head(features(m, x))

# head_init: rescale the first head layer so it sees z-scored pooled features at step 0
function head_init!(m::Classifier, batches)
    Flux.testmode!(m)
    E = reduce(hcat, [Array(features(m, x)) for x in batches])
    Flux.trainmode!(m)
    mu = mean(E, dims=2); sd = std(E, dims=2); sd[sd .< 1f-6] .= 1f0
    d1 = m.head.layers[1]
    Wn = Array(d1.weight) ./ permutedims(sd)
    copyto!(d1.weight, Wn); copyto!(d1.bias, Array(d1.bias) .- Wn * vec(mu))
    return median(vec(sd))
end
