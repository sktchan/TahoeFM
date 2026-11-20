# heads and pooling

# pretext heads (as the original pretrain models)
class_head(d::Int, n::Int) = Flux.Chain(Flux.Dense(d => d, gelu), Flux.LayerNorm(d), Flux.Dense(d => n))
reg_head(d::Int, h::Int) = Flux.Chain(Flux.Dense(d => h, gelu), Flux.LayerNorm(h), Flux.Dense(h => 1))
latent_head(d::Int, h::Int) = Flux.Chain(Flux.Dense(d => h, gelu), Flux.LayerNorm(h), Flux.Dense(h => d))

# finetune classification head
ft_head(d::Int, h::Int, n::Int, p::Float64) = Flux.Chain(Flux.Dense(d => h, gelu), Flux.Dropout(p), Flux.Dense(h => n))

meanpool(h) = dropdims(mean(h, dims=2), dims=2)
# padded tokens: w = pool weights (1 x K x B, 1 / len on real tokens); nothing = plain mean
meanpool(h, ::Nothing) = meanpool(h)
meanpool(h, w) = dropdims(sum(h .* w, dims=2), dims=2)

