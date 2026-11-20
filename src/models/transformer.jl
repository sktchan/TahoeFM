# transformer block, pre-LN (default) or post-LN (post = true, as BERT); same math as Flux.MultiHeadAttention, fused softmax from fastattn.jl

struct Transf{A<:Flux.MultiHeadAttention, D<:Flux.Dropout, N<:Flux.LayerNorm, M<:Flux.Chain}
    mha::A
    att_dropout::D
    att_norm::N
    mlp::M
    mlp_norm::N
    post::Bool
end
Flux.@layer Transf

function Transf(embed_dim::Int, hidden_dim::Int; n_heads::Int, dropout_prob::Float64, post::Bool = false)
    mha = Flux.MultiHeadAttention((embed_dim, embed_dim, embed_dim) => (embed_dim, embed_dim) => embed_dim,
                                  nheads=n_heads, dropout_prob=dropout_prob)
    mlp = Flux.Chain(Flux.Dense(embed_dim => hidden_dim, gelu), Flux.Dropout(dropout_prob),
                     Flux.Dense(hidden_dim => embed_dim), Flux.Dropout(dropout_prob))
    return Transf(mha, Flux.Dropout(dropout_prob), Flux.LayerNorm(embed_dim), mlp, Flux.LayerNorm(embed_dim), post)
end

# mask = attention mask (K x 1 x 1 x B Bool, true = a real key) for padded tokens, nothing = attend to every token
function (tf::Transf)(x, mask = nothing)
    D, L, B = size(x)
    if tf.post
        r = tf.att_norm(x + tf.att_dropout(attend(tf.mha, x, mask)))
        return tf.mlp_norm(r + reshape(tf.mlp(reshape(r, D, L * B)), D, L, B))
    end
    r = x + tf.att_dropout(attend(tf.mha, tf.att_norm(x), mask))
    return r + reshape(tf.mlp(reshape(tf.mlp_norm(r), D, L * B)), D, L, B)
end

# dropout layers use the CUDA rng on the gpu
gpu_dropout(m) = fmap(m; exclude = x -> x isa Flux.Dropout) do x
    x isa Flux.Dropout ? Flux.Dropout(x.p, x.dims, x.active, CUDA.default_rng()) : x
end

# set every Dropout to p
set_dropout(m, p::Real) = fmap(m; exclude = x -> x isa Flux.Dropout) do x
    x isa Flux.Dropout ? Flux.Dropout(Float64(p), x.dims, x.active, x.rng) : x
end

# Zygote >= 0.7.12 evaluates gpu masks correctly inside withgradient (older versions do not)
const _AD_OK = Ref(false)
function check_ad_masks()
    _AD_OK[] && return true
    CUDA.functional() || return true
    y = CuArray(Float32[-100, 1.5, -100, 2.5])
    n = Flux.withgradient(v -> sum(v) * 0f0 + Float32(sum(y .!= -100f0)), CuArray([1f0]))[1]
    n == 2f0 || error("AD mask check failed: upgrade Zygote to >= 0.7.12")
    _AD_OK[] = true
end

to_device(m) = CUDA.functional() ? (check_ad_masks(); gpu_dropout(cu(m))) : m
dev(x) = CUDA.functional() ? cu(x) : x
