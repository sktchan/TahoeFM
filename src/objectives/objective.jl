# pretraining objectives: each defines its corruption, target, head and loss; one training loop (train/pretrain.jl)

abstract type Objective end

# defaults for every objective (overridden by the objective or the config)
init_scheme(::Objective) = :tok_norm
hide_what(::Objective, ::Input) = :value
uses_teacher(::Objective) = false
trivial_alt(::Objective, ctx) = NaN   # second logged trivial (topk: the old uniform / predict-zero value)
# extra ctx fields of the topk trivials, from the train inputs Xtr and the fixed val inputs / masks (none by default)
topk_trivials(::Objective, input::Input, Xtr, Xva, xin_va, hide, ctx) = (;)
# targets that need the teacher (lrecon topk: per-gene centring; update = true also updates the running means, training only)
prepare_targets(::Objective, teacher, x, t; update::Bool = false) = t

# hidden genes: k = ceil(G * ratio) per column, one randperm per column; true = hidden
function hide_mask(G::Int, B::Int, ratio::Real, rng::AbstractRNG)
    k = ceil(Int, G * ratio); m = falses(G, B)
    for b in 1:B
        @inbounds for g in view(randperm(rng, G), 1:k)
            m[g, b] = true
        end
    end
    return m
end

# padded tokens: k = ceil(len * ratio) of each column's real tokens (same draws as above when len = G)
function hide_mask(G::Int, B::Int, ratio::Real, rng::AbstractRNG, len::AbstractVector{<:Integer})
    m = falses(G, B)
    for b in 1:B
        n = Int(len[b]); k = ceil(Int, n * ratio)
        @inbounds for g in view(randperm(rng, n), 1:k)
            m[g, b] = true
        end
    end
    return m
end
hide_mask(G::Int, B::Int, ratio::Real, rng::AbstractRNG, ::Nothing) = hide_mask(G, B, ratio, rng)

# gather the hidden tokens: D x (number hidden)
hidden_tokens(h, idx) = reshape(h, size(h, 1), :)[:, idx]

function objective_from(cfg::Dict)
    name = cfg["objective"]
    r = Float64(cfg["mask_ratio"])
    name == "mlm" && return MLM(r)
    if name == "erecon"
        tg = Symbol(get(cfg, "erecon_target", "z")); tg in (:z, :raw) || error("erecon_target must be z or raw, got $tg")
        return ERecon(r, Float32(get(cfg, "z_clip", TOPK_Z_CLIP)), tg)
    end
    name == "lrecon" && return LRecon(r, Float32(cfg["ema_decay"]); topk_norm = get(cfg, "lrecon_topk_norm", "gene"))
    error("unknown objective $name (mlm, erecon, lrecon)")
end
