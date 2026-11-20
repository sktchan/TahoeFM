# erecon: hide the value of a fraction of genes and predict their per-gene z-scored expression (same masks and targets for ETF and RTF)

# topk tokens: z of each hidden token's own gene is clipped to +-z_clip (low-sd genes in the top-k give z up to ~550); 0 or Inf = no clip
const TOPK_Z_CLIP = 10f0

# target = :z (default, per-gene z) or :raw (the token's own log1p CP10k value, no clip, no standardisation)
struct ERecon <: Objective
    mask_ratio::Float64
    z_clip::Float32
    target::Symbol
end
ERecon(r::Real) = ERecon(Float64(r), TOPK_Z_CLIP, :z)
ERecon(r::Real, clip::Real) = ERecon(Float64(r), Float32(clip), :z)

make_heads(::ERecon, D::Int, H::Int, G::Int) = (reg = reg_head(D, H),)

# z target of each hidden gene (cpu; field z also holds the raw target); ctx.mu_g / ctx.sd_g = train mean / sd per gene
function targets(o::ERecon, ::Input, Xraw, xin, hide::AbstractMatrix{Bool}, ctx)
    idx = findall(vec(hide))
    o.target == :raw && return (idx = idx, z = Float32.(vec(Xraw)[idx]))
    return (idx = idx, z = Float32.(vec((Xraw .- ctx.mu_g) ./ ctx.sd_g)[idx]))
end

# topk tokens: z of each hidden token's own gene (ctx stats over all genes), clipped to +-clip
_zclip(z, clip) = (0 < clip < Inf) ? clamp.(z, -Float32(clip), Float32(clip)) : z
topk_z(v, g, ctx; clip::Real = TOPK_Z_CLIP) = _zclip(Float32.((v .- ctx.mu_g[g]) ./ ctx.sd_g[g]), clip)
function targets(o::ERecon, ::Input, Xraw::TopkInput, xin, hide::AbstractMatrix{Bool}, ctx)
    idx = findall(vec(hide)); g = vec(Xraw.ids)[idx]
    o.target == :raw && return (idx = idx, z = Float32.(vec(Xraw.v)[idx]))
    return (idx = idx, z = topk_z(vec(Xraw.v)[idx], g, ctx; clip = o.z_clip))
end

# per-gene mean of the clipped topk z over train top-k tokens X (genes never in a train top-k: 0); the no-context prediction when the gene id is visible
# raw = true: mean of the raw values instead (genes never in a train top-k: the all-PB train mean)
function topk_gene_mean_z(X::TopkInput, ctx, n_ids::Int; clip::Real = TOPK_Z_CLIP, raw::Bool = false)
    g = vec(X.ids); z = raw ? vec(X.v) : topk_z(vec(X.v), g, ctx; clip); s = zeros(Float64, n_ids); c = zeros(Int, n_ids)
    real = pad_mask(X)   # padded samples: real tokens only
    for (i, (gg, zz)) in enumerate(zip(g, z)); (isnothing(real) || real[i]) || continue; s[gg] += zz; c[gg] += 1; end
    return Float32.(raw ? ifelse.(c .> 0, s ./ max.(c, 1), ctx.mu_g) : s ./ max.(c, 1))
end

function pretext_loss(o::ERecon, input::Input, model, x, hide, t; teacher = nothing)
    h = encode(model.encoder, x, hide, :value)
    pred = vec(model.heads.reg(hidden_tokens(h, t.idx)))
    return mean((pred .- t.z) .^ 2)
end

# predict-zero on the val masks; topk or raw target: predict the gene's train mean target (gene ids are visible), ctx.gene_mse
trivial_loss(o::ERecon, ctx) = (get(ctx, :topk, false) || o.target == :raw) ? ctx.gene_mse : ctx.zero_mse
trivial_alt(o::ERecon, ctx) = (get(ctx, :topk, false) || o.target == :raw) ? ctx.zero_mse : NaN

# topk: MSE on the val masks of the per-gene train mean target
function topk_trivials(o::ERecon, input::Input, Xtr::TopkInput, Xva::TopkInput, xin_va, hide, ctx)
    mg = topk_gene_mean_z(Xtr, ctx, ctx.n_ids; clip = o.z_clip, raw = o.target == :raw); t = targets(o, input, Xva, xin_va, hide, ctx)
    return (gene_mse = Float64(mean(abs2, t.z .- mg[vec(Xva.ids)[t.idx]])),)
end
# panel, raw target: MSE on the val masks of the per-gene train mean (ctx.mu_g, from the train columns)
topk_trivials(o::ERecon, input::Input, Xtr::AbstractMatrix, Xva::AbstractMatrix, xin_va, hide, ctx) =
    o.target == :raw ? (gene_mse = Float64(mean(abs2, (Xva .- ctx.mu_g)[hide])),) : (;)
