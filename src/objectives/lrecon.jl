# lrecon: hide values of a fraction of genes; predict an EMA teacher's normalized, layer-averaged states of those tokens (cosine loss)

struct LRecon <: Objective
    mask_ratio::Float64
    tau::Float32
    centre::Base.RefValue{Any}   # topk: running per-gene means of the teacher states (GeneCentre), set by pretrain; panel: unused
    topk_norm::Symbol            # topk target: :gene (gene centre, default) or :slot (panel norm per rank slot; the reviewed shortcut, ablation only)
end
function LRecon(mask_ratio::Real, tau::Real; topk_norm = :gene)
    Symbol(topk_norm) in (:gene, :slot) || error("lrecon_topk_norm must be gene or slot, got $topk_norm")
    return LRecon(Float64(mask_ratio), Float32(tau), Ref{Any}(nothing), Symbol(topk_norm))
end

uses_teacher(::LRecon) = true
make_heads(::LRecon, D::Int, H::Int, G::Int) = (dec = latent_head(D, H),)

function targets(::LRecon, ::Input, Xraw, xin, hide::AbstractMatrix{Bool}, ctx)
    return (idx = findall(vec(hide)),)
end

# per-sample token-mean removal, then batch z per (feature, gene) (panel tokens)
function _dbl(h)
    t = h .- mean(h, dims=2)
    return (t .- mean(t, dims=3)) ./ (std(t, dims=3) .+ 1f-6)
end

# layer-averaged normalised teacher states (D x G x B); the per-slot batch norm has no padded form
function teacher_targets(teacher::Encoder, x)
    isnothing(pad_len(x)) || error("lrecon_topk_norm = slot is not built for padded tokens (use gene)")
    hs = token_layers(teacher, x)
    return sum(_dbl, hs) ./ Float32(length(hs))
end

# topk tokens: every sample has its own genes, so the per-gene mean comes from a running buffer; per layer: per-sample mean removal,
# minus the running mean of the token's gene (D x n_ids, train batches only), then one sd per feature over the batch
# (no second per-sample mean removal: it re-adds a gene-set offset that a gene-id lookup predicts, logs/port_fix/s1_proto3)
struct GeneCentre{A<:AbstractMatrix,V<:AbstractVector}
    mean::Vector{A}      # per layer state (h0..hL), D x n_ids
    count::V             # decayed token count per gene
    momentum::Float32    # per update (batch)
end
const LRECON_MOMENTUM = 0.99f0   # per-batch decay of the per-gene means (~100 batches; the teacher's EMA moves on ~2000 steps)
const LRECON_WARM = 2000         # train PBs that fill the per-gene means before step 0

_scent(h) = h .- mean(h, dims=2)
function _gene_dbl(h, B, ids)
    u = _scent(h) .- NNlib.gather(B, ids)
    return u ./ (std(u, dims=(2, 3)) .+ 1f-6)
end
# padded tokens (w = pool weights, 1 / len on real tokens, 0 on padding): token mean and batch sd over the real tokens only
_scent(h, ::Nothing) = _scent(h)
_scent(h, w) = h .- sum(h .* w, dims=2)
_gene_dbl(h, B, ids, ::Nothing) = _gene_dbl(h, B, ids)
function _gene_dbl(h, B, ids, w)
    u = _scent(h, w) .- NNlib.gather(B, ids); r = Float32.(w .> 0); n = sum(r)
    m = sum(u .* r, dims=(2, 3)) ./ n
    sd = sqrt.(sum(abs2.((u .- m) .* r), dims=(2, 3)) ./ (n - 1))
    return u ./ (sd .+ 1f-6)
end
teacher_targets(teacher::Encoder, x::TopkInput, gc::GeneCentre) =
    (hs = token_layers(teacher, x); w = pool_weights(x); sum(_gene_dbl(h, B, x.ids, w) for (h, B) in zip(hs, gc.mean)) ./ Float32(length(hs)))

# fresh (empty) gene centre for an encoder with n_ids gene rows, on the encoder's device
function init_centre!(o::LRecon, teacher::Encoder, n_ids::Int; momentum::Real = LRECON_MOMENTUM)
    D = embed_dim(teacher); L = length(teacher.layers.layers) + 1; z(dims...) = dev(zeros(Float32, dims...))
    o.centre[] = GeneCentre([z(D, n_ids) for _ in 1:L], z(n_ids), Float32(momentum))
    return o
end

# decayed running mean per gene: count <- m count + n, mean += n / count x (batch mean - mean); genes absent from the batch keep their mean
# w = pool weights of padded tokens (padding counts for no gene), nothing = every token is real
function update_centre!(gc::GeneCentre, hs, ids, w = nothing)
    iv = vec(ids); one = isnothing(w) ? fill!(similar(gc.count, length(iv)), 1f0) : vec(Float32.(w .> 0))
    n = NNlib.scatter(+, one, iv; dstsize=size(gc.count))
    gc.count .= gc.momentum .* gc.count .+ n
    a = n ./ max.(gc.count, 1f-12)
    for (B, h) in zip(gc.mean, hs)
        sc = reshape(_scent(h, w), size(h, 1), :)
        isnothing(w) || (sc = sc .* one')
        S = NNlib.scatter(+, sc, iv; dstsize=size(B))
        B .+= a' .* (S ./ max.(n, 1f0)' .- B)
    end
    return gc
end
update_centre!(o::LRecon, teacher::Encoder, x::TopkInput) = (update_centre!(o.centre[], token_layers(teacher, x), x.ids, pool_weights(x)); o)

# topk: teacher targets of the hidden tokens computed once with the gene centre (update = true: training batches update it first)
function prepare_targets(o::LRecon, teacher, x::TopkInput, t; update::Bool = false)
    o.topk_norm == :slot && return merge(t, (Y = Flux.ChainRulesCore.ignore_derivatives(() -> hidden_tokens(teacher_targets(teacher, x), t.idx)),))
    isnothing(o.centre[]) && error("lrecon with topk tokens needs a gene centre (init_centre!)")
    Y = Flux.ChainRulesCore.ignore_derivatives() do
        update && update_centre!(o, teacher, x)
        hidden_tokens(teacher_targets(teacher, x, o.centre[]), t.idx)
    end
    return merge(t, (Y = Y,))
end

# a gene centre estimated with a frozen encoder on model inputs X (topk), batches of bs
function fresh_centre(enc::Encoder, X::TopkInput; bs::Int = 100)
    e = eval_device(enc); o = init_centre!(LRecon(0.0, 1f0), e, n_genes(e))
    for c in Iterators.partition(1:size(X, 2), bs); update_centre!(o, e, dev(X[:, c])); end
    return o.centre[]
end

# gene centre for a checkpoint's extra (cpu arrays), and back
centre_extra(::Objective) = Dict{String,Any}()
centre_extra(o::LRecon) = isnothing(o.centre[]) ? Dict{String,Any}() :
    Dict{String,Any}("gene_centre" => [Array(B) for B in o.centre[].mean], "gene_centre_count" => Array(o.centre[].count), "gene_centre_momentum" => o.centre[].momentum)
centre_from_extra(ex) = haskey(ex, "gene_centre") ? GeneCentre(map(dev, ex["gene_centre"]), dev(ex["gene_centre_count"]), Float32(ex["gene_centre_momentum"])) : nothing

_cos_loss(a, b) = mean(1f0 .- sum(a .* b, dims=1) ./ (sqrt.(sum(a .^ 2, dims=1)) .* sqrt.(sum(b .^ 2, dims=1)) .+ 1f-6))

# hidden-token targets: prepared ones (topk), else the panel targets computed here (as LincsFM)
lrecon_Y(teacher, x, t) = haskey(t, :Y) ? t.Y : Flux.ChainRulesCore.ignore_derivatives(() -> hidden_tokens(teacher_targets(teacher, x), t.idx))
lrecon_Y(teacher, x::TopkInput, t) = haskey(t, :Y) ? t.Y : error("lrecon with topk tokens: targets come from prepare_targets")

function pretext_loss(o::LRecon, input::Input, model, x, hide, t; teacher = nothing)
    Y = lrecon_Y(teacher, x, t)
    h = encode(model.encoder, x, hide, :value)
    return _cos_loss(model.heads.dec(hidden_tokens(h, t.idx)), Y)
end

trivial_loss(::LRecon, ctx) = 1.0   # cosine loss of an unrelated direction

# teacher <- tau * teacher + (1 - tau) * student
function ema_update!(teacher::Encoder, student::Encoder, tau::Float32)
    for (pt, ps) in zip(Flux.trainables(teacher), Flux.trainables(student))
        @. pt = tau * pt + (1f0 - tau) * ps
    end
    return teacher
end
