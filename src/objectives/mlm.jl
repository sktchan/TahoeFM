# mlm: ETF hides the value and predicts the gene's rank; RTF hides the gene and predicts which gene it is (topk: among all genes)

struct MLM <: Objective
    mask_ratio::Float64
end

init_scheme(::MLM) = :default
hide_what(::MLM, ::RTF) = :gene
make_heads(::MLM, D::Int, H::Int, G::Int) = (cls = class_head(D, G),)

# class of each hidden token (cpu): ETF = rank of the gene, RTF = gene id
function targets(::MLM, input::Input, Xraw, xin, hide::AbstractMatrix{Bool}, ctx)
    idx = findall(vec(hide))
    y = input isa ETF ? vec(gene_ranks(Xraw, ctx.medians))[idx] : mod1.(idx, size(hide, 1))
    return (idx = idx, y = Int32.(y))
end

# topk tokens: ETF = the token's rank (its slot), RTF = its gene id among all genes
function targets(::MLM, input::Input, Xraw::TopkInput, xin, hide::AbstractMatrix{Bool}, ctx)
    idx = findall(vec(hide))
    y = input isa ETF ? mod1.(idx, size(hide, 1)) : vec(Xraw.ids)[idx]
    return (idx = idx, y = Int32.(y))
end

# classes of the mlm head: G (ranks or panel genes), or every gene for topk RTF
mlm_classes(input::Input, ctx) = get(ctx, :topk, false) && input isa RTF ? ctx.n_ids : ctx.G

# cross-entropy without a one-hot matrix (= Flux.logitcrossentropy up to rounding): logsumexp minus the target logit; linear index of (y[i], i) in lg
_target_lin(lg, y) = Flux.ChainRulesCore.ignore_derivatives(() -> y .+ size(lg, 1) .* (0:length(y)-1))
token_ce(lg::AbstractMatrix, y::AbstractVector{<:Integer}) = vec(logsumexp(lg; dims=1)) .- NNlib.gather(vec(lg), _target_lin(lg, y))
class_ce(lg::AbstractMatrix, y::AbstractVector{<:Integer}) = mean(token_ce(lg, y))

# gradient of class_ce in one classes x tokens buffer: (softmax - onehot) / n, built in place
function Flux.ChainRulesCore.rrule(::typeof(class_ce), lg::AbstractMatrix, y::AbstractVector{<:Integer})
    lse = logsumexp(lg; dims=1); li = _target_lin(lg, y); n = length(y)
    l = mean(vec(lse) .- NNlib.gather(vec(lg), li))
    function class_ce_pullback(dl)
        G = exp.(lg .- lse)
        NNlib.scatter!(+, vec(G), fill!(similar(lse, n), -1), li)
        G .*= Flux.ChainRulesCore.unthunk(dl) / n
        return Flux.ChainRulesCore.NoTangent(), G, Flux.ChainRulesCore.NoTangent()
    end
    return l, class_ce_pullback
end

function pretext_loss(o::MLM, input::Input, model, x, hide, t; teacher = nothing)
    h = encode(model.encoder, x, hide, hide_what(o, input))
    logits = model.heads.cls(hidden_tokens(h, t.idx))
    return class_ce(logits, t.y)
end

# trivial: uniform over the classes; topk RTF: the per-slot gene prior of the train top-k sets on the val masks (ctx.slot_ce, no context)
trivial_loss(::MLM, ctx) = get(ctx, :topk, false) && ctx.input isa RTF ? ctx.slot_ce : log(Float64(mlm_classes(ctx.input, ctx)))
# second logged trivial (topk RTF: the uniform log n_classes), else NaN
trivial_alt(::MLM, ctx) = get(ctx, :topk, false) && ctx.input isa RTF ? log(Float64(mlm_classes(ctx.input, ctx))) : NaN

# log p(gene | slot) from the train top-k ids (n_ids x K): gene counts per slot pooled over `window` neighbouring slots, add-a smoothing
function slot_prior(ids::AbstractMatrix{<:Integer}, n_ids::Int; window::Int = 8, a::Float64 = 0.01, len = nothing)
    K = size(ids, 1); C = zeros(Float64, n_ids, K)
    for j in axes(ids, 2), r in 1:(isnothing(len) ? K : Int(len[j])); C[ids[r, j], r] += 1; end
    Cs = cumsum(C; dims=2); P = zeros(Float32, n_ids, K)
    for r in 1:K
        lo = max(1, r - window ÷ 2); hi = min(K, lo + window - 1)
        cb = (Cs[:, hi] .- (lo > 1 ? Cs[:, lo-1] : 0.0)) ./ (hi - lo + 1)
        P[:, r] .= Float32.(log.((cb .+ a) ./ (sum(cb) + a * n_ids)))
    end
    return P
end

# mean of -log p(gene | slot) over the hidden tokens of the masks hide on topk inputs X (the no-context trivial of topk RTF mlm)
slot_ce(logp::AbstractMatrix, X::TopkInput, hide::AbstractMatrix{Bool}) =
    (idx = findall(vec(hide)); -mean(Float64(logp[vec(X.ids)[i], mod1(i, size(hide, 1))]) for i in idx))

topk_trivials(o::MLM, input::RTF, Xtr::TopkInput, Xva::TopkInput, xin_va, hide, ctx) = (slot_ce = slot_ce(slot_prior(Xtr.ids, ctx.n_ids; len = pad_len(Xtr)), Xva, hide),)
