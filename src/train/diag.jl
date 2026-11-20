# pretraining diagnostics on the fixed val set: per hidden token loss (and extras), summed per gene, per rank and per sample

# per hidden token stats (cpu vectors over t.idx); loss averages to pretext_loss
function token_stats(o::MLM, input::Input, model, x, hide, t; teacher = nothing)
    h = encode(model.encoder, x, hide, hide_what(o, input))
    lg = model.heads.cls(hidden_tokens(h, t.idx))
    # target logit gathered (no one-hot matrix: topk RTF has 19020 classes)
    tl = NNlib.gather(vec(lg), _target_lin(lg, t.y))
    loss = token_ce(lg, t.y)
    correct = tl .>= vec(maximum(lg; dims=1))
    st = (loss = Array(loss), correct = Array(Float32.(correct)))
    # ETF classes are ranks: abs error of the expected rank
    if input isa ETF
        ls = logsoftmax(lg; dims=1)
        er = vec(sum(exp.(ls) .* Float32.(dev(collect(1:size(lg, 1)))); dims=1))
        st = merge(st, (rank_abserr = Array(abs.(er .- Float32.(t.y))),))
    end
    return st
end

function token_stats(o::ERecon, input::Input, model, x, hide, t; teacher = nothing)
    h = encode(model.encoder, x, hide, :value)
    pred = Array(vec(model.heads.reg(hidden_tokens(h, t.idx))))
    z = Array(t.z)
    return (loss = (pred .- z) .^ 2, pred = pred, z = z)
end

function token_stats(o::LRecon, input::Input, model, x, hide, t; teacher = nothing)
    Y = lrecon_Y(teacher, x, t)
    P = model.heads.dec(hidden_tokens(encode(model.encoder, x, hide, :value), t.idx))
    c = vec(sum(P .* Y; dims=1) ./ (sqrt.(sum(P .^ 2; dims=1)) .* sqrt.(sum(Y .^ 2; dims=1)) .+ 1f-6))
    return (loss = Array(1f0 .- c),)
end

# sums per gene, per rank (rank of the gene in the sample by x / median, both inputs; topk: the token's slot) and per sample
function new_diag(G::Int, n::Int; n_rank::Int = G)
    z(k) = zeros(Float64, k)
    return Dict{String,Any}("gene_n" => z(G), "rank_n" => z(n_rank), "sample_n" => z(n),
                            "gene_loss" => z(G), "rank_loss" => z(n_rank), "sample_loss" => z(n))
end

# G = tokens per sample; R[t, s] = rank of token t in sample s; ids[t, s] = its gene (topk), nothing = token t is gene t (panel)
function add_diag!(D::Dict, st::NamedTuple, idx::AbstractVector, cols::AbstractVector{<:Integer}, R::AbstractMatrix, G::Int; ids = nothing)
    nG, nR = length(D["gene_n"]), length(D["rank_n"])
    # idx may be a device vector (pretrain moves the targets to the GPU): iterate a host copy
    for (j, li) in enumerate(Array(idx))
        t = mod1(li, G); s = cols[div(li - 1, G) + 1]; r = R[t, s]; g = isnothing(ids) ? t : Int(ids[t, s])
        D["gene_n"][g] += 1; D["rank_n"][r] += 1; D["sample_n"][s] += 1
        D["gene_loss"][g] += st.loss[j]; D["rank_loss"][r] += st.loss[j]; D["sample_loss"][s] += st.loss[j]
        for k in keys(st)
            k in (:loss, :pred, :z) && continue
            v = st[k][j]
            get!(D, "gene_$k", zeros(nG))[g] += v; get!(D, "rank_$k", zeros(nR))[r] += v
        end
        # erecon: sums for per-gene R2 / correlation (trivial predict-zero = gene_z2)
        if haskey(st, :pred)
            p = st.pred[j]; y = st.z[j]
            get!(D, "gene_p", zeros(nG))[g] += p; get!(D, "gene_p2", zeros(nG))[g] += p^2
            get!(D, "gene_z", zeros(nG))[g] += y; get!(D, "gene_z2", zeros(nG))[g] += y^2; get!(D, "gene_pz", zeros(nG))[g] += p * y
            get!(D, "rank_z2", zeros(nR))[r] += y^2
        end
    end
    return D
end

save_diag(path::String, D::Dict, step::Int, val_idx) = jldsave(path; step, val_sample_idx=collect(val_idx), sums=D)

# median over features of the across-sample sd of the mean-pooled last layer (unmasked; caller sets testmode)
function pooled_sd(enc, xin, bs::Int)
    P = reduce(hcat, [Array(pooled(enc, dev(xin[:, c]))) for c in Iterators.partition(1:size(xin, 2), bs)])
    return Float64(median(vec(std(P; dims=2))))
end

# lrecon: median across-sample sd of the raw (unnormalised) layer-averaged teacher states
function teacher_raw_sd(teacher, x)
    hs = token_layers(teacher, x)
    H = Array(sum(hs) ./ Float32(length(hs)))
    return Float64(median(vec(std(H; dims=3))))
end
