# health checks of a pretrained encoder against its init checkpoint: activation spread, variance shares, weight change, pretext vs trivial, lrecon target lookups

# lrecon: a gene-id lookup of the target below this fraction of trivial = target_shortcut (the gene id of a hidden token is visible; its slot is
# not: the slot lookup is reported only, it measures how much of the target is the hidden rank)
const LOOKUP_MAX = 0.95

# flat Float64 vector of all trainable arrays
_flat(m) = (l = [vec(Float64.(Array(x))) for x in Flux.trainables(m)]; isempty(l) ? Float64[] : reduce(vcat, l))
_relchg(w, w0) = norm(w .- w0) / norm(w0)
# rms change of the seed-independent init entries (equal in two inits of different seeds: LayerNorm, zero biases)
function _detchg(b, b0, br)
    w, w0, wr = _flat(b), _flat(b0), _flat(br); k = w0 .== wr
    return any(k) ? norm(w[k] .- w0[k]) / sqrt(count(k)) : NaN
end
_blocks(e::Encoder) = vcat([("value", e.value), ("gene", e.gene)], [("L$i", l) for (i, l) in enumerate(e.layers.layers)])

# last-layer token-state stats streamed over batches: total / sample / gene / token x sample sums of squares (balanced two-way split for the panel),
# plus the share of rank-position means (ranks by x / median, the old RTF position axis); topk: ids = token genes, the gene share then uses per-gene counts
# (unbalanced: genes appear in different numbers of samples, so the split is approximate) and ranks are the token slots
function state_stats(enc::Encoder, Xin, R::AbstractMatrix{<:Integer}; bs::Int = 50, ids = nothing)
    isnothing(pad_len(Xin)) || return state_stats_padded(enc, Xin, R; bs, ids)
    m = eval_device(enc); D = embed_dim(enc); T, N = size(Xin); nG = isnothing(ids) ? T : n_genes(enc)
    A = zeros(Float64, D, N); Pg = zeros(Float64, D, nG); cg = zeros(Float64, nG); Pr = zeros(Float64, D, T); s1 = zeros(Float64, D); s2 = zeros(Float64, D)
    for s in 1:bs:N
        b = s:min(s + bs - 1, N)
        h = Float64.(Array(token_states(m, dev(Xin[:, b]))[end]))
        A[:, b] .= dropdims(mean(h, dims=2), dims=2)
        if isnothing(ids)
            Pg .+= dropdims(sum(h, dims=3), dims=3)
        else
            for (jj, j) in enumerate(b); Pg[:, ids[:, j]] .+= h[:, :, jj]; cg[ids[:, j]] .+= 1; end
        end
        for (jj, j) in enumerate(b); Pr[:, R[:, j]] .+= h[:, :, jj]; end
        s1 .+= vec(sum(h, dims=(2, 3))); s2 .+= vec(sum(h .^ 2, dims=(2, 3)))
    end
    mu = s1 ./ (N * T)
    sst = s2 .- N * T .* mu .^ 2; ssa = T .* vec(sum((A .- mu) .^ 2, dims=2))
    ssb = isnothing(ids) ? N .* vec(sum((Pg ./ N .- mu) .^ 2, dims=2)) : (k = cg .> 0; vec(sum(cg[k]' .* (Pg[:, k] ./ cg[k]' .- mu) .^ 2, dims=2)))
    ssc = N .* vec(sum((Pr ./ N .- mu) .^ 2, dims=2)); ssr = sst .- ssa .- ssb
    tot = sum(sst)
    return (pool_sd = median(vec(std(A, dims=2))), tok_sd = median(sqrt.(max.(sst, 0) ./ (N * T - 1))), frac_sample = sum(ssa) / tot,
            frac_gene = sum(ssb) / tot, frac_rank = sum(ssc) / tot, frac_tokxsmp = sum(ssr) / tot, var_tot = tot / (D * T * N),
            var_tokxsmp = sum(ssr) / (D * T * N))
end

# the same decomposition over the real tokens of padded inputs (single cells): per-sample, per-gene and per-slot means weighted by their real-token counts
function state_stats_padded(enc::Encoder, Xin, R::AbstractMatrix{<:Integer}; bs::Int = 50, ids = nothing)
    m = eval_device(enc); D = embed_dim(enc); T, N = size(Xin); nG = n_genes(enc); len = Xin.len
    A = zeros(Float64, D, N); Pg = zeros(Float64, D, nG); cg = zeros(Float64, nG); Pr = zeros(Float64, D, T); cr = zeros(Float64, T); s1 = zeros(Float64, D); s2 = zeros(Float64, D)
    for s in 1:bs:N
        b = s:min(s + bs - 1, N)
        h = Float64.(Array(token_states(m, dev(Xin[:, b]))[end]))
        for (jj, j) in enumerate(b)
            L = Int(len[j]); hj = view(h, :, 1:L, jj)
            A[:, j] .= vec(mean(hj, dims=2)); s1 .+= vec(sum(hj, dims=2)); s2 .+= vec(sum(hj .^ 2, dims=2))
            for t in 1:L; Pg[:, ids[t, j]] .+= hj[:, t]; cg[ids[t, j]] += 1; Pr[:, R[t, j]] .+= hj[:, t]; cr[R[t, j]] += 1; end
        end
    end
    n = sum(Float64, len); mu = s1 ./ n
    sst = s2 .- n .* mu .^ 2; ssa = vec(sum(Float64.(len)' .* (A .- mu) .^ 2, dims=2))
    wsum(P, c) = (k = c .> 0; vec(sum(c[k]' .* (P[:, k] ./ c[k]' .- mu) .^ 2, dims=2)))
    ssb = wsum(Pg, cg); ssc = wsum(Pr, cr); ssr = sst .- ssa .- ssb; tot = sum(sst)
    return (pool_sd = median(vec(std(A, dims=2))), tok_sd = median(sqrt.(max.(sst, 0) ./ (n - 1))), frac_sample = sum(ssa) / tot,
            frac_gene = sum(ssb) / tot, frac_rank = sum(ssc) / tot, frac_tokxsmp = sum(ssr) / tot, var_tot = tot / (D * n), var_tokxsmp = sum(ssr) / (D * n))
end

# lrecon: cosine loss of two no-context lookups of the teacher targets (trivial 1.0): the mean target per gene id and per token slot, fit on the first half
# of the columns of Xin and scored on the second half (batches of 100 as pretraining); gc = gene centre for topk tokens (panel: nothing, slot = gene); tgt(enc, x) = the targets
_cosd(a, b) = (s = 0.0; na = 0.0; nb = 0.0; @inbounds for i in eachindex(a, b); s += a[i] * b[i]; na += a[i]^2; nb += b[i]^2; end; s / (sqrt(na) * sqrt(nb) + 1e-6))
function lrecon_lookups(enc::Encoder, Xin, gc; bs::Int = 100, tgt = (e, x) -> isnothing(gc) ? teacher_targets(e, x) : teacher_targets(e, x, gc))
    m = eval_device(enc); N = size(Xin, 2); h = N ÷ 2; D = embed_dim(enc); T = size(Xin, 1); topk = Xin isa TopkInput
    nG = topk ? n_genes(enc) : T
    tg(c) = Array(tgt(m, dev(Xin[:, c])))
    gid(c) = topk ? Int.(Xin.ids[:, c]) : repeat(1:T, 1, length(c))
    nreal(c, j) = isnothing(pad_len(Xin)) ? T : Int(Xin.len[c[j]])   # padded inputs: real slots only
    Sg = zeros(Float64, D, nG); cg = zeros(Int, nG); Ss = zeros(Float64, D, T)
    for c in Iterators.partition(1:h, bs)
        Y = tg(c); g = gid(c)
        for j in axes(Y, 3), t in 1:nreal(c, j); Sg[:, g[t, j]] .+= view(Y, :, t, j); cg[g[t, j]] += 1; Ss[:, t] .+= view(Y, :, t, j); end
    end
    cs = zeros(Int, T); for c in Iterators.partition(1:h, bs), j in eachindex(c); cs[1:nreal(c, j)] .+= 1; end
    Mg = Sg ./ max.(cg', 1); Ms = Ss ./ max.(cs', 1); lg = 0.0; ls = 0.0; n = 0
    for c in Iterators.partition(h+1:N, bs)
        Y = tg(c); g = gid(c)
        for j in axes(Y, 3), t in 1:nreal(c, j)
            y = view(Y, :, t, j); k = g[t, j]
            lg += cg[k] > 0 ? 1 - _cosd(view(Mg, :, k), y) : 1.0; ls += 1 - _cosd(view(Ms, :, t), y); n += 1
        end
    end
    return (gene = lg / n, slot = ls / n)
end

# pretext value and trivial of a checkpoint from the run's log.tsv: init = step 0, best = its saved step, else the last row
function pretext_from_log(ck::AbstractString, ex::Dict)
    lp = joinpath(dirname(ck), "log.tsv")
    isfile(lp) || return NaN, NaN
    rows = [split(l, '\t') for l in readlines(lp)]; h = rows[1]; rows = rows[2:end]
    isempty(rows) && return NaN, NaN
    col(k) = findfirst(==(k), h)
    st = basename(ck) == "init" ? 0 : (basename(ck) == "best" ? get(ex, "step", -1) : -1)
    r = st >= 0 ? something(findlast(x -> parse(Int, x[col("step")]) == st, rows), length(rows)) : length(rows)
    return parse(Float64, rows[r][col("val")]), parse(Float64, rows[r][col("trivial")])
end

function run_health(cfg::Dict; data::Union{Nothing,TahoeData} = nothing)
    t_start = time()
    ck = String(rstrip(string(cfg["ckpt"]), '/')); name = string(get(cfg, "name", join(splitpath(ck)[end-2:end], "/")))
    nh = get(cfg, "nh", 5000); bs = get(cfg, "probe_bs", 50)
    icfg = merge(cfg, ckpt_input_cfg(ckpt_config(ck)))   # the checkpoint's token set
    d = cfg_data(icfg, data)
    vs = sort(shuffle(MersenneTwister(3), pretrain_split(d).val)[1:min(nh, length(pretrain_split(d).val))])
    Xraw = raw_input(d, vs, icfg)
    println("health subset: $(length(vs)) pretrain-val samples")

    # checkpoint, its run's init/ and an unrelated init (seed + 1) of the same arch and scheme
    is_ref = basename(ck) == "init"
    initdir = joinpath(dirname(ck), "init")
    m, pcfg, ex = load_encoder(ck)
    m0, _, _ = load_encoder(initdir)
    seed = Int(pcfg["seed"])
    mr, _, _ = load_encoder(initdir; random=true, init_seed=seed + 1)
    mre, _, _ = load_encoder(initdir; random=true, init_seed=seed)
    rebuild = maximum(abs.(_flat(mre) .- _flat(m0)))
    @printf("init rebuild from seed %d: max |diff| vs saved init %.3g\n", seed, rebuild)
    input = parse_input(pcfg["input"])
    Xin = model_input(input, Xraw, d, pcfg); R = token_ranks(Xraw, d); ids = token_genes(Xraw)
    # lrecon topk: the checkpoint's saved gene centre, else one estimated with this encoder on train PBs (init checkpoints); slot norm: no centre
    lr = pcfg["objective"] == "lrecon"; gc = nothing
    if lr && icfg["tokens"] == "topk" && get(pcfg, "lrecon_topk_norm", "gene") == "gene"
        gc = centre_from_extra(ex)
        if isnothing(gc)
            tr = pretrain_split(d).train; wc = sort(shuffle(MersenneTwister(4), tr)[1:min(LRECON_WARM, length(tr))])
            gc = fresh_centre(m, model_input(input, raw_input(d, wc, icfg), d, pcfg))
        end
    end
    d = nothing; Xraw = nothing; GC.gc()

    B, B0, BR = _blocks(m), _blocks(m0), _blocks(mr)
    chg = Dict(n => _relchg(_flat(b), _flat(b0)) for ((n, b), (_, b0)) in zip(B, B0))
    ref = Dict(n => _relchg(_flat(b), _flat(b0)) for ((n, b), (_, b0)) in zip(BR, B0))
    det = Dict(n => _detchg(b, b0, br) for ((n, b), (_, b0), (_, br)) in zip(B, B0, BR))
    Ls = ["L$i" for i in 1:length(m.layers.layers)]; Llast = Ls[end]

    t0 = time()
    S = state_stats(m, Xin, R; bs, ids); GC.gc(); CUDA.functional() && CUDA.reclaim()
    S0 = state_stats(m0, Xin, R; bs, ids)
    lk = lr ? lrecon_lookups(m, Xin, gc) : (gene = NaN, slot = NaN)
    t_fwd = time() - t0

    # pretext vs trivial (log.tsv, or pretext= / trivial= overrides)
    pv, tv = pretext_from_log(ck, ex)
    haskey(cfg, "pretext") && (pv = Float64(cfg["pretext"])); haskey(cfg, "trivial") && (tv = Float64(cfg["trivial"]))
    ratio = (isfinite(pv) && isfinite(tv)) ? pv / tv : NaN
    learned = isnan(ratio) ? "na" : ratio < 0.97 ? "yes" : "no"

    # drug probe of the same name, if present
    pdrug = NaN; pt = joinpath(cfg["results_dir"], "probe.tsv")
    if isfile(pt)
        L = [split(l, '\t') for l in readlines(pt)]; c = findfirst(==("drug_h2_test"), L[1])
        for f in L[2:end]; f[1] == name && (pdrug = parse(Float64, f[c])); end
    end

    # verdict, rules checked in order
    x_pool, x_tok, x_txs, x_tot = S.pool_sd / S0.pool_sd, S.tok_sd / S0.tok_sd, S.var_tokxsmp / S0.var_tokxsmp, S.var_tot / S0.var_tot
    mov = maximum(chg[l] for l in Ls); fpos = max(S.frac_gene, S.frac_rank)
    verdict, vnote = if is_ref
        "reference", "init checkpoint (seed $seed)"
    elseif mov < 0.1
        "near_init", @sprintf("transformer weights moved %.3g < 0.1 of init", mov)
    elseif x_txs < 0.1 && fpos > 0.98
        "token_uniform_collapse", @sprintf("h_L position-constant (gene share %.3f rank share %.3f, tokxsmp var x%.3g init)", S.frac_gene, S.frac_rank, x_txs)
    elseif x_pool < 0.1
        "scale_collapse", @sprintf("pooled sd x%.3g init", x_pool)
    elseif x_pool > 1000
        "scale_blowup", @sprintf("pooled sd x%.3g init", x_pool)
    elseif lk.gene < LOOKUP_MAX
        "target_shortcut", @sprintf("lrecon target partly a lookup of the visible gene id: %.3f of trivial (< %.2f)", lk.gene, LOOKUP_MAX)
    elseif learned == "no"
        "pretext_not_learned", @sprintf("pretext %.4g / trivial %.4g = %.3f >= 0.97", pv, tv, ratio)
    else
        "healthy", learned == "na" ? "no trivial baseline" : @sprintf("pretext %.3f of trivial", ratio)
    end
    x_tot > 100 && (vnote *= @sprintf("; h_L variance x%.3g init", x_tot))

    row = (name = name, dataset = get(cfg, "dataset", "pb"), input = pcfg["input"], tokens = icfg["tokens"], objective = pcfg["objective"], verdict = verdict, pool_drug_test = pdrug,
           pool_sd_med = S.pool_sd, pool_sd_x_init = x_pool, tok_sd_med = S.tok_sd, tok_sd_x_init = x_tok,
           frac_sample = S.frac_sample, frac_gene = S.frac_gene, frac_rank = S.frac_rank, frac_tokxsmp = S.frac_tokxsmp,
           var_tokxsmp = S.var_tokxsmp, var_tokxsmp_x_init = x_txs, var_tot_x_init = x_tot,
           chg_value = chg["value"], chg_gene = chg["gene"], chg_L1 = chg["L1"], chg_L2 = chg[Llast], ref_L1 = ref["L1"], ref_L2 = ref[Llast],
           det_value = det["value"], det_L1 = det["L1"], det_L2 = det[Llast], init_seed = seed, step = get(ex, "step", -1),
           pretext = pv, trivial = tv, pretext_ratio = ratio, pretext_learned = learned, lookup_gene = lk.gene, lookup_slot = lk.slot, n_val = length(vs),
           verdict_note = vnote * @sprintf("; fwd %.0fs total %.0fs", t_fwd, time() - t_start), ckpt = ck)
    append_row(joinpath(cfg["results_dir"], "health.tsv"), row)
    println(join(["$k=$(v isa AbstractFloat ? @sprintf("%.4g", v) : v)" for (k, v) in pairs(row)], " | "))
    println("appended to ", joinpath(cfg["results_dir"], "health.tsv"))
    return row
end
