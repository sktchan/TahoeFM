# frozen encoder probe: shrinkage LDA on mean-pooled layer states for drug, drug after cell-line centring, cell line, and the moa readout on held-out drugs

const PROBE_COLS = ["drug_h2", "drug_clc_h2", "drug_meanstd", "cell_h2", "drug_all", "drug_clc_all", "cell_all", "drug_h0", "drug_h1"]

# encoder to probe and its input name; random[:seed] = default arch, Random.seed!(seed), default init (as the old random:seed); G = gene table rows, n_ranks = RTF rank table rows
function probe_encoder(ck::AbstractString, input_name::AbstractString, G::Int; n_ranks::Int = G, random::Bool = false, init_seed::Int = 0)
    if startswith(ck, "random")
        seed = occursin(':', ck) ? parse(Int, split(ck, ':')[2]) : 0
        Random.seed!(seed)
        enc = Encoder(parse_input(input_name); n_genes=G, n_ranks, embed_dim=DEFAULTS["embed_dim"], hidden_dim=DEFAULTS["hidden_dim"],
                      n_heads=DEFAULTS["n_heads"], n_layers=DEFAULTS["n_layers"], drop_prob=0.1)
        return enc, input_name, DEFAULTS
    end
    enc, pcfg, _ = load_encoder(ck; random, init_seed)
    return enc, pcfg["input"], pcfg
end

# pooled token states of every layer and the last layer's token sd (D x n each)
function pooled_states(enc::Encoder, Xin; bs::Int = 50)
    m = eval_device(enc); D = embed_dim(enc); n = size(Xin, 2); L = length(enc.layers.layers)
    pools = [zeros(Float32, D, n) for _ in 0:L]; sdL = zeros(Float32, D, n)
    for s in 1:bs:n
        b = s:min(s + bs - 1, n)
        hs = token_states(m, dev(Xin[:, b]))
        for l in eachindex(hs); pools[l][:, b] .= Array(meanpool(hs[l])); end
        sdL[:, b] .= Array(dropdims(std(hs[end], dims=2, corrected=false), dims=2))
    end
    return pools, sdL
end

# input reference for topk tokens, on the K genes most often in the train columns' top-k sets: graded top-k rank features (rtf), or the tokens' values (etf, vals = K x cols), 0 off the sample's top-k
function topk_reference(d::TahoeData, cols, tr, K::Int; vals::Union{Nothing,AbstractMatrix} = nothing)
    ids = d.topk.ids[1:K, cols]; cnt = zeros(Int, length(d.full.medians))
    for j in tr, r in 1:K; cnt[ids[r, j]] += 1; end
    keep = sort(partialsortperm(cnt, 1:K; rev=true)); row = zeros(Int, length(cnt)); row[keep] .= 1:K
    F = zeros(Float32, K, length(cols))
    for j in axes(ids, 2), r in 1:K; (q = row[ids[r, j]]) > 0 && (F[q, j] = isnothing(vals) ? Float32((K + 1 - r) / K) : Float32(vals[r, j])); end
    return F
end

# subtract per-key means over the ref columns of the key; a key with < 10 ref columns uses its fallback columns (probe: train + val, never test), none left = uncentred
function centre_by(F::AbstractMatrix, key::AbstractVector, ref::AbstractVector{Bool}; fallback::AbstractVector{Bool} = ref)
    Y = similar(F, Float32); g = Dict{eltype(key),Vector{Int}}()
    for (j, k) in enumerate(key); push!(get!(g, k, Int[]), j); end
    for c in values(g)
        r = filter(j -> ref[j], c); length(r) < 10 && (r = filter(j -> fallback[j], c))
        Y[:, c] .= isempty(r) ? F[:, c] : F[:, c] .- mean(view(F, :, r), dims=2)
    end
    return Y
end

# moa readout setup (DESIGN_v2 7.3): moa classes with >= min_drugs of the lvl2 drugs; 5 folds of drugs within each class; used columns = fit (train + val) columns of those drugs
function moa_setup(meta::NamedTuple, drugs::AbstractVector{<:AbstractString}, cols, fit; min_drugs::Int = 4, seed::Int = 1)
    dm = Dict{String,String}(); for c in cols; dm[meta.drug[c]] = meta.moa[c]; end
    for (i, dr) in enumerate(meta.drug); haskey(dm, dr) || (dm[dr] = meta.moa[i]); end
    bycls = Dict{String,Vector{String}}()
    for dr in drugs; c = get(dm, dr, ""); c in ("", "unclear") || push!(get!(bycls, c, String[]), dr); end
    cls = sort([c for (c, v) in bycls if length(v) >= min_drugs]); cid = Dict(c => i for (i, c) in enumerate(cls))
    fold = Dict{String,Int}(); dcl = Dict{String,Int}()
    for c in cls, (i, dr) in enumerate(shuffle(Xoshiro(seed), sort(bycls[c]))); fold[dr] = mod1(i, 5); dcl[dr] = cid[c]; end
    use = [j for j in fit if haskey(dcl, meta.drug[cols[j]])]
    dr = meta.drug[cols[use]]
    return (K = length(cls), cls = cls, use = use, y = [dcl[x] for x in dr], fold = [fold[x] for x in dr], drug = dr, pos = meta.pos[cols[use]],
            ndrugs = length(unique(dr)))
end

_balacc(correct, y) = isempty(y) ? NaN : mean(mean(correct[y .== c]) for c in unique(y))

# bootstrap se of the balanced accuracy, resampling drugs within class
function _moa_se(c, S; B::Int = 500, seed::Int = 1)
    ud = unique(S.drug); di = Dict(x => i for (i, x) in enumerate(ud)); s = zeros(length(ud)); n = zeros(length(ud)); dc = zeros(Int, length(ud))
    for j in eachindex(c); k = di[S.drug[j]]; s[k] += c[j]; n[k] += 1; dc[k] = S.y[j]; end
    by = [findall(==(k), dc) for k in 1:S.K]; rng = Xoshiro(seed); v = zeros(B)
    for b in 1:B
        v[b] = mean((r = rand(rng, by[k], length(by[k])); sum(s[r]) / sum(n[r])) for k in 1:S.K if !isempty(by[k]))
    end
    return std(v)
end

# moa readout of features F (columns = probe columns, already centred): out-of-fold shrinkage LDA per alpha, alpha cross-fitted over drug halves; balanced accuracy and se
function moa_readout(F::AbstractMatrix, S)
    S.K < 2 && return (bal = NaN, se = NaN)
    Fu = F[:, S.use]; P = zeros(Int, length(S.use), length(LDA_ALPHAS))
    for f in 1:5
        te = findall(S.fold .== f); ft = findall(S.fold .!= f)
        isempty(te) && continue
        Z = lda_zprep(Fu, ft); P[te, :] = lda_alpha_preds(Z[:, ft], S.y[ft], Z[:, te], S.K)
    end
    ud = shuffle(Xoshiro(1), sort(unique(S.drug))); hA = Set(ud[1:div(length(ud), 2)]); half = [x in hA for x in S.drug]
    ba(k) = [_balacc(P[k, a] .== S.y[k], S.y[k]) for a in axes(P, 2)]
    aA = argmax(ba(half)); aB = argmax(ba(.!half))
    c = similar(S.y, Bool); c[half] .= P[half, aB] .== S.y[half]; c[.!half] .= P[.!half, aA] .== S.y[.!half]
    return (bal = _balacc(c, S.y), se = _moa_se(c, S))
end

# position-only moa baseline: majority class of the fit drugs' PBs at the same plate position
function moa_position(S)
    S.K < 2 && return NaN
    pp = zeros(Int, length(S.use))
    for f in 1:5
        C = zeros(Int, S.K, 96)
        for j in eachindex(S.y); S.fold[j] != f && (C[S.y[j], S.pos[j] + 1] += 1); end
        for j in eachindex(S.y); S.fold[j] == f && (pp[j] = sum(C[:, S.pos[j] + 1]) == 0 ? 0 : argmax(C[:, S.pos[j] + 1])); end
    end
    return _balacc(pp .== S.y, S.y)
end

function run_probe(cfg::Dict; data::Union{Nothing,TahoeData} = nothing)
    get(cfg, "dataset", "pb") == "pb" || error("the probe is not ported to dataset = $(cfg["dataset"]) yet")
    t_start = time()
    name = string(cfg["name"]); ck = String(rstrip(string(cfg["ckpt"]), '/'))
    ntr = get(cfg, "ntr", 60000); nva = get(cfg, "nva", 10000); bs = get(cfg, "probe_bs", 50); notes = string(get(cfg, "notes", ""))
    is_input = ck == "input"; pm = pos_method(get(cfg, "pos_correct", false)); pc = !isnothing(pm)
    # token settings: a checkpoint's own, else the cfg's (input, tokens, top_k)
    icfg = is_input || startswith(ck, "random") ? merge(cfg, Dict("tokens" => token_set(cfg))) : merge(cfg, ckpt_input_cfg(ckpt_config(ck)))
    tok = icfg["tokens"]
    d = cfg_data(icfg, data)
    nid, T = token_dims(d, icfg)

    # lvl2 split exactly as finetune; cell line = lvl1 label of the same samples
    idx, y, names = labels(d, "lvl2")
    sp = finetune_split(d, idx)
    @printf("lvl2 split: train %d val %d test %d (%d drugs) | %s tokens%s\n", length(sp.train), length(sp.val), length(sp.test), length(names), tok, pc ? " | position-corrected ($pm)" : "")
    str = sort(randperm(MersenneTwister(1), length(sp.train))[1:min(ntr, length(sp.train))])
    sva = sort(randperm(MersenneTwister(2), length(sp.val))[1:min(nva, length(sp.val))])
    pos = vcat(sp.train[str], sp.val[sva], sp.test); cols = idx[pos]
    ydr = y[pos]; Kd = length(names)
    cls = d.meta.cell_line[cols]; CL = sort(unique(cls)); cid = Dict(c => i for (i, c) in enumerate(CL)); ycl = [cid[c] for c in cls]; Kc = length(CL)
    n1, n2, n3 = length(str), length(sva), length(sp.test)
    TR, VA, TE = 1:n1, n1+1:n1+n2, n1+n2+1:n1+n2+n3
    @printf("probe sizes: train %d val %d test %d | drugs %d cell lines %d\n", n1, n2, n3, Kd, Kc)

    gpu = "cpu"; t_fwd = 0.0
    if is_input
        input = string(cfg["input"])
        F = if input == "etf" && tok == "topk"
            # the top-k tokens' ETF values (as model_input: etf_scale raw or gene z) on the K most frequent train top-k genes, z-scored on the probe-train columns
            V = topk_reference(d, cols, TR, T; vals = topk_scale(raw_input(d, cols, icfg), d, get(icfg, "etf_scale", "raw")))
            zmu = mean(V[:, TR], dims=2); zsd = std(V[:, TR], dims=2); zsd[zsd .< 1f-6] .= 1f0
            (V .- zmu) ./ zsd
        elseif input == "etf"
            Xt = d.X[:, idx[sp.train]]
            zmu = vec(mean(Xt, dims=2)); zsd = vec(std(Xt, dims=2)); zsd[zsd .< 1f-6] .= 1f0; Xt = nothing
            (d.X[:, cols] .- zmu) ./ zsd
        elseif input == "rtf"
            tok == "topk" ? topk_reference(d, cols, TR, T) : rank_features(gene_ranks(d.X[:, cols], d.medians), d.X[:, cols])
        elseif input == "raw"
            d.X[:, cols]
        else
            error("input reference must be etf (z), rtf (rank) or raw, got $input")
        end
        pc && (F = pos_correct(F, d.meta, cols, TR; method = pm))
        P = pca_project(F, TR, 256)
        pools = [P, P, P]; sdL = nothing
        tok == "topk" && (notes = join(filter(!isempty, ["$(input == "etf" ? "top-k values" : "graded top-k ranks") on the $T genes most often in probe-train top-k sets", notes]), "; "))
        println("input reference ($input, $tok): $(size(F, 1)) dims; h2 columns = PCA256 fit on probe train, *_all columns = all dims", tok == "topk" ? "; " * notes : "")
    else
        enc, input, pcfg = probe_encoder(ck, string(get(cfg, "input", "")), nid; n_ranks=T, random=get(cfg, "random", false), init_seed=get(cfg, "init_seed", 0))
        Xin = model_input(parse_input(input), raw_input(d, cols, icfg), d, pcfg)
        t0 = time()
        pools, sdL = pooled_states(enc, Xin; bs)
        t_fwd = time() - t0; gpu = gpu_name()
        @printf("forward %.0f s (%.2f ms/sample) on %s\n", t_fwd, 1000 * t_fwd / size(Xin, 2), gpu); flush(stdout)
        Xin = nothing
        em = joinpath(cfg["results_dir"], "probe_emb"); mkpath(em)
        jldsave(joinpath(em, "$name.jld2"); pools, std_last=sdL, ydrug=ydr, ycell=ycl, cell_lines=CL, sample_idx=cols, ntr=n1, nva=n2, nte=n3, ckpt=ck, input, tokens=tok)
        pc && (pools = [pos_correct(Q, d.meta, cols, TR; method = pm) for Q in pools]; sdL = pos_correct(sdL, d.meta, cols, TR; method = pm))
    end
    meta = d.meta; d = nothing; GC.gc()

    # readouts
    t1 = time()
    rd(F, yy, K) = lda_readout(F, yy, K, TR, VA, TE)
    function clc(F)
        Z = lda_zprep(F, TR); Mc, _ = lda_group_means(Z[:, TR], ycl[TR], Kc); R = Z .- Mc[:, ycl]
        s = std(R[:, TR], dims=2); s[s .< 1e-12] .= 1; R ./= s
        return shrink_lda(R[:, TR], ydr[TR], R[:, VA], ydr[VA], R[:, TE], ydr[TE], Kd)
    end
    NA = (NaN, NaN, NaN, Int[])
    hL = pools[end]
    ALL = is_input ? F : vcat(pools..., sdL)
    r = if is_input
        Dict("drug_h2" => rd(hL, ydr, Kd), "drug_clc_h2" => clc(hL), "drug_meanstd" => NA, "cell_h2" => rd(hL, ycl, Kc),
             "drug_all" => rd(F, ydr, Kd), "drug_clc_all" => clc(F), "cell_all" => rd(F, ycl, Kc), "drug_h0" => NA, "drug_h1" => NA)
    else
        Dict("drug_h2" => rd(hL, ydr, Kd), "drug_clc_h2" => clc(hL), "drug_meanstd" => rd(vcat(hL, sdL), ydr, Kd),
             "cell_h2" => rd(hL, ycl, Kc), "drug_all" => rd(ALL, ydr, Kd), "drug_clc_all" => clc(ALL), "cell_all" => rd(ALL, ycl, Kc),
             "drug_h0" => rd(pools[1], ydr, Kd), "drug_h1" => rd(pools[2], ydr, Kd))
    end
    # moa on held-out drugs: train + val columns only, features centred per (cell line, plate) on the train columns
    S = moa_setup(meta, names, cols, vcat(collect(TR), collect(VA)))
    key = collect(zip(meta.cell_line[cols], meta.plate[cols])); ref = falses(length(cols)); ref[TR] .= true; fb = copy(ref); fb[VA] .= true
    mh = moa_readout(centre_by(hL, key, ref; fallback = fb), S); ma = moa_readout(centre_by(ALL, key, ref; fallback = fb), S); mp = moa_position(S)
    t_lda = time() - t1
    println(name, " | ", join([@sprintf("%s %.3f (val %.3f)", c, r[c][2], r[c][1]) for c in PROBE_COLS], " | "))
    @printf("moa: %d classes, %d drugs, %d PBs | h2 %.4f (se %.4f) | all %.4f (se %.4f) | position-only %.4f | chance %.4f\n",
            S.K, S.ndrugs, length(S.use), mh.bal, mh.se, ma.bal, ma.se, mp, 1 / max(S.K, 1))

    ks = vcat([:name, :dataset, :input, :tokens, :pos_correct, :ckpt], Symbol.(PROBE_COLS .* "_test"),
              [:moa_h2, :moa_h2_se, :moa_all, :moa_all_se, :moa_pos, :moa_classes, :moa_drugs, :moa_pbs],
              [:ntr, :nva, :nte, :val_accs, :gpu, :fwd_s, :lda_s, :total_s, :notes])
    vs = vcat(Any[name, get(cfg, "dataset", "pb"), is_input ? "input_" * input : input, token_label(tok, icfg), pc ? pm : false, ck], Any[r[c][2] for c in PROBE_COLS],
              Any[mh.bal, mh.se, ma.bal, ma.se, mp, S.K, S.ndrugs, length(S.use)], Any[n1, n2, n3],
              Any[join([@sprintf("%.4f", r[c][1]) for c in PROBE_COLS], "/"), replace(gpu, ' ' => '_'),
                  round(Int, t_fwd), round(Int, t_lda), round(Int, time() - t_start), isempty(notes) ? "-" : notes])
    row = NamedTuple{Tuple(ks)}(Tuple(vs))
    append_row(joinpath(cfg["results_dir"], "probe.tsv"), row)
    println("appended to ", joinpath(cfg["results_dir"], "probe.tsv"))
    return row
end
