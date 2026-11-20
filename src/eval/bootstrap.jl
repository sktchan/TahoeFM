# cluster paired bootstrap of test (or reloc) predictions: per-arm accuracy with 95% CIs and contrasts between arms; clusters = drug|dose groups (wells) or classes

const BOOT_DELTA = 0.03   # equivalence margin
const BOOT_PT_MIN = 0.07  # pretrain contrasts (one pretrain seed) below this are never called different

# preds, trues, sample idx, groups (drug|dose or DMSO plate|well) and source of <run dir | file.jld2>[#final|best|reloc_final|reloc_best|key]
function load_preds(src::AbstractString)
    path, key = occursin('#', src) ? String.(split(src, '#'; limit=2)) : (String(src), "preds_final")
    isdir(path) && (path = joinpath(path, "predstrues.jld2"))
    key in ("final", "best", "reloc_final", "reloc_best") && (key = "preds_" * key)
    s = load(path)
    haskey(s, key) || error("$path has no key $key (keys: $(join(keys(s), ", ")))")
    rl = startswith(key, "preds_reloc")
    tk = rl ? "trues_reloc" : haskey(s, "trues") ? "trues" : "all_trues"
    return Int.(s[key]), Int.(s[tk]), get(s, rl ? "reloc_sample_idx" : "test_sample_idx", nothing), get(s, rl ? "reloc_groups" : "test_groups", nothing), path * "#" * key
end

# cluster = group (resample drug|dose groups = wells, the Tahoe default) or class (resample true classes, the LincsFM drug cluster)
function run_bootstrap(out::AbstractString, arms::Vector{Pair{String,String}}; contrasts = Tuple{Vector{String},Bool}[],
                       seed_gap::Real = 0.0, all_pairs::Bool = false, B::Int = 2000, cluster::AbstractString = "group")
    isempty(arms) && error("no arms")
    cluster in ("group", "class") || error("cluster must be group or class, got $cluster")
    L = [(n, load_preds(s)...) for (n, s) in arms]
    y = L[1][3]; tsi = nothing
    grp = cluster == "group" ? L[1][5] : nothing
    cluster == "group" && isnothing(grp) && error("$(L[1][6]) has no groups: rerun it, or use cluster = class")
    for (n, p, t, ix, gr, path) in L
        length(p) == length(y) || error("$n: $(length(p)) preds vs $(length(y)) test samples")
        t == y || error("$n: trues differ from $(L[1][1]) (different test set or order): $path")
        if !isnothing(ix)
            isnothing(tsi) ? (tsi = ix) : (ix == tsi || error("$n: test_sample_idx differs: $path"))
        end
        isnothing(grp) || isnothing(gr) || gr == grp || error("$n: groups differ: $path")
    end
    println("$(length(L)) arms, $(length(y)) test samples, test_sample_idx checked on $(count(x -> !isnothing(x[4]), L)) arms")

    # clusters = groups or true classes; per-arm correct counts per cluster
    ck = isnothing(grp) ? y : grp
    K = sort(unique(ck)); kid = Dict(c => j for (j, c) in enumerate(K)); nc = length(K)
    n = zeros(Int, nc); for t in ck; n[kid[t]] += 1; end
    corr = Dict(nm => (c = zeros(Int, nc); for (pp, t, k) in zip(p, y, ck); c[kid[k]] += (pp == t); end; c) for (nm, p, _, _, _, _) in L)
    C = sort(unique(y))
    @printf("%d %s clusters, %d-%d samples each\n", nc, cluster, minimum(n), maximum(n))

    # shared resample: multiplicity of each cluster in each draw (depends only on nc, B and the seed)
    rng = MersenneTwister(2026)
    W = zeros(Int32, nc, B)
    for b in 1:B, j in rand(rng, 1:nc, nc); W[j, b] += 1; end
    den = vec(n' * W)
    bacc(c) = vec(c' * W) ./ den
    pct(v, q) = quantile(v, q)
    balacc(p) = mean([mean(p[y .== c] .== c) for c in C])

    mkpath(dirname(abspath(out)))
    arm_rows = NamedTuple[]
    for (nm, p, _, _, _, path) in L
        v = bacc(corr[nm]); a = mean(p .== y)
        @printf("%-28s acc %.4f [%.4f, %.4f] bal %.4f\n", nm, a, pct(v, .025), pct(v, .975), balacc(p))
        push!(arm_rows, (name = nm, acc = a, lo95 = pct(v, .025), hi95 = pct(v, .975), boot_sd = std(v), bal_acc = balacc(p),
                         n_test = length(y), n_clusters = nc, cluster = cluster, B = B, source = path))
    end
    write_curve(out * "_arms.tsv", arm_rows)

    cs = copy(contrasts)
    all_pairs && for a in 1:length(L), b in a+1:length(L); push!(cs, ([L[a][1], L[b][1]], false)); end
    con_rows = NamedTuple[]
    for (nm, pt) in cs
        all(x -> haskey(corr, x), nm) || error("contrast $(join(nm, ':')) names an unknown arm")
        ws = length(nm) == 2 ? [1, -1] : [1, -1, -1, 1]
        v = sum(w .* bacc(corr[x]) for (w, x) in zip(ws, nm)); dlt = sum(w * sum(corr[x]) for (w, x) in zip(ws, nm)) / length(y)
        lo, hi = pct(v, .025), pct(v, .975)
        dec = (lo > 0 || hi < 0) && abs(dlt) > seed_gap ? "different" : (lo >= -BOOT_DELTA && hi <= BOOT_DELTA ? "equivalent" : "inconclusive")
        (pt && dec == "different" && abs(dlt) < BOOT_PT_MIN) && (dec = "inconclusive")
        lab = length(nm) == 2 ? "$(nm[1]) - $(nm[2])" : "($(nm[1]) - $(nm[2])) - ($(nm[3]) - $(nm[4]))"
        @printf("%s: %+.4f [%+.4f, %+.4f] %s\n", lab, dlt, lo, hi, dec)
        cd_ = length(nm) == 2 ? ["-", "-"] : nm[3:4]
        push!(con_rows, (A = nm[1], B = nm[2], C = cd_[1], D = cd_[2], delta = dlt, lo95 = lo, hi95 = hi, p_le0 = mean(v .<= 0),
                         decision = dec, pt_rule = pt ? "pt" : "-", seed_gap = Float64(seed_gap)))
    end
    isempty(con_rows) || write_curve(out * "_contrasts.tsv", con_rows)
    println("wrote $(out)_arms.tsv" * (isempty(con_rows) ? "" : " and $(out)_contrasts.tsv"))
    return arm_rows, con_rows
end
