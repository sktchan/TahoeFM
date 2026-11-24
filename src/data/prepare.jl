# data preparation pieces shared by scripts/prepare_data.jl and the tests: split v2, train moments, named gene panels, per-sample top-k genes

# drug block of a plate (dose-level plates that carry the same drugs at the same positions); 0 = plate 13, the mixed-dose plate
const PLATE_BLOCK = Dict("plate1" => 1, "plate2" => 1, "plate3" => 1, "plate4" => 2, "plate5" => 2, "plate6" => 2, "plate14" => 2,
                         "plate7" => 3, "plate8" => 3, "plate9" => 3, "plate10" => 4, "plate11" => 4, "plate12" => 4, "plate13" => 0)
const RESERVED_PLATE = "plate13"
const SPLIT_SIDES = (:train, :val, :test, :ood, :reloc)

dose_value(s::AbstractString) = parse(Float64, first(split(s)))

# split v2 by construction (one rng stream): ood = ood_frac of the drugs with a well off the reserved plate + the drugs confined to it; reloc = other wells of the reserved plate; per remaining drug at most one dose group to val or test; at most one DMSO well per plate leaves train
function split_v2(meta::NamedTuple; seed::Integer = 42, ood_frac::Real = 0.1, holdout::Union{Real,AbstractString} = "all", dmso_frac::Real = 0.1,
                  reserved::AbstractString = RESERVED_PLATE)
    N = length(meta.drug); rng = Xoshiro(seed)
    dmso = meta.drug .== "DMSO"; res = meta.plate .== reserved
    drugs = sort(unique(meta.drug[.!dmso]))
    home = Set(meta.drug[.!dmso .& .!res])
    confined = [d for d in drugs if !(d in home)]
    # control-like drugs (one of their drug-doses is on every plate) are never ood and never lose a dose
    np = length(unique(meta.plate)); gp = Dict{Tuple{String,String},Set{String}}()
    for i in 1:N; dmso[i] || push!(get!(gp, (meta.drug[i], meta.dose[i]), Set{String}()), meta.plate[i]); end
    ctrl = sort(unique(k[1] for (k, p) in gp if length(p) == np))
    pool = [d for d in drugs if d in home && !(d in ctrl)]
    n_ood = round(Int, ood_frac * length(home))
    ood = Set(vcat(shuffle(rng, pool)[1:n_ood], confined))
    side = Int8[!dmso[i] && meta.drug[i] in ood ? 4 : res[i] ? 5 : 1 for i in 1:N]
    # dose groups left to each drug off the reserved plate, by dose level
    dg = Dict{String,Vector{String}}()
    for i in 1:N; side[i] == 1 && !dmso[i] && (v = get!(dg, meta.drug[i], String[]); meta.dose[i] in v || push!(v, meta.dose[i])); end
    foreach(v -> sort!(v; by = dose_value), values(dg))
    n_rest = sum(length, values(dg))
    cand = shuffle(rng, sort([d for (d, v) in dg if length(v) >= 3 && !(d in ctrl)]))
    n_val, n_test = holdout == "all" ? (cld(length(cand), 2), fld(length(cand), 2)) : (h = round(Int, Float64(holdout) * n_rest); (h, h))
    n_val + n_test <= length(cand) || error("holdout $holdout needs $(n_val + n_test) drugs with 3 dose groups, $(length(cand)) available")
    # drugs in pairs (val, test) that lose the same dose level; the level cycles over the pairs
    held = Dict{Tuple{String,String},Int8}()
    for j in 1:n_val + n_test
        v = dg[cand[j]]; held[(cand[j], v[mod1(cld(j, 2), length(v))])] = isodd(j) ? 2 : 3
    end
    for i in 1:N; side[i] == 1 && !dmso[i] && (side[i] = get(held, (meta.drug[i], meta.dose[i]), Int8(1))); end
    # DMSO wells off the reserved plate: one well of a plate at most leaves train
    dw = Dict{String,Vector{String}}()
    for i in 1:N; side[i] == 1 && dmso[i] && (v = get!(dw, meta.plate[i], String[]); meta.well[i] in v || push!(v, meta.well[i])); end
    n_d = floor(Int, dmso_frac * sum(length, values(dw); init = 0))
    2n_d <= length(dw) || error("dmso_frac $dmso_frac needs $(2n_d) plates with DMSO, $(length(dw)) available")
    out = Dict{String,Int8}()
    for (j, p) in enumerate(shuffle(rng, sort(collect(keys(dw))))[1:2n_d]); out[rand(rng, sort(dw[p]))] = j <= n_d ? 2 : 3; end
    for i in 1:N; side[i] == 1 && dmso[i] && (side[i] = get(out, meta.well[i], Int8(1))); end
    f(k) = shuffle(rng, findall(==(k), side))
    return (split = (train = f(1), val = f(2), test = f(3), ood = f(4), reloc = f(5)), ood_drugs = sort(collect(ood)), ctrl_drugs = ctrl, confined_drugs = confined)
end

# Float64 train moments per gene in a fixed number of column chunks (same result for any thread count): mean / variance of the log values and of expm1 (CP10k), detection rate, variance within the classes cls (one id per sample)
function train_moments(X::AbstractMatrix{<:Real}, cols::AbstractVector{<:Integer}; cls::Union{Nothing,AbstractVector{<:Integer}} = nothing, chunks::Integer = 16)
    G = size(X, 1); c = sort(cols); n = length(c); K = isnothing(cls) ? 0 : maximum(cls); kid = isnothing(cls) ? Int[] : Vector{Int}(cls)
    parts = collect(Iterators.partition(1:n, cld(n, chunks)))
    acc = [zeros(G, 5 + K) for _ in parts]
    Threads.@threads for q in eachindex(parts)
        moments_chunk!(acc[q], X, c, parts[q], kid)
    end
    A = reduce(+, acc); s2 = A[:, 2]
    mu = A[:, 1] ./ n; lm = A[:, 3] ./ n
    # CP10k variance in scanpy's order of operations (mean of squares minus squared mean, times n / (n - 1))
    va = max.((s2 .- n .* mu .^ 2) ./ (n - 1), 0); lv = max.((A[:, 4] ./ n .- lm .^ 2) .* (n / (n - 1)), 0)
    vw = K == 0 ? Float64[] : (nk = counts(kid[c], K); max.((s2 .- vec(sum(A[:, 5 .+ findall(>(0), nk)] .^ 2 ./ nk[nk .> 0]', dims=2))) ./ (n - count(>(0), nk)), 0))
    return (n = n, mean = mu, var = va, lin_mean = lm, lin_var = lv, det = A[:, 5] ./ n, var_within = vw)
end

# sums of one column chunk: x, x^2, expm1(x), expm1(x)^2, x > 0, then x per class
function moments_chunk!(A::Matrix{Float64}, X::AbstractMatrix{<:Real}, c::Vector{Int}, rng, kid::Vector{Int})
    G = size(X, 1)
    @inbounds for i in rng
        j = c[i]; k = isempty(kid) ? 0 : 5 + kid[j]
        for g in 1:G
            x = Float64(X[g, j]); e = expm1(x)
            A[g, 1] += x; A[g, 2] += x * x; A[g, 3] += e; A[g, 4] += e * e; A[g, 5] += x > 0
            k > 0 && (A[g, k] += x)
        end
    end
    return A
end

# the n highest-scoring genes, sorted by index; ties go to the lower gene index, NaN counts as -Inf
function top_genes(score::AbstractVector{<:Real}, n::Integer)
    s = [isnan(v) ? -Inf : Float64(v) for v in score]
    return sort(sortperm(s; rev=true, alg=MergeSort)[1:n])
end

# normalised dispersion as scanpy highly_variable_genes flavor "seurat" (Julia re-implementation, checked against scanpy in logs/port_hvg): log(var / mean) of the CP10k values, z-scored inside 20 equal-width bins of log1p(mean) (pandas.cut edges); a one-gene bin gets 1
function seurat_dispersion(lin_mean::AbstractVector{Float64}, lin_var::AbstractVector{Float64}; n_bins::Integer = 20)
    n = length(lin_mean); m = copy(lin_mean); m[m .== 0] .= 1e-12
    dsp = lin_var ./ m; dsp[dsp .== 0] .= NaN; ld = log.(dsp); lm = log1p.(m)
    lo, hi = extrema(lm); st = (hi - lo) / n_bins; edges = [lo + i * st for i in 0:n_bins]; edges[end] = hi; edges[1] = lo - 0.001 * (hi - lo)
    bin = [clamp(searchsortedfirst(edges, v) - 1, 1, n_bins) for v in lm]; nd = fill(-Inf, n)
    for b in 1:n_bins
        gi = findall(i -> bin[i] == b && !isnan(ld[i]), 1:n); isempty(gi) && continue
        bm = mean(ld[gi]); bs = length(gi) > 1 ? std(ld[gi]) : NaN
        isnan(bs) && (bs = bm; bm = 0.0)
        nd[gi] .= (ld[gi] .- bm) ./ bs
    end
    return nd
end

# dispersion panel on the genes detected in >= min_det of the train samples (the filter_genes step before HVG)
function dispersion_panel(mo::NamedTuple, n::Integer; min_det::Real = 0.9)
    cand = findall(>=(min_det), mo.det)
    return cand[top_genes(seurat_dispersion(mo.lin_mean[cand], mo.lin_var[cand]), n)]
end

# local quadratic fit of y on x at every point (tricube weights over the nearest span * n points)
function loess_fit(x::Vector{Float64}, y::Vector{Float64}; span::Real = 0.3)
    n = length(x); o = sortperm(x); xs = x[o]; ys = y[o]; q = min(n, ceil(Int, span * n)); fit = similar(xs)
    Threads.@threads for i in 1:n
        l = i; r = i
        while r - l + 1 < q
            if l == 1; r += 1
            elseif r == n; l -= 1
            elseif xs[i] - xs[l-1] <= xs[r+1] - xs[i]; l -= 1
            else; r += 1; end
        end
        h = max(xs[i] - xs[l], xs[r] - xs[i], 1e-12)
        A = zeros(3, 3); b = zeros(3)
        @inbounds for k in l:r
            u = (xs[k] - xs[i]) / h; w = max(1 - abs(u)^3, 0.0)^3; p = (1.0, u, u * u)
            for a in 1:3
                b[a] += w * p[a] * ys[k]
                for c in 1:3; A[a, c] += w * p[a] * p[c]; end
            end
        end
        fit[i] = (pinv(A) * b)[1]
    end
    out = similar(fit); out[o] = fit
    return out
end

# scran-style score: variance of the log values above its trend on the mean
function scran_score(mu::AbstractVector{Float64}, va::AbstractVector{Float64})
    ok = va .> 0; bio = fill(-Inf, length(va))
    bio[ok] = va[ok] .- exp.(loess_fit(mu[ok], log.(va[ok])))
    return bio
end

# label-free named panels from the train moments: var (top variance of the log values), hvg (dispersion, detection >= 0.9), disp (dispersion, no filter), scran
gene_panels(mo::NamedTuple, n::Integer) = (var = top_genes(mo.var, n), hvg = dispersion_panel(mo, n; min_det=0.9),
                                           disp = dispersion_panel(mo, n; min_det=0.0), scran = top_genes(scran_score(mo.mean, mo.var), n))

# per sample, the K genes with the highest x / median in rank order (rows 1..K = ranks 1..K of gene_ranks over all genes)
function topk_genes(X::AbstractMatrix{<:Real}, medians::AbstractVector{<:Real}, K::Integer)
    G, N = size(X); ids = Matrix{Int32}(undef, K, N)
    Threads.@threads for ch in collect(Iterators.partition(1:N, cld(N, 4 * Threads.nthreads())))
        v = Vector{Float32}(undef, G); p = Vector{Int}(undef, G)
        @inbounds for j in ch
            for g in 1:G; v[g] = X[g, j] / medians[g]; end
            sortperm!(p, v, rev=true)
            for r in 1:K; ids[r, j] = Int32(p[r]); end
        end
    end
    return ids
end

# topk_rule = "raw": per sample, the K genes with the highest raw x in descending-x order, ties to the lower gene index (as topk_genes with median 1)
topk_genes_raw(X::AbstractMatrix{<:Real}, K::Integer) = topk_genes(X, ones(Float32, size(X, 1)), K)

# values of the top-k genes of each sample (K x N)
topk_values(X::AbstractMatrix{<:Real}, ids::AbstractMatrix{<:Integer}) = Float32[X[ids[r, j], j] for r in axes(ids, 1), j in axes(ids, 2)]
