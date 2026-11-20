# builds data/tahoe_pb.jld2 from the raw Tahoe pseudobulk files in raw_dir (log1p CP10k, 19020 protein-coding genes): split v2, train stats, named gene panels, top-k tokens

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
using DataFrames, JLD2, PyCall, Random, Statistics, StatsBase, TahoeFM
const M = TahoeFM

opts = Dict(String(k) => String(v) for (k, v) in (split(a, "="; limit=2) for a in ARGS))
isempty(setdiff(keys(opts), ("raw_dir", "out", "n_hvg", "top_k", "holdout", "ood", "landmarks"))) || error("arguments: raw_dir= out= n_hvg= top_k= holdout= ood= landmarks=, got $(collect(keys(opts)))")
raw_dir = get(opts, "raw_dir", M.RAW_DIR)
out = get(opts, "out", joinpath(dirname(@__DIR__), "data", "tahoe_pb.jld2"))
n_hvg = parse(Int, get(opts, "n_hvg", "1024")); top_k = parse(Int, get(opts, "top_k", "1024"))
# holdout = "all" (default: every eligible drug loses one dose, half of the drugs to val, half to test) or the share of the remaining treated drug|dose groups sent to EACH of val and test
holdout = get(opts, "holdout", "all") == "all" ? "all" : parse(Float64, opts["holdout"])
ood_frac = parse(Float64, get(opts, "ood", "0.1"))
# LINCS landmark gene list (key genes of the LincsFM prepared file, read only); "none" skips the landmark panels
landmarks = get(opts, "landmarks", "/home/golem/scratch/chans/thesis/lincs/data/lincs.jld2")
const SPLIT_SEED = 42
raw = joinpath(raw_dir, "filtered_pseudobulks_alpha_10000.jld2"); hgnc = joinpath(raw_dir, "protein-coding_gene.txt")
vocab = joinpath(raw_dir, "metadata", "gene_vocabulary.jsonl")

# gene rows = Tahoe vocabulary genes whose symbol is an HGNC protein-coding symbol, in token order (as the pseudobulk builder)
function coding_genes(hgnc::AbstractString, vocab::AbstractString)
    lines = readlines(hgnc); col = findfirst(==("symbol"), split(lines[1], '\t'))
    sym = Set(split(l, '\t')[col] for l in lines[2:end])
    rx = r"\"ensembl_id\":\"([^\"]*)\",\"token_id\":(\d+),\"gene_symbol\":\"([^\"]*)\""
    rows = [(parse(Int, m[2]), String(m[1]), String(m[3])) for m in (match(rx, l) for l in eachline(vocab)) if m !== nothing && m[3] in sym]
    sort!(rows, by=first)
    return (symbol = last.(rows), ensembl = [r[2] for r in rows], token = first.(rows))
end

# string columns of a parquet file through pandas (needs pandas + pyarrow in the PyCall python); missing -> ""
function parquet_columns(path::AbstractString, cols)
    t = pyimport("pandas").read_parquet(path)
    return [String[x === nothing ? "" : string(x) for x in t.__getitem__(c).astype("object").where(t.__getitem__(c).notna(), nothing).tolist()] for c in cols]
end

# LINCS landmark genes as rows of the gene matrix: by symbol, else through the Entrez id and its current HGNC symbol
function landmark_rows(file::AbstractString, hgnc::AbstractString, symbols::Vector{String})
    lg = jldopen(f -> f["genes"], file, "r"); row = Dict(s => i for (i, s) in enumerate(symbols))
    lines = readlines(hgnc); hdr = split(lines[1], '\t'); ci = findfirst(==("symbol"), hdr); ce = findfirst(==("entrez_id"), hdr)
    e2s = Dict{String,String}()
    for l in lines[2:end]; f = split(l, '\t'); length(f) >= max(ci, ce) && f[ce] != "" && (e2s[String(f[ce])] = String(f[ci])); end
    by_sym = [get(row, s, 0) for s in lg.symbol]; r = [by_sym[i] > 0 ? by_sym[i] : get(row, get(e2s, string(lg.id[i]), ""), 0) for i in eachindex(by_sym)]
    return sort(unique(filter(>(0), r))), count(>(0), by_sym), String.(lg.symbol[r .== 0])
end

t0 = time(); el() = round(time() - t0, digits=1)
println("raw_dir $raw_dir | out $out | n_hvg $n_hvg | top_k $top_k | holdout $holdout | ood $ood_frac | seed $SPLIT_SEED | $(Threads.nthreads()) threads")
df = load(raw, "df")
str(v) = String.(string.(v))
meta = (cell_line = str(df.cell_line), well = str(df.sample), plate = str(df.plate), drug = str(df.drug), dose = str(df.dose),
        total_cells = Vector{Int}(df.total_cells), total_umis = Vector{Float32}(df.total_umis))
all(df.qc_pass) && all(str(df.time) .== "24 h") || error("unexpected qc_pass / time values")
X = reduce(hcat, df.expr)::Matrix{Float32}
df = nothing; GC.gc()
G, N = size(X)
println("pseudobulks: $G genes x $N samples ($(el()) s)")

genes = coding_genes(hgnc, vocab)
length(genes.symbol) == G || error("$(length(genes.symbol)) coding genes in the vocabulary, X has $G rows")
println("genes: $G from $(basename(vocab)) & $(basename(hgnc)), tokens $(extrema(genes.token))")

# plate position = sample number minus the first sample number of the plate (96 consecutive ids per plate); block from the plate; moa-fine class of the drug
s_well, s_plate = parquet_columns(joinpath(raw_dir, "metadata", "sample_metadata.parquet"), ("sample", "plate"))
sn(s) = parse(Int, replace(s, "smp_" => ""))
first_id = Dict{String,Int}(); for (w, p) in zip(s_well, s_plate); first_id[p] = min(get(first_id, p, typemax(Int)), sn(w)); end
wp = Dict(zip(s_well, s_plate)); all(wp[w] == p for (w, p) in zip(meta.well, meta.plate)) || error("a pseudobulk well is on another plate in sample_metadata")
pos = [sn(meta.well[i]) - first_id[meta.plate[i]] for i in 1:N]
all(in(0:95), pos) && all(==(96), values(countmap(s_plate))) || error("plate positions outside 0:95 or a plate without 96 samples")
block = [M.PLATE_BLOCK[p] for p in meta.plate]
d_drug, d_moa = parquet_columns(joinpath(raw_dir, "metadata", "drug_metadata.parquet"), ("drug", "moa-fine"))
moa_of = Dict(strip(d) => m for (d, m) in zip(d_drug, d_moa))
GC.gc(); GC.gc()   # finalise the pandas objects on the main thread (a PyCall finalizer run from a worker thread segfaults)
dmso = meta.drug .== "DMSO"
moa = String[get(moa_of, strip(d), "") for d in meta.drug]
# group = drug|dose for treated samples (all wells / plates of a drug-dose together), plate|well for DMSO
group = [dmso[i] ? meta.plate[i] * "|" * meta.well[i] : meta.drug[i] * "|" * meta.dose[i] for i in 1:N]
meta = merge(meta, (group = group, moa = moa, pos = pos, block = block))
drugs = sort(unique(meta.drug[.!dmso]))
println("meta: $(length(drugs)) drugs, $(count(d -> get(moa_of, strip(d), "") == "", drugs)) without a drug_metadata row, moa-fine \"unclear\" for $(count(d -> get(moa_of, strip(d), "") == "unclear", drugs)), $(length(unique(moa[.!dmso])) - 1) named classes")

# structure the split relies on: one drug|dose and one position per well; plates of a block share their drugs, plates of different blocks do not
wk = Dict{String,Set{Tuple{String,Int}}}(); for i in 1:N; push!(get!(wk, meta.well[i], Set{Tuple{String,Int}}()), (group[i], pos[i])); end
all(length(v) == 1 for v in values(wk)) || error("a well holds more than one drug|dose or position")
pd_ = Dict(p => Set(meta.drug[(meta.plate .== p) .& .!dmso]) for p in unique(meta.plate)); pl = sort([p for p in keys(pd_) if M.PLATE_BLOCK[p] > 0])
ov = [(M.PLATE_BLOCK[a] == M.PLATE_BLOCK[b], length(intersect(pd_[a], pd_[b]))) for a in pl for b in pl if a < b]
w_min = minimum(last(o) for o in ov if first(o)); a_max = maximum(last(o) for o in ov if !first(o))
w_min > a_max || error("plate blocks do not match the drug layout: within-block overlap $w_min, across $a_max")
println("layout: $(length(wk)) wells, one drug|dose and position each; drugs shared by two plates of a block >= $w_min, of different blocks <= $a_max")

# split v2
sv = M.split_v2(meta; seed = SPLIT_SEED, ood_frac, holdout)
sp = sv.split; side = zeros(Int, N); for (k, s) in enumerate(M.SPLIT_SIDES); side[getfield(sp, s)] .= k; end
all(>(0), side) || error("split does not cover every sample")
println("split v2: control-like drugs $(sv.ctrl_drugs) | drugs confined to $(M.RESERVED_PLATE): $(length(sv.confined_drugs)) | ood drugs $(length(sv.ood_drugs)): $(join(sv.ood_drugs, "; "))")
doses = sort(unique(meta.dose[.!dmso]); by = M.dose_value)
for (k, s) in enumerate(M.SPLIT_SIDES)
    ii = side .== k; tt = ii .& .!dmso; gs = unique(group[tt])
    println("  $s: PBs $(count(ii)) ($(round(100count(ii) / N, digits=1)) %) | treated groups $(length(gs)) drugs $(length(unique(meta.drug[tt]))) wells $(length(unique(meta.well[ii]))) | DMSO wells $(length(unique(meta.well[ii .& dmso]))) PBs $(count(ii .& dmso)) | groups by dose ",
            join(["$d $(count(g -> endswith(g, "|" * d), gs))" for d in doses], ", "), " | PBs by dose ", join(["$d $(count(tt .& (meta.dose .== d)))" for d in doses], ", "), " | plates $(length(unique(meta.plate[ii]))) cell lines $(length(unique(meta.cell_line[ii])))")
end
sets(f) = [Set(f[side .== k]) for k in 1:5]
W = sets(meta.well); Gs = sets(group); D = sets(meta.drug)
sum(length(intersect(W[a], W[b])) for a in 1:5 for b in a+1:5) == 0 || error("a well is on two sides of the split")
sum(length(intersect(Gs[a], Gs[b])) for a in 1:4 for b in a+1:4) == 0 || error("a group straddles train / val / test / ood")
isempty(intersect(setdiff(D[2], ["DMSO"]), D[3])) && isempty(intersect(D[4], union(D[1], D[2], D[3], D[5]))) || error("val and test share a drug, or an ood drug is elsewhere")
ntr = Dict(d => length(unique(meta.dose[(meta.drug .== d) .& (side .== 1)])) for d in drugs if !(d in D[4]))
nav = Dict(d => length(unique(meta.dose[(meta.drug .== d) .& (side .<= 3)])) for d in keys(ntr))
all(ntr[d] >= min(2, nav[d]) for d in keys(ntr)) || error("a drug keeps fewer than 2 doses in train")
println("  non-ood drugs by dose groups in train: ", sort(collect(countmap(collect(values(ntr))))), " | drugs with 2 dose groups off $(M.RESERVED_PLATE) (never lose one): $(count(==(2), values(nav)))")
println("  DMSO wells out of train: ", sort([(meta.plate[i], meta.well[i], String(M.SPLIT_SIDES[side[i]])) for i in unique(i -> meta.well[i], findall(dmso .& (side .∈ Ref((2, 3)))))]))

# per-gene stats on the train split: RTF medians, ETF mean / sd (guards: median 1 if never nonzero, sd 1 if 0)
st = M.gene_stats(X, sp.train)
medians, gene_mean, gene_sd = st.medians, st.mean, st.sd
cl = (u = sort(unique(meta.cell_line)); id = Dict(c => i for (i, c) in enumerate(u)); [id[c] for c in meta.cell_line])
mo = M.train_moments(X, sp.train; cls = cl)
println("train stats ($(el()) s): $(count(==(0), mo.det)) genes never nonzero (median set to 1), $(count(==(0f0), gene_sd)) genes with sd 0 (sd set to 1), $(count(>=(0.9), mo.det)) genes detected in >= 90 % of train PBs")
gene_sd[gene_sd .== 0] .= 1f0

# named panels, all from the train moments
panels = M.gene_panels(mo, n_hvg)
if landmarks != "none"
    lm, n_sym, lost = landmark_rows(landmarks, hgnc, genes.symbol)
    have = Set(lm); pad = [g for g in sortperm(mo.var_within; rev=true, alg=MergeSort) if !(g in have)][1:max(0, n_hvg - length(lm))]
    panels = merge(panels, (landmark = lm, landmark_pad = sort(vcat(lm, pad))))
    println("landmark: $(length(lm)) LINCS landmark genes ($n_sym by symbol, $(length(lm) - n_sym) more through Entrez), not in the matrix: $lost; landmark_pad adds the $(length(pad)) top within-cell-line-variance genes")
end
sd64 = sqrt.(mo.var); tr5 = sp.train[1:5:end]
for (k, p) in pairs(panels)
    z = [count(==(0f0), view(X, p, j)) for j in tr5]
    println("  panel $k: $(length(p)) genes | train zero fraction $(round(sum(z) / (length(p) * length(tr5)), digits=4)) (per PB median $(median(z)), max $(maximum(z))) | min train sd $(round(minimum(sd64[p]), digits=4)) | min detection $(round(minimum(mo.det[p]), digits=3))")
end
println("  panel overlaps: ", join(["$a&$b $(length(intersect(panels[a], panels[b])))" for a in keys(panels) for b in keys(panels) if a < b], ", "))
vs = sort(sd64, rev=true); println("  var panel boundary: sd rank $n_hvg = $(vs[n_hvg]), rank $(n_hvg + 1) = $(vs[n_hvg + 1])")

# top-k tokens: per PB the top_k genes by x / median over all genes, rank order; no padding
nnz = [count(>(0f0), view(X, :, j)) for j in 1:N]
println("nonzero genes per PB: min $(minimum(nnz)), median $(median(nnz)), max $(maximum(nnz)) (top_k $top_k)")
minimum(nnz) >= top_k || error("a pseudobulk has $(minimum(nnz)) nonzero genes < top_k = $top_k: padding is not implemented")
tk_ids = M.topk_genes(X, medians, top_k); tk_x = M.topk_values(X, tk_ids)
println("top-k tokens ($(el()) s): $(length(unique(tk_ids))) distinct genes, value at rank $top_k >= $(minimum(view(tk_x, top_k, :)))")
# topk_rule = "raw": per PB the top_k genes by raw x, descending x, ties to the lower gene index
tkr_ids = M.topk_genes_raw(X, top_k); tkr_x = M.topk_values(X, tkr_ids)
println("raw top-k tokens ($(el()) s): $(length(unique(tkr_ids))) distinct genes, value at rank $top_k >= $(minimum(view(tkr_x, top_k, :)))")

mkpath(dirname(out)); tmp = out * ".tmp"
jldopen(tmp, "w") do f
    f["X"] = X
    for (k, p) in pairs(panels); f["X_panel/$k"] = X[p, :]; end
    for (k, v) in pairs((; meta, genes, medians, gene_mean, gene_sd, gene_det = Float32.(mo.det), split = sp, ood_drugs = sv.ood_drugs, panels, topk_ids = tk_ids, topk_x = tk_x,
                         topk_raw_ids = tkr_ids, topk_raw_x = tkr_x, top_k, n_hvg,
                         holdout = string(holdout), ood_frac, split_seed = SPLIT_SEED, split_version = 2, source = raw))
        f[String(k)] = v
    end
end
mv(tmp, out; force=true)
println("wrote $out ($(round(filesize(out) / 2^30, digits=2)) GiB) in $(el()) s")
