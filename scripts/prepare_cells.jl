# builds data/tahoe_sc.jld2 for dataset = sc: the index of every cell of the raw parquet shards (read in place, no expression copied), train-cell gene statistics and the gene panel

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
using JLD2, Printf, Random, Statistics, StatsBase, TahoeFM
const M = TahoeFM

opts = Dict(String(k) => String(v) for (k, v) in (split(a, "="; limit=2) for a in ARGS))
isempty(setdiff(keys(opts), ("raw_dir", "pb", "out", "n_hvg", "stat_shards", "seed"))) || error("arguments: raw_dir= pb= out= n_hvg= stat_shards= seed=, got $(collect(keys(opts)))")
raw_dir = get(opts, "raw_dir", M.RAW_DIR); pb = get(opts, "pb", M.default_data_path()); out = get(opts, "out", M.default_sc_path())
n_hvg = parse(Int, get(opts, "n_hvg", "1024")); n_stat = parse(Int, get(opts, "stat_shards", "100")); seed = parse(Int, get(opts, "seed", "42"))
t0 = time(); el() = round(time() - t0, digits=1)
println("raw_dir $raw_dir | pseudobulk file $pb | out $out | n_hvg $n_hvg | stat_shards $n_stat | seed $seed")

# split, well annotations and gene rows come from the pseudobulk file
meta, sp, genes, pb_panels = jldopen(f -> (f["meta"], f["split"], f["genes"], f["panels"]), pb, "r")
G = length(genes.token); tokrow = M.token_rows(genes.token)

ix = M.build_cell_index(raw_dir; wells = unique(meta.well))
side, wm = M.well_table(meta, sp, ix.wells)
# cells per split side (index 1 = wells outside the table) and per (well, cell line)
function cell_counts(well::Vector{UInt16}, line::Vector{UInt8}, side::Vector{UInt8}, nw::Int, nl::Int)
    cs = zeros(Int, 6); cnt = zeros(Int, nw, nl)
    @inbounds for c in eachindex(well); cs[side[well[c]] + 1] += 1; cnt[well[c], line[c]] += 1; end
    return cs, cnt
end
n = length(ix.well); cs, cnt = cell_counts(ix.well, ix.line, side, length(ix.wells), length(ix.lines))
println("index ($(el()) s): $n cells in $(length(ix.shards)) shards, $(length(ix.wells)) wells, $(length(ix.lines)) cell lines")
println("  cells by the split side of their well: ", join(["$s $(cs[k + 1]) ($(round(100cs[k + 1] / n, digits=1)) %)" for (k, s) in enumerate(M.SPLIT_SIDES)], ", "), " | in wells outside the pseudobulk table: $(cs[1])")
# pseudobulk membership: every pseudobulk's cells are in the shards; cells of (well, cell line) pairs without a pseudobulk are kept (as the old single-cell code)
wid = Dict(w => i for (i, w) in enumerate(ix.wells)); lid = Dict(l => i for (i, l) in enumerate(ix.lines)); inpb = falses(size(cnt))
for i in eachindex(meta.well); haskey(lid, meta.cell_line[i]) && (inpb[wid[meta.well[i]], lid[meta.cell_line[i]]] = true); end
haskey(meta, :total_cells) && println("  pseudobulks whose cell count matches the shards: $(count(i -> haskey(lid, meta.cell_line[i]) && cnt[wid[meta.well[i]], lid[meta.cell_line[i]]] == meta.total_cells[i], eachindex(meta.well))) of $(length(meta.well))")
println("  cells outside every pseudobulk: $(sum(cnt[.!inpb])) in $(count(>(0), cnt[.!inpb])) (well, cell line) pairs (kept) | cell lines not in the pseudobulk table: $(setdiff(ix.lines, unique(meta.cell_line)))")

# statistics shards: the old compute_medians.jl sample (shuffle of the sorted shard list with MersenneTwister(seed))
pick = shuffle(MersenneTwister(seed), collect(1:length(ix.shards)))[1:min(n_stat, length(ix.shards))]
st = M.cell_stats(raw_dir, ix, tokrow, G, side .== 1, sort(pick))
sd = copy(st.sd); n0 = count(==(0f0), sd); sd[sd .== 0] .= 1f0
println("train statistics ($(el()) s): $(st.n_train) train cells of $(st.n_all) in $(length(pick)) shards | genes never detected in a train cell $(count(==(0f0), st.det)) (median 1), sd 0: $n0 (sd 1)")
# panel = the old compute_hvg.jl rule: the n_hvg genes with the highest variance of log1p CP10k over cells (here: train cells only); pb = the pseudobulk file's hvg panel, for reference
hvg = sort(sortperm(st.var; rev=true)[1:n_hvg]); panels = (hvg = hvg, pb = pb_panels.hvg)
println("  panel hvg: top $n_hvg variance genes, variance $(round(maximum(st.var[hvg]), digits=4)) to $(round(minimum(st.var[hvg]), digits=4)) | detection of the panel genes: median $(round(median(st.det[hvg]), digits=3)), min $(round(minimum(st.det[hvg]), digits=4)) | shared with the pseudobulk hvg panel: $(length(intersect(hvg, pb_panels.hvg)))")

# against the old single-cell files, when present (read only)
old_med = joinpath(raw_dir, "sc_gene_medians.jld2"); old_hvg = joinpath(dirname(raw_dir), "hvg_indices.jld2")
if isfile(old_med)
    om = load(old_med); same = Set(om["scan_shards"]) == Set(ix.shards[pick]); k = om["n_nonzero"] .> 0
    println("  old sc_gene_medians.jld2: same shards $same, cells $(om["n_cells_scanned"]) vs $(st.n_all) | all-cell medians max |diff| $(maximum(abs.(om["medians"][k] .- st.medians_all[k]))) | train-cell medians vs old: max |diff| $(round(maximum(abs.(om["medians"][k] .- st.medians[k])), digits=4)), r $(round(cor(om["medians"][k], st.medians[k]), digits=6))")
end
isfile(old_hvg) && println("  old hvg_indices.jld2 (other shards, all cells): $(length(intersect(load(old_hvg, "hvg_idx"), hvg))) of $n_hvg genes shared")

mkpath(dirname(out)); tmp = out * ".tmp"
jldopen(tmp, "w") do f
    for (k, v) in pairs((shards = ix.shards, off = ix.off, well = ix.well, line = ix.line, wells = ix.wells, lines = ix.lines, well_side = side, tokens = genes.token,
                         medians = st.medians, gene_mean = st.mean, gene_sd = sd, gene_det = st.det, gene_var = Float32.(st.var), medians_all = st.medians_all, panels = panels,
                         stat_shards = ix.shards[sort(pick)], n_stat_train = st.n_train, n_stat_all = st.n_all, n_hvg = n_hvg, seed = seed, raw_dir = raw_dir, pb = pb))
        f[String(k)] = v
    end
end
mv(tmp, out; force=true)
println("wrote $out ($(round(filesize(out) / 2^20, digits=1)) MiB) in $(el()) s")
