# prepared Tahoe pseudobulk data (scripts/prepare_data.jl), gene sets, labels and splits

struct TahoeData
    X::Matrix{Float32}        # genes x samples for the loaded gene set (a named panel or all genes), log1p CP10k as stored
    meta::NamedTuple          # per-sample vectors (cell_line, well, plate, drug, dose, group, moa, pos, block, total_cells, total_umis)
    genes::NamedTuple         # symbol, ensembl, token of the rows of X
    medians::Vector{Float32}  # per-gene median of nonzero values, canonical train split (RTF ranking); 1 if never nonzero
    gene_mean::Vector{Float32} # per-gene mean, canonical train split (ETF standardization)
    gene_sd::Vector{Float32}   # per-gene sd, canonical train split; 1 where the train sd is 0
    split::NamedTuple         # canonical train / val / test / ood / reloc sample indices (split v2, seed 42)
    gene_idx::Vector{Int}     # rows of X in the full 19020-gene matrix
    panel::Vector{Int}        # the selected panel (full-matrix rows, sorted)
    panel_name::String        # its name in panels
    panels::NamedTuple        # every stored panel (var, hvg, disp, scran, landmark, landmark_pad), all from the train split
    full::NamedTuple          # all-gene symbol, medians, gene_mean, gene_sd (what top-k token ids index)
    topk::NamedTuple          # ids (K x N Int32, rank order), x (their values) and rule (topk_rule); empty unless loaded with topk = true
    path::String              # the prepared file, for the accessors that read on demand
    cells::Any                # single cells: the CellSource the columns were read from (src/data/cells.jl); nothing for pseudobulks
end
TahoeData(X, meta, genes, medians, gene_mean, gene_sd, split, gene_idx, panel, panel_name, panels, full, topk, path) =
    TahoeData(X, meta, genes, medians, gene_mean, gene_sd, split, gene_idx, panel, panel_name, panels, full, topk, path, nothing)

is_sc(d) = false
is_sc(d::TahoeData) = !isnothing(d.cells)
dataset_name(d::TahoeData) = is_sc(d) ? "sc" : "pb"

default_data_path() = joinpath(PKG_DIR, "data", "tahoe_pb.jld2")

# the raw Tahoe-100M files (pseudobulk table, single-cell parquet shards, vocabulary, metadata): read in place, never copied
const RAW_DIR = get(ENV, "TAHOE_RAW_DIR", "/home/golem/scratch/chans/transformers/data/tahoe")

# top-k selection rules: median = by x / train median (Geneformer, the default), raw = by raw x; the prepared-file keys of each
const TOPK_RULES = ("median", "raw")
check_topk_rule(rule::AbstractString) = (rule in TOPK_RULES || error("topk_rule must be $(join(TOPK_RULES, " or ")), got $rule"); String(rule))
topk_keys(rule::AbstractString) = check_topk_rule(rule) == "median" ? ("topk_ids", "topk_x") : ("topk_$(rule)_ids", "topk_$(rule)_x")

# genes = "panel" (the named panel, for transformers and panel baselines) or "all" (every gene, for all-gene baselines); topk = true also loads the top-k tokens of topk_rule
function load_data(path::AbstractString = default_data_path(); genes::AbstractString = "panel", panel::AbstractString = "hvg", topk::Bool = false,
                   topk_rule::AbstractString = "median")
    genes in ("panel", "all") || error("genes must be panel or all, got $genes")
    ki, kx = topk_keys(topk_rule)
    return jldopen(path, "r") do f
        panels = f["panels"]
        haskey(panels, Symbol(panel)) || error("panel must be one of $(join(keys(panels), ", ")), got $panel")
        p = panels[Symbol(panel)]; med = f["medians"]; mu = f["gene_mean"]; sd = f["gene_sd"]; gn = f["genes"]
        gi = genes == "panel" ? p : collect(1:length(med))
        X = genes == "panel" ? f["X_panel/$panel"] : f["X"]
        topk && !haskey(f, ki) && error("$path has no $ki (topk_rule = $topk_rule): rebuild it with scripts/prepare_data.jl")
        tk = topk ? (ids = f[ki], x = f[kx], rule = String(topk_rule)) : (ids = Matrix{Int32}(undef, 0, 0), x = Matrix{Float32}(undef, 0, 0), rule = "")
        TahoeData(X, f["meta"], map(v -> v[gi], gn), med[gi], mu[gi], sd[gi], f["split"], gi, p, String(panel), panels,
                  (symbol = gn.symbol, medians = med, gene_mean = mu, gene_sd = sd), tk, String(path))
    end
end

is_all(d::TahoeData) = length(d.gene_idx) == length(d.full.medians)

# a panel view of the data: the subset of all-gene data, else a load of that panel from the prepared file
function panel_data(d::TahoeData; panel::AbstractString = d.panel_name)
    !is_all(d) && panel == d.panel_name && return d
    is_all(d) || return load_data(d.path; panel, topk = !isempty(d.topk.ids), topk_rule = loaded_topk_rule(d))
    p = d.panels[Symbol(panel)]
    return TahoeData(d.X[p, :], d.meta, map(v -> v[p], d.genes), d.medians[p], d.gene_mean[p], d.gene_sd[p], d.split, p, p, String(panel), d.panels, d.full, d.topk, d.path)
end

# panel x samples matrix of the selected panel
panel_matrix(d::TahoeData) = is_all(d) ? d.X[d.panel, :] : d.X

# all genes x samples (read from the prepared file unless the data was loaded with genes = "all")
all_matrix(d::TahoeData) = is_all(d) ? d.X : jldopen(f -> f["X"], d.path, "r")

# rule of the loaded top-k tokens ("median" for data built without one)
loaded_topk_rule(d) = (r = get(d.topk, :rule, "median"); isempty(r) ? "median" : String(r))

# top-k token gene ids (K x N, rows in rank order, ids index the full gene set) and the values of those genes, for topk_rule rule (default: the loaded one)
topk_ids(d::TahoeData, rule::AbstractString = loaded_topk_rule(d)) = !isempty(d.topk.ids) && loaded_topk_rule(d) == rule ? d.topk.ids : jldopen(f -> f[topk_keys(rule)[1]], d.path, "r")
topk_values(d::TahoeData, rule::AbstractString = loaded_topk_rule(d)) = !isempty(d.topk.x) && loaded_topk_rule(d) == rule ? d.topk.x : jldopen(f -> f[topk_keys(rule)[2]], d.path, "r")

# per-gene median of nonzero values, mean and sd over the given columns
function gene_stats(X::AbstractMatrix, cols::AbstractVector{<:Integer})
    G = size(X, 1); med = Vector{Float32}(undef, G); mu = similar(med); sd = similar(med)
    buf = Vector{Float32}(undef, length(cols))
    for g in 1:G
        n = 0
        for j in cols
            v = X[g, j]
            v > 0 && (n += 1; buf[n] = v)
        end
        med[g] = n == 0 ? 1f0 : Float32(median!(view(buf, 1:n)))
        r = view(X, g, cols); mu[g] = Float32(mean(r)); sd[g] = Float32(std(r))
    end
    return (medians = med, mean = mu, sd = sd)
end

n_genes(d::TahoeData) = size(d.X, 1)

# one mean / sd over all genes and train samples, from the per-gene stats
global_stats(d::TahoeData) = (m = mean(d.gene_mean); (m, sqrt(mean(d.gene_sd .^ 2 .+ d.gene_mean .^ 2) - m^2)))

# ETF input scaling with the train stats: raw, gene (per-gene z) or global (one mean / sd), clipped +-10
function etf_scale(X::AbstractMatrix, d::TahoeData, how::AbstractString)
    how == "raw" && return Matrix{Float32}(X)
    mu, sd = how == "gene" ? (d.gene_mean, d.gene_sd) : how == "global" ? Float32.(global_stats(d)) : error("etf_scale must be raw, gene or global, got $how")
    return clamp.((X .- mu) ./ sd, -10f0, 10f0)
end

# samples of the canonical train / val / test split (ood drugs and the reserved plate-13 wells excluded); reloc = true adds the plate-13 wells
keep_mask(d::TahoeData; reloc::Bool = false) = (k = falses(size(d.X, 2)); for v in (d.split.train, d.split.val, d.split.test); k[v] .= true; end; reloc && (k[d.split.reloc] .= true); k)

reloc_mask(d::TahoeData) = (k = falses(size(d.X, 2)); k[d.split.reloc] .= true; k)

# canonical split for pretraining: train / val / test only (ood and reloc are never pretrained on)
pretrain_split(d::TahoeData) = (train = d.split.train, val = d.split.val, test = d.split.test)

# lvl1 = cell line; lvl2 = drug (doses merged), DMSO excluded, drugs with >= min_n non-ood samples (reloc wells counted). returns sample idx (train / val / test, plus the reloc wells of these classes if reloc = true), class ids, class names
function labels(d::TahoeData, level::AbstractString; min_n::Integer = 100, reloc::Bool = false)
    is_sc(d) && return cell_labels(d, level; reloc)
    k = keep_mask(d; reloc)
    if level == "lvl1"
        idx = findall(k); lab = d.meta.cell_line[idx]
    elseif level == "lvl2"
        cnt = countmap(d.meta.drug[keep_mask(d; reloc = true) .& (d.meta.drug .!= "DMSO")])
        valid = Set(c for (c, n) in cnt if n >= min_n)
        idx = findall(i -> k[i] && d.meta.drug[i] in valid, eachindex(k))
        lab = d.meta.drug[idx]
    else
        error("level must be lvl1 or lvl2, got $level")
    end
    names = unique(lab[.!reloc_mask(d)[idx]])  # first-appearance order over train / val / test, so class ids do not depend on reloc
    id = Dict(n => i for (i, n) in enumerate(names))
    return idx, Int[id[l] for l in lab], names
end

# canonical split mapped to positions within idx (train, val, test, and reloc as an extra evaluation set: empty unless idx holds reloc wells)
function finetune_split(d::TahoeData, idx::AbstractVector{<:Integer})
    pos = Dict(s => j for (j, s) in enumerate(idx))
    f(v) = [pos[s] for s in v if haskey(pos, s)]
    return (train = f(d.split.train), val = f(d.split.val), test = f(d.split.test), reloc = f(d.split.reloc))
end

# class names of a level from the pseudobulk table alone (meta, split): the names labels() gives, in the same order
function class_names(meta::NamedTuple, sp::NamedTuple, level::AbstractString; min_n::Integer = 100)
    N = length(meta.drug); k = falses(N); for v in (sp.train, sp.val, sp.test); k[v] .= true; end
    level == "lvl1" && return unique(meta.cell_line[k])
    level == "lvl2" || error("level must be lvl1 or lvl2, got $level")
    kr = copy(k); kr[sp.reloc] .= true
    valid = Set(c for (c, n) in countmap(meta.drug[kr .& (meta.drug .!= "DMSO")]) if n >= min_n)
    return unique(d for d in meta.drug[k] if d in valid)
end

# single cells: classes come from the pseudobulk table (same ids as the pseudobulk runs; lvl1 adds the cell lines that only the cells have), every cell of a class is kept
function cell_labels(d::TahoeData, level::AbstractString; reloc::Bool = false)
    haskey(d.cells.classes, level) || error("level must be lvl1 or lvl2, got $level")
    names = d.cells.classes[level]; id = Dict(n => i for (i, n) in enumerate(names))
    k = keep_mask(d; reloc); lab = level == "lvl1" ? d.meta.cell_line : d.meta.drug
    idx = findall(i -> k[i] && haskey(id, lab[i]), eachindex(k))
    return idx, Int[id[lab[i]] for i in idx], names
end
