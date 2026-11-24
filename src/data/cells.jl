# Tahoe-100M single cells, read in place from the raw parquet shards (cfg raw_dir): cell index, per-cell log1p CP10k and top-k tokens, the in-memory cell sample, the training stream

default_sc_path() = joinpath(PKG_DIR, "data", "tahoe_sc.jld2")
shard_dir(raw_dir::AbstractString) = joinpath(raw_dir, "data")
list_shards(raw_dir::AbstractString) = sort(filter(f -> endswith(f, ".parquet"), readdir(shard_dir(raw_dir))))

# pyarrow.parquet, its thread pools capped to the job's cpus (default = every core of the node)
const PARQUET = Ref{Any}(nothing)
function parquet()
    if isnothing(PARQUET[])
        pa = PyCall.pyimport("pyarrow")
        nt = something(tryparse(Int, get(ENV, "LANES_CPUS", "")), tryparse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "")), 4)
        pa.set_cpu_count(nt); pa.set_io_thread_count(nt)
        pa.set_memory_pool(pa.system_memory_pool())   # malloc returns freed shard buffers; the default pool (mimalloc) kept them (308 shards: 37 GB vs 9 GB peak)
        PARQUET[] = PyCall.pyimport("pyarrow.parquet")
    end
    return PARQUET[]
end

# Julia copy of a 1-d numpy array, straight from its buffer (PyCall's own conversion of these read-only arrays goes element by element: 6-9 s per column of a shard)
function np_vector(::Type{T}, arr::PyCall.PyObject) where T
    string(arr.dtype.name) == lowercase(string(T)) || error("numpy dtype $(arr.dtype.name), expected $T")
    n = convert(Int, arr.size); out = Vector{T}(undef, n)
    n > 0 && unsafe_copyto!(pointer(out), Ptr{T}(convert(UInt, arr.__array_interface__["data"][1])), n)
    return out
end
np_vector(::Type{T}, col, field::Symbol) where T = np_vector(T, PyCall.pycall(getproperty(col, field).to_numpy, PyCall.PyObject))

# one shard: flat gene tokens and counts, per-cell offsets (cell i = entries off[i]+1 : off[i+1])
function read_shard(path::AbstractString)
    t = parquet().read_table(path, columns=["genes", "expressions"])
    g = t.column("genes").combine_chunks(); e = t.column("expressions").combine_chunks()
    return (genes = np_vector(Int64, g, :values), off = Int64.(np_vector(Int32, g, :offsets)), expr = np_vector(Float32, e, :values))
end

# the cells locs (1-based within the shard, sorted) of one shard: whole shard, or only the row groups that hold them when that is under half of the shard;
# returns the shard columns and each wanted cell's row in them (rows[l] for cell l, 0 = not read)
function read_shard_cells(path::AbstractString, locs::AbstractVector{<:Integer}, ncell::Int)
    pf = parquet().ParquetFile(path); md = pf.metadata; ng = convert(Int, md.num_row_groups)
    sz = [convert(Int, md.row_group(i).num_rows) for i in 0:ng-1]; st = cumsum([0; sz[1:end-1]])
    sum(sz) == ncell || error("shard $(basename(path)) has $(sum(sz)) cells, the index $ncell: rebuild it with scripts/prepare_cells.jl")
    need = sort(unique(searchsortedlast.(Ref(st), locs .- 1)))
    if 2 * length(need) >= ng
        sh = read_shard(path); return sh, collect(1:ncell)
    end
    t = pf.read_row_groups(need .- 1, columns=["genes", "expressions"])
    g = t.column("genes").combine_chunks(); e = t.column("expressions").combine_chunks()
    sh = (genes = np_vector(Int64, g, :values), off = Int64.(np_vector(Int32, g, :offsets)), expr = np_vector(Float32, e, :values))
    rows = zeros(Int, ncell); at = 0
    for k in need; rows[st[k]+1:st[k]+sz[k]] .= at .+ (1:sz[k]); at += sz[k]; end
    length(sh.off) - 1 == at || error("row groups of $(basename(path)) gave $(length(sh.off) - 1) cells, expected $at")
    return sh, rows
end

# well (sample) and cell line of every cell of a shard, as codes into the shard's own value lists
function read_shard_meta(path::AbstractString)
    t = parquet().read_table(path, columns=["sample", "cell_line_id"])
    f(name) = (c = t.column(name).combine_chunks().dictionary_encode(); (convert(Vector{Int32}, c.indices.to_numpy()) .+ Int32(1), convert(Vector{String}, c.dictionary.to_pylist())))
    return (well = f("sample"), line = f("cell_line_id"))
end

# index of every cell in shard order: well and cell-line codes (wells = the given names first, e.g. the pseudobulk table's, then any other)
function build_cell_index(raw_dir::AbstractString; wells::Vector{String} = String[], every::Int = 400)
    files = list_shards(raw_dir); wells = copy(wells); wid = Dict(w => i for (i, w) in enumerate(wells)); lines = String[]; lid = Dict{String,Int}()
    well = UInt16[]; line = UInt8[]; off = Int[0]; t0 = time()
    for (s, f) in enumerate(files)
        m = read_shard_meta(joinpath(shard_dir(raw_dir), f))
        wc = [get!(() -> (push!(wells, w); length(wells)), wid, w) for w in m.well[2]]
        lc = [get!(() -> (push!(lines, l); length(lines)), lid, l) for l in m.line[2]]
        append!(well, UInt16.(wc[m.well[1]])); append!(line, UInt8.(lc[m.line[1]])); push!(off, length(well))
        s % every == 0 && (@printf("  indexed %d / %d shards, %d cells, %.0f s\n", s, length(files), length(well), time() - t0); flush(stdout))
    end
    return (shards = files, off = off, well = well, line = line, wells = wells, lines = lines)
end

# vocabulary token + 1 -> gene row (0 = not one of the protein-coding rows)
function token_rows(tokens::AbstractVector{<:Integer})
    r = zeros(Int32, maximum(tokens) + 2)
    for (i, t) in enumerate(tokens); r[t+1] = i; end
    return r
end

# work buffers of one cell: gene rows, values, sort keys, order
struct CellBuf
    g::Vector{Int32}
    x::Vector{Float32}
    key::Vector{Float32}
    p::Vector{Int}
end
CellBuf(G::Int) = CellBuf(Vector{Int32}(undef, G), Vector{Float32}(undef, G), Vector{Float32}(undef, G), Vector{Int}(undef, G))

# detected protein-coding genes of cell i: rows b.g, values b.x = log1p(10000 count / coding total) (as the pseudobulks and the old cell_to_dense_flat!); returns their number and the coding total
function cell_values!(b::CellBuf, sh::NamedTuple, i::Int, tokrow::Vector{Int32})
    n = 0; tot = 0f0; nt = length(tokrow)
    @inbounds for j in sh.off[i]+1:sh.off[i+1]
        v = sh.expr[j]; v > 0f0 || continue          # the first entry of a cell is a negative marker
        t = sh.genes[j]; (0 <= t < nt) || continue
        r = tokrow[t+1]; r == 0 && continue
        n += 1; b.g[n] = r; b.x[n] = v; tot += v
    end
    sf = 10000f0 / tot
    @inbounds for k in 1:n; b.x[k] = log1p(sf * b.x[k]); end
    return n, tot
end

# top-K tokens of the buffered cell into column j: by x / key (key = train medians, or ones for the raw rule), ties to the lower gene row; slots past the detected genes are padding (gene 1, value 0); returns the real token count (>= 1)
function topk_cell!(ids::Matrix{Int32}, x::Matrix{Float32}, j::Int, b::CellBuf, n::Int, key::Vector{Float32})
    K = size(ids, 1); m = min(K, n); g = b.g; k = b.key
    @inbounds for q in 1:n; k[q] = b.x[q] / key[g[q]]; b.p[q] = q; end
    p = view(b.p, 1:n); lt = (a, c) -> @inbounds(k[a] > k[c] || (k[a] == k[c] && g[a] < g[c]))
    m == 0 || (m < n ? partialsort!(p, 1:m; lt = lt) : sort!(p; lt = lt))
    @inbounds for r in 1:m; ids[r, j] = g[p[r]]; x[r, j] = b.x[p[r]]; end
    @inbounds for r in m+1:K; ids[r, j] = Int32(1); x[r, j] = 0f0; end
    return Int32(max(m, 1))
end

function fill_cells!(X, ids, x, len, ndet, umis, sh::NamedTuple, cols, cells, base::Int, tokrow::Vector{Int32}, ppos::Vector{Int32}, key::Vector{Float32}, b::CellBuf;
                     rows = nothing)
    K = size(ids, 1); P = size(X, 1)
    for j in cols
        l = Int(cells[j]) - base
        n, tot = cell_values!(b, sh, isnothing(rows) ? l : rows[l], tokrow)
        ndet[j] = n; umis[j] = tot
        if P > 0
            @inbounds for q in 1:P; X[q, j] = 0f0; end
            @inbounds for k in 1:n; q = ppos[b.g[k]]; q > 0 && (X[q, j] = b.x[k]); end
        end
        K > 0 && (len[j] = topk_cell!(ids, x, j, b, n, key))
    end
end

# histogram of the nonzero values per gene (as the old compute_medians.jl) and the median read from it
const CELL_BINS = 2000
const CELL_BIN_W = 10f0 / CELL_BINS
function hist_medians(hist::Matrix{Int32}, nz::Vector{Int})
    med = ones(Float32, length(nz))
    for g in eachindex(nz)
        nz[g] == 0 && continue
        half = nz[g] / 2; cum = 0
        for bin in 1:size(hist, 1)
            c = hist[bin, g]
            if cum + c >= half
                med[g] = Float32((bin - 1 + (c == 0 ? 0.0 : (half - cum) / c)) * CELL_BIN_W); break
            end
            cum += c
        end
    end
    return med
end

function stats_shard!(a::NamedTuple, sh::NamedTuple, wells::AbstractVector{UInt16}, train_well::AbstractVector{Bool}, tokrow::Vector{Int32}, b::CellBuf)
    ntr = 0
    for i in 1:length(sh.off)-1
        n, _ = cell_values!(b, sh, i, tokrow); tr = train_well[wells[i]]; ntr += tr
        @inbounds for k in 1:n
            g = b.g[k]; v = b.x[k]; bin = clamp(ceil(Int, v / CELL_BIN_W), 1, CELL_BINS)
            a.h_all[bin, g] += Int32(1); a.nz_all[g] += 1
            tr || continue
            a.h_tr[bin, g] += Int32(1); a.nz_tr[g] += 1; a.s1[g] += v; a.s2[g] += Float64(v)^2
        end
    end
    return ntr
end

# per-gene statistics of log1p CP10k over the cells of the given shards: train cells (mean, sd, variance, detection, nonzero median) and all cells (nonzero median, comparable to the old sc_gene_medians.jld2)
function cell_stats(raw_dir::AbstractString, ix::NamedTuple, tokrow::Vector{Int32}, G::Int, train_well::AbstractVector{Bool}, shards::AbstractVector{<:Integer}; every::Int = 10)
    a = (h_all = zeros(Int32, CELL_BINS, G), h_tr = zeros(Int32, CELL_BINS, G), nz_all = zeros(Int, G), nz_tr = zeros(Int, G), s1 = zeros(Float64, G), s2 = zeros(Float64, G))
    b = CellBuf(G); ntr = 0; nall = 0; t0 = time()
    for (q, s) in enumerate(shards)
        sh = read_shard(joinpath(shard_dir(raw_dir), ix.shards[s])); nc = length(sh.off) - 1
        nc == ix.off[s+1] - ix.off[s] || error("shard $(ix.shards[s]) has $nc cells, the index $(ix.off[s+1] - ix.off[s])")
        ntr += stats_shard!(a, sh, view(ix.well, ix.off[s]+1:ix.off[s+1]), train_well, tokrow, b); nall += nc
        q % every == 0 && (@printf("  stats %d / %d shards, %d train cells of %d, %.0f s\n", q, length(shards), ntr, nall, time() - t0); flush(stdout))
    end
    mu = a.s1 ./ ntr; va = max.(a.s2 .- ntr .* mu .^ 2, 0.0) ./ (ntr - 1)
    return (medians = hist_medians(a.h_tr, a.nz_tr), medians_all = hist_medians(a.h_all, a.nz_all), mean = Float32.(mu), sd = Float32.(sqrt.(va)), var = va,
            det = Float32.(a.nz_tr ./ ntr), n_train = ntr, n_all = nall)
end

# hvg moments of the train cells of the given shards: per gene, raw-count sum / sum of squares and CP10k (linear) sum / sum of squares; n = train cells
function hvg_moments_shard!(a, sh::NamedTuple, wells::AbstractVector{UInt16}, train_well::AbstractVector{Bool}, tokrow::Vector{Int32}, b::CellBuf)
    nt = length(tokrow); ntr = 0
    for i in 1:length(sh.off)-1
        train_well[wells[i]] || continue
        n = 0; tot = 0f0
        @inbounds for j in sh.off[i]+1:sh.off[i+1]
            v = sh.expr[j]; v > 0f0 || continue
            t = sh.genes[j]; (0 <= t < nt) || continue
            r = tokrow[t+1]; r == 0 && continue
            n += 1; b.g[n] = r; b.x[n] = v; tot += v
        end
        ntr += 1; tot > 0 || continue
        @inbounds for k in 1:n
            g = b.g[k]; c = Float64(b.x[k]); e = 10000.0 * c / tot
            a.c1[g] += c; a.c2[g] += c * c; a.e1[g] += e; a.e2[g] += e * e
        end
    end
    return ntr
end

# seurat_v3 second pass: per gene, sum / sum of squares of the counts clipped at mean + sd_fit * sqrt(n) (nonzero counts only)
function v3_shard!(a, sh::NamedTuple, wells::AbstractVector{UInt16}, train_well::AbstractVector{Bool}, tokrow::Vector{Int32}, clip::Vector{Float64})
    nt = length(tokrow)
    for i in 1:length(sh.off)-1
        train_well[wells[i]] || continue
        @inbounds for j in sh.off[i]+1:sh.off[i+1]
            v = sh.expr[j]; v > 0f0 || continue
            t = sh.genes[j]; (0 <= t < nt) || continue
            r = tokrow[t+1]; r == 0 && continue
            x = min(Float64(v), clip[r]); a.x1[r] += x; a.x2[r] += x * x
        end
    end
end

# hvg panels of the train cells, as scanpy highly_variable_genes: seurat (normalised dispersion of CP10k in 20 mean bins; the rule of the pseudobulk hvg panel, without its
# detection filter, which no gene passes on cells) and seurat_v3 (raw counts: variance of counts standardised by a loess fit of log variance on log mean, clipped at sqrt(n))
function cell_hvg(raw_dir::AbstractString, ix::NamedTuple, tokrow::Vector{Int32}, G::Int, train_well::AbstractVector{Bool}, shards::AbstractVector{<:Integer}, n_hvg::Int)
    a = (c1 = zeros(G), c2 = zeros(G), e1 = zeros(G), e2 = zeros(G)); b = CellBuf(G); n = 0; t0 = time()
    rd(s) = (sh = read_shard(joinpath(shard_dir(raw_dir), ix.shards[s])); (sh, view(ix.well, ix.off[s]+1:ix.off[s+1])))
    for s in shards; sh, w = rd(s); n += hvg_moments_shard!(a, sh, w, train_well, tokrow, b); end
    em = a.e1 ./ n; ev = max.((a.e2 .- n .* em .^ 2) ./ (n - 1), 0.0)
    seurat = top_genes(seurat_dispersion(em, ev), n_hvg)
    cm = a.c1 ./ n; cv = max.((a.c2 .- n .* cm .^ 2) ./ (n - 1), 0.0); ok = findall(>(0), cv)
    sdf = zeros(G); sdf[ok] = sqrt.(10 .^ loess_fit(log10.(cm[ok]), log10.(cv[ok]); span = 0.3))
    clip = cm .+ sdf .* sqrt(n); x = (x1 = zeros(G), x2 = zeros(G))
    for s in shards; sh, w = rd(s); v3_shard!(x, sh, w, train_well, tokrow, clip); end
    nv = fill(-Inf, G); nv[ok] = (n .* cm[ok] .^ 2 .+ x.x2[ok] .- 2 .* cm[ok] .* x.x1[ok]) ./ ((n - 1) .* sdf[ok] .^ 2)
    v3 = top_genes(nv, n_hvg)
    @printf("  hvg on %d train cells of %d shards: two passes in %.0f s\n", n, length(shards), time() - t0)
    return (seurat = seurat, seurat_v3 = v3, n = n)
end

# per well of the index: split side (1..5 = SPLIT_SIDES, 0 = a well outside the pseudobulk table) and the well's plate, drug, dose, group, moa, position, block
function well_table(meta::NamedTuple, sp::NamedTuple, wells::Vector{String})
    N = length(meta.well); ps = zeros(UInt8, N)
    for (k, s) in enumerate(SPLIT_SIDES); ps[getfield(sp, s)] .= k; end
    pb = Dict{String,Int}(); for i in N:-1:1; pb[meta.well[i]] = i; end
    col(v, z) = [haskey(pb, w) ? v[pb[w]] : z for w in wells]
    return col(ps, UInt8(0)), (plate = col(meta.plate, ""), drug = col(meta.drug, ""), dose = col(meta.dose, ""), group = col(meta.group, ""), moa = col(meta.moa, ""),
                               pos = col(meta.pos, -1), block = col(meta.block, -1))
end

# everything needed to read cells: the index, the pseudobulk table's split and well annotations, the train-cell gene statistics and panels, the class lists
struct CellSource
    raw_dir::String
    shards::Vector{String}
    off::Vector{Int}                       # cells of shard s = off[s]+1 : off[s+1] (global cell ids)
    well::Vector{UInt16}                   # per cell: code into wells
    line::Vector{UInt8}                    # per cell: code into lines
    wells::Vector{String}
    lines::Vector{String}
    side::Vector{UInt8}                    # per well: split side
    wmeta::NamedTuple                      # per well: plate, drug, dose, group, moa, pos, block
    plates::Vector{String}
    shard_plate::Vector{Int}               # per shard: the plate (code into plates) of most of its cells
    tokrow::Vector{Int32}
    genes::NamedTuple                      # symbol, ensembl, token of the 19020 gene rows (the pseudobulk rows)
    medians::Vector{Float32}               # all genes, train cells of the statistics shards
    gene_mean::Vector{Float32}
    gene_sd::Vector{Float32}
    panels::NamedTuple
    classes::Dict{String,Vector{String}}   # level => class names (pseudobulk classes first, in the pseudobulk order)
    in_pb::BitMatrix                       # wells x lines: the (well, cell line) pairs that are a pseudobulk of the table
    path::String
end

const CELL_SOURCES = Dict{Tuple{String,String,String},CellSource}()

# cell source of a config: sc_path (scripts/prepare_cells.jl), raw_dir (the shards) and data_path (the pseudobulk file: split, well annotations, classes); cached per process
function cell_source(cfg::AbstractDict)
    path = String(get(cfg, "sc_path", default_sc_path())); raw = String(cfg["raw_dir"]); pbp = String(cfg["data_path"])
    haskey(CELL_SOURCES, (path, raw, pbp)) && return CELL_SOURCES[(path, raw, pbp)]
    isfile(path) || error("no cell index $path: build it with scripts/prepare_cells.jl")
    isdir(shard_dir(raw)) || error("no shard directory $(shard_dir(raw)) (cfg raw_dir)")
    meta, sp, genes = jldopen(f -> (f["meta"], f["split"], f["genes"]), pbp, "r")
    f = load(path)
    f["tokens"] == genes.token || error("$path and $pbp have different gene rows")
    f["shards"] == list_shards(raw) || error("$path indexes other shard files than $(shard_dir(raw)) holds: rebuild it with scripts/prepare_cells.jl")
    side, wm = well_table(meta, sp, f["wells"])
    side == f["well_side"] || error("$path was built with another split than $pbp has: rebuild it with scripts/prepare_cells.jl")
    plates = sort(unique(filter(!isempty, wm.plate))); pid = Dict(p => i for (i, p) in enumerate(plates)); wp = [get(pid, p, 0) for p in wm.plate]
    off = f["off"]; well = f["well"]; cnt = zeros(Int, length(plates)); sp_plate = zeros(Int, length(off) - 1)
    for s in eachindex(sp_plate)
        fill!(cnt, 0)
        @inbounds for c in off[s]+1:off[s+1]; p = wp[well[c]]; p > 0 && (cnt[p] += 1); end
        sp_plate[s] = argmax(cnt)
    end
    lines = f["lines"]; c1 = class_names(meta, sp, "lvl1")
    classes = Dict("lvl1" => vcat(c1, sort(setdiff(lines, c1))), "lvl2" => class_names(meta, sp, "lvl2"))
    sd = copy(f["gene_sd"]); sd[sd .== 0] .= 1f0
    wid = Dict(w => i for (i, w) in enumerate(f["wells"])); lid = Dict(l => i for (i, l) in enumerate(lines)); in_pb = falses(length(wid), length(lid))
    for (w, l) in zip(meta.well, meta.cell_line); haskey(wid, w) && haskey(lid, l) && (in_pb[wid[w], lid[l]] = true); end
    src = CellSource(raw, f["shards"], off, well, f["line"], f["wells"], lines, side, wm, plates, sp_plate, token_rows(genes.token), genes, f["medians"], f["gene_mean"], sd,
                     f["panels"], classes, in_pb, path)
    @printf("cells: %d in %d shards (%s) | train %d val %d test %d ood %d reloc %d wells | classes lvl1 %d lvl2 %d\n", length(well), length(src.shards), shard_dir(raw),
            (count(==(UInt8(k)), side) for k in 1:5)..., length(classes["lvl1"]), length(classes["lvl2"])); flush(stdout)
    return CELL_SOURCES[(path, raw, pbp)] = src
end

# log1p CP10k of the given cells (global ids, any order, repeats allowed): the panel rows (P x n; none without a panel) and the top-K tokens (rule median: by x / train median, raw: by x), padded past the detected genes
function read_cells(src::CellSource, cells::AbstractVector{<:Integer}; panel::Union{Nothing,AbstractVector{<:Integer}} = nothing, K::Int = 0, rule::AbstractString = "median")
    n = length(cells); G = length(src.medians); check_topk_rule(rule)
    X = Matrix{Float32}(undef, isnothing(panel) ? 0 : length(panel), n)
    ids = Matrix{Int32}(undef, K, n); x = Matrix{Float32}(undef, K, n); len = zeros(Int32, n); ndet = zeros(Int32, n); umis = zeros(Float32, n)
    ppos = zeros(Int32, G); isnothing(panel) || (ppos[panel] .= 1:length(panel))
    key = rule == "median" ? src.medians : ones(Float32, G)
    o = sortperm(cells); b = CellBuf(G); j = 1
    while j <= n
        s = searchsortedlast(src.off, cells[o[j]] - 1); hi = src.off[s+1]; e = j
        while e < n && cells[o[e+1]] <= hi; e += 1; end
        locs = unique(Int.(cells[view(o, j:e)]) .- src.off[s])
        sh, rows = read_shard_cells(joinpath(shard_dir(src.raw_dir), src.shards[s]), locs, hi - src.off[s])
        fill_cells!(X, ids, x, len, ndet, umis, sh, view(o, j:e), cells, src.off[s], src.tokrow, ppos, key, b; rows)
        j = e + 1
    end
    try PyCall.pyimport("pyarrow").default_memory_pool().release_unused() catch end
    return (X = X, ids = ids, x = x, len = len, ndet = ndet, umis = umis)
end

# the fixed cell sample of a config: per plate, seeded shards are taken until every train / val / test / reloc well of the plate has cells (at least sc_shards of them);
# from their cells, up to sc_train_cells / sc_val_cells / sc_test_cells / sc_reloc_cells per split side, shared equally among the side's wells, in random order (ood wells: none);
# sc_test_per_pb > 0 (default 50) replaces the test side: every test pseudobulk with sc_test_per_pb of its cells (all its cells if fewer), from every shard
function sample_cells(src::CellSource, cfg::AbstractDict)
    rng = MersenneTwister(Int(get(cfg, "sc_sample_seed", 42))); per = Int(cfg["sc_shards"]); picks = Int[]
    wells(s) = Set(view(src.well, src.off[s]+1:src.off[s+1]))
    for p in eachindex(src.plates)
        ss = shuffle(rng, findall(==(p), src.shard_plate)); ws = map(wells, ss)
        need = Set(w for v in ws for w in v if src.side[w] in (1, 2, 3, 5))
        for (q, s) in enumerate(ss)
            q > per && isempty(need) && break
            (q <= per || !isdisjoint(ws[q], need)) || continue
            push!(picks, s); setdiff!(need, ws[q])
        end
    end
    by = [Dict{UInt16,Vector{Int}}() for _ in 1:5]
    for s in sort(picks), c in src.off[s]+1:src.off[s+1]
        w = src.well[c]; k = src.side[w]; k > 0 && push!(get!(by[k], w, Int[]), c)
    end
    cap = (Int(cfg["sc_train_cells"]), Int(cfg["sc_val_cells"]), Int(cfg["sc_test_cells"]), 0, Int(cfg["sc_reloc_cells"]))
    sel = Vector{Int}[]
    for (g, n) in zip(by, cap)
        q = isempty(g) ? 0 : cld(n, length(g)); v = Int[]
        for w in sort(collect(keys(g))); append!(v, shuffle(rng, g[w])[1:min(q, length(g[w]))]); end
        push!(sel, shuffle(rng, v)[1:min(n, length(v))])
    end
    npb = Int(get(cfg, "sc_test_per_pb", 50))
    if npb > 0
        sel[3] = test_per_pb(src, npb, MersenneTwister(Int(get(cfg, "sc_sample_seed", 42)) + 7))
        @printf("test sample: %d cells, %d of each test pseudobulk (all of them if fewer)\n", length(sel[3]), npb)
    end
    cells = reduce(vcat, sel); e = cumsum(length.(sel)); r(k) = collect(e[k]-length(sel[k])+1:e[k])
    @printf("cell sample: %d shards | wells with cells: train %d val %d test %d reloc %d\n", length(picks), (length(by[k]) for k in (1, 2, 3, 5))...)
    return cells, (train = r(1), val = r(2), test = r(3), ood = r(4), reloc = r(5))
end

# n cells of every test pseudobulk (a (well, cell line) of a test well that is a pseudobulk of the table), all of them if fewer; random order
function test_per_pb(src::CellSource, n::Int, rng::AbstractRNG)
    L = length(src.lines); key(c) = (Int(src.well[c]) - 1) * L + Int(src.line[c])
    ks = Int[]; cs = Int[]
    @inbounds for c in eachindex(src.well)
        w = src.well[c]; (src.side[w] == 3 && src.in_pb[w, src.line[c]]) || continue
        push!(ks, key(c)); push!(cs, c)
    end
    o = sortperm(ks); ks = ks[o]; cs = cs[o]; out = Int[]; i = 1
    while i <= length(ks)
        j = i; while j < length(ks) && ks[j+1] == ks[i]; j += 1; end
        g = view(cs, i:j); append!(out, length(g) <= n ? g : g[randperm(rng, length(g))[1:n]]); i = j + 1
    end
    return shuffle(rng, out)
end

# cells as TahoeData columns: panel values, top-k tokens (with their real lengths) and per-cell meta (the well's annotations, the global cell id, detected genes, coding UMIs)
function materialize(src::CellSource, cells::Vector{Int}, sp::NamedTuple, panel::AbstractString, rule::AbstractString, K::Int)
    haskey(src.panels, Symbol(panel)) || error("panel must be one of $(join(keys(src.panels), ", ")) for dataset = sc, got $panel")
    p = src.panels[Symbol(panel)]; t0 = time()
    r = read_cells(src, cells; panel = p, K, rule)
    w = src.well[cells]; wm = src.wmeta
    meta = (cell_line = src.lines[src.line[cells]], well = src.wells[w], plate = wm.plate[w], drug = wm.drug[w], dose = wm.dose[w], group = wm.group[w], moa = wm.moa[w],
            pos = wm.pos[w], block = wm.block[w], cell = cells, in_pb = Bool[src.in_pb[src.well[c], src.line[c]] for c in cells], n_det = r.ndet, umis = r.umis)
    full = (symbol = src.genes.symbol, medians = src.medians, gene_mean = src.gene_mean, gene_sd = src.gene_sd)
    @printf("cell sample: train %d val %d test %d reloc %d cells | panel %s, top-%d (%s) | detected genes median %d, cells under %d: %.1f %% | %.0f s\n", length(sp.train), length(sp.val),
            length(sp.test), length(sp.reloc), panel, K, rule, median(r.ndet), K, 100 * mean(r.ndet .< K), time() - t0); flush(stdout)
    return TahoeData(r.X, meta, map(v -> v[p], src.genes), src.medians[p], src.gene_mean[p], src.gene_sd[p], sp, p, p, String(panel), src.panels, full,
                     (ids = r.ids, x = r.x, rule = String(rule), len = r.len), src.path, src)
end

# data of a single-cell config: the cell sample with its panel and top-k tokens (given data reused when its panel, topk_rule and top_k fit, else its cells are read again)
function cell_data(cfg::AbstractDict, data = nothing; genes::AbstractString = "panel", topk::Bool = true)
    genes == "all" && error("genes = all is not built for dataset = sc (use panel or topk)")
    panel = String(get(cfg, "panel", "hvg")); rule = topk_rule(cfg); K = Int(get(cfg, "top_k", 1024))
    if !isnothing(data)
        is_sc(data) || error("config dataset = sc but the data given holds pseudobulks")
        data.panel_name == panel && loaded_topk_rule(data) == rule && size(data.topk.ids, 1) >= K && return data
        return materialize(data.cells, data.meta.cell, data.split, panel, rule, K)
    end
    src = cell_source(cfg); cells, sp = sample_cells(src, cfg)
    return materialize(src, cells, sp, panel, rule, K)
end

# pseudobulk of each column: an id per (well, cell line) among cols, 0 for cells whose (well, cell line) is no pseudobulk of the table
function pb_groups(d::TahoeData, cols::AbstractVector{<:Integer})
    id = Dict{Tuple{String,String},Int}()
    return [d.meta.in_pb[i] ? get!(id, (d.meta.well[i], d.meta.cell_line[i]), length(id) + 1) : 0 for i in cols]
end

# training batches drawn from every train cell: pools of sc_pool_shards shards per plate, sc_pool_cells cells per pool (uniform, or class-balanced: uniform class of the pool, then a uniform cell of it)
mutable struct CellStream
    src::CellSource
    panel::Union{Nothing,Vector{Int}}
    K::Int
    rule::String
    shuf::Bool
    cls::Union{Nothing,Vector{Int32}}      # class of a well (lvl2) or a cell line (lvl1), 0 = none; nothing = unlabelled (pretraining)
    by_line::Bool
    balanced::Bool
    rng::MersenneTwister
    order::Vector{Vector{Int}}             # per plate: its shards, shuffled
    at::Vector{Int}
    per_plate::Int
    pool_cells::Int
    X::Any                                 # raw input of the current pool
    y::Vector{Int}
    cells::Vector{Int}                     # its cells (global ids)
    pos::Int
    pools::Int
    load_s::Float64
end

# stream of a single-cell config (nothing for pseudobulks or sc_stream = false: training then uses the in-memory train cells); level = nothing for unlabelled batches
function cell_stream(d::TahoeData, cfg::AbstractDict; level::Union{Nothing,AbstractString} = nothing, seed::Integer = 1, tokens::AbstractString = token_set(cfg),
                     balanced::Bool = false)
    (is_sc(d) && get(cfg, "sc_stream", true) == true) || return nothing
    src = d.cells; cls = nothing
    if !isnothing(level)
        id = Dict(n => Int32(i) for (i, n) in enumerate(src.classes[level]))
        cls = level == "lvl1" ? Int32[get(id, l, Int32(0)) for l in src.lines] : Int32[get(id, dr, Int32(0)) for dr in src.wmeta.drug]
    end
    rng = MersenneTwister(seed); order = [shuffle(rng, findall(==(p), src.shard_plate)) for p in eachindex(src.plates)]
    return CellStream(src, tokens == "topk" ? nothing : d.panel, tokens == "topk" ? Int(get(cfg, "top_k", 1024)) : 0, topk_rule(cfg), get(cfg, "rank_shuffle", false) == true,
                      cls, level == "lvl1", balanced, rng, order, ones(Int, length(order)), Int(cfg["sc_pool_shards"]), Int(cfg["sc_pool_cells"]), nothing, Int[], Int[], 1, 0, 0.0)
end

function next_pool!(st::CellStream)
    t0 = time(); src = st.src; cand = Int[]; lab = Int32[]; ns = 0
    for p in eachindex(st.order), _ in 1:min(st.per_plate, length(st.order[p]))
        st.at[p] > length(st.order[p]) && (shuffle!(st.rng, st.order[p]); st.at[p] = 1)
        s = st.order[p][st.at[p]]; st.at[p] += 1; ns += 1
        for c in src.off[s]+1:src.off[s+1]
            w = src.well[c]; src.side[w] == 1 || continue
            if isnothing(st.cls)
                push!(cand, c)
            else
                l = st.by_line ? st.cls[src.line[c]] : st.cls[w]
                l > 0 && (push!(cand, c); push!(lab, l))
            end
        end
    end
    isempty(cand) && error("cell stream: no training cells in a pool of $ns shards")
    if st.balanced
        byc = Dict{Int32,Vector{Int}}(); for (i, l) in enumerate(lab); push!(get!(byc, l, Int[]), i); end
        cs = sort(collect(keys(byc))); pick = [rand(st.rng, byc[rand(st.rng, cs)]) for _ in 1:st.pool_cells]
    else
        pick = randperm(st.rng, length(cand))[1:min(st.pool_cells, length(cand))]
    end
    cells = cand[pick]
    st.X = nothing; GC.gc()   # free the last pool and the shard buffers before reading the next (peak RSS reached 64 GB without)
    r = read_cells(src, cells; panel = st.panel, K = st.K, rule = st.rule)
    st.K > 0 && st.shuf && shuffle_tokens!(r.ids, r.x, r.len, cells)
    st.X = st.K > 0 ? TopkInput(r.ids, r.x, r.len) : r.X
    st.y = isnothing(st.cls) ? Int[] : Int.(lab[pick]); st.cells = cells; st.pos = 1; st.pools += 1; st.load_s += time() - t0
    (st.pools == 1 || st.pools % 10 == 0) && (@printf("cell stream pool %d: %d cells of %d train cells in %d shards%s, %.0f s (%.0f s in all)\n", st.pools, length(cells), length(cand), ns,
                                                      isnothing(st.cls) ? "" : ", $(length(unique(st.y))) classes", time() - t0, st.load_s); flush(stdout))
    return st
end

# next bs cells of the stream: raw input (panel matrix or top-k tokens) and class ids (empty when unlabelled)
function next_batch!(st::CellStream, bs::Int)
    (isnothing(st.X) || st.pos + bs - 1 > size(st.X, 2)) && next_pool!(st)
    size(st.X, 2) >= bs || error("cell stream: a pool of $(size(st.X, 2)) cells is smaller than the batch ($bs)")
    c = st.pos:st.pos+bs-1; st.pos += bs
    return st.X[:, c], (isempty(st.y) ? st.y : st.y[c])
end
