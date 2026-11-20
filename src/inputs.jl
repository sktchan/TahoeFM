# inputs: ETF = expression values, RTF = rank of each gene by x / gene median (rank 1 = highest); token sets: panel (all panel genes in gene order) or topk (each sample's top-k genes in rank order)

abstract type Input end
struct ETF <: Input end
struct RTF <: Input end

parse_input(s::AbstractString) = s == "etf" ? ETF() : s == "rtf" ? RTF() : error("input must be etf or rtf, got $s")
input_name(::ETF) = "etf"
input_name(::RTF) = "rtf"

# rank of gene g in sample j (gene order); ties keep gene order, as sortperm in the original rank_genes
function gene_ranks(X::AbstractMatrix{<:Real}, medians::AbstractVector{<:Real})
    G, N = size(X)
    R = Matrix{Int32}(undef, G, N)
    v = Vector{Float32}(undef, G); p = Vector{Int}(undef, G)
    @inbounds for j in 1:N
        @. v = X[:, j] / medians
        sortperm!(p, v, rev=true)
        for r in 1:G
            R[p[r], j] = Int32(r)
        end
    end
    return R
end

# model input for one set of columns
encode_input(::ETF, X, medians) = Matrix{Float32}(X)
encode_input(::RTF, X, medians) = gene_ranks(X, medians)

# model input of raw columns X: ETF scaled by cfg etf_scale, RTF gene ranks
model_input(input::Input, X, d, cfg::AbstractDict) = input isa RTF ? gene_ranks(X, d.medians) : etf_scale(X, d, get(cfg, "etf_scale", "raw"))

# rank baseline features as the original rmlp/rlog: (n_det + 1 - r) / G for detected genes (x > 0), 0 for undetected
function rank_features(R::AbstractMatrix{<:Integer}, X::AbstractMatrix{<:Real})
    G = size(R, 1); nd = vec(sum(X .> 0, dims=1))
    return Float32.(ifelse.((R .<= nd') .& (X .> 0), (nd' .+ 1 .- R) ./ G, 0))
end

# topk tokens of a set of samples: gene ids (K x B, rows of the all-gene table, rank order) and the value channel input (raw values, ETF values or RTF rank positions)
# len = real tokens per sample when the tail of a column is padding (single cells with fewer than K detected genes; pad slots hold gene 1, value 0), nothing = no padding
struct TopkInput{I<:AbstractMatrix{<:Integer},V<:AbstractMatrix,L}
    ids::I
    v::V
    len::L
end
TopkInput(ids, v) = TopkInput(ids, v, nothing)
Base.size(t::TopkInput, i...) = size(t.ids, i...)
Base.getindex(t::TopkInput, ::Colon, c) = TopkInput(t.ids[:, c], t.v[:, c], isnothing(t.len) ? nothing : t.len[c])
dev(t::TopkInput) = TopkInput(dev(t.ids), dev(t.v), t.len)

# padding helpers: real tokens per sample (nothing = every slot is real), the real-slot matrix (K x B Bool), the values of the real tokens
pad_len(x) = nothing
pad_len(x::TopkInput) = isnothing(x.len) || all(==(size(x, 1)), x.len) ? nothing : x.len
pad_mask(x) = (l = pad_len(x); isnothing(l) ? nothing : (1:size(x, 1)) .<= permutedims(l))
real_values(x) = (m = pad_mask(x); isnothing(m) ? token_values(x) : Array(token_values(x))[m])
_like(x::TopkInput, a) = x.ids isa CuArray ? cu(a) : a
# attention mask (K x 1 x 1 x B Bool, true = a real key) and mean-pool weights (1 x K x B, 1 / len on real tokens) on the input's device; nothing without padding
attn_mask(x) = nothing
attn_mask(x::TopkInput) = Flux.ChainRulesCore.ignore_derivatives(() -> (m = pad_mask(x); isnothing(m) ? nothing : _like(x, reshape(Array(m), size(m, 1), 1, 1, size(m, 2)))))
pool_weights(x) = nothing
pool_weights(x::TopkInput) = Flux.ChainRulesCore.ignore_derivatives(() -> (m = pad_mask(x); isnothing(m) ? nothing : _like(x, reshape(Float32.(m) ./ Float32.(permutedims(x.len)), 1, size(m)...))))

const RANK_SHUFFLE_SEED = 100_000   # per-sample permutations of the rank controls: Xoshiro(RANK_SHUFFLE_SEED + sample index)

# primary token set of an input: etf -> panel (hvg), rtf -> topk (Geneformer); the other two cells are the later 2x2 controls
primary_tokens(input::AbstractString) = input == "rtf" ? "topk" : "panel"

# token set of a transformer config: tokens = panel | topk; default the input's primary set
function token_set(cfg::AbstractDict)
    t = get(cfg, "tokens", primary_tokens(get(cfg, "input", "etf")))
    t in ("panel", "topk") || error("tokens must be panel or topk, got $t")
    return t
end

# top-k selection rule of a config: median (x / train median, Geneformer; default) or raw (raw x); only for top-k tokens / features
topk_rule(cfg::AbstractDict) = check_topk_rule(get(cfg, "topk_rule", "median"))

# token-set label of result rows: topkraw for top-k tokens / features with topk_rule = raw
token_label(t::AbstractString, cfg::AbstractDict) = t == "topk" && topk_rule(cfg) == "raw" ? "topkraw" : String(t)

# arm name: input, plus the token set when it is not the input's primary one (topkraw for topk_rule = raw), plus _shuf for the rank-shuffle control
function arm_name(cfg::AbstractDict)
    i = cfg["input"]; t = token_set(cfg); raw = t == "topk" && topk_rule(cfg) == "raw"
    return i * (t == primary_tokens(i) && !raw ? "" : "_" * t * (raw ? "raw" : "")) * (get(cfg, "rank_shuffle", false) == true ? "_shuf" : "")
end

# gene table rows and value table rows (RTF) of the encoder: panel G and G; topk all genes and K
token_dims(d, cfg::AbstractDict) = token_set(cfg) == "topk" ? (length(d.full.medians), Int(get(cfg, "top_k", 1024))) : (n_genes(d), n_genes(d))

# data for a config: given data reused (switched to the config's panel, top-k tokens of the config's topk_rule added when needed), else loaded; genes = all for all-gene baselines
function cfg_data(cfg::AbstractDict, data = nothing; genes::AbstractString = "panel", topk::Bool = token_set(cfg) == "topk")
    ds = get(cfg, "dataset", "pb")
    ds == "sc" && return cell_data(cfg, data; genes, topk)
    ds == "pb" || error("dataset must be pb or sc, got $ds")
    isnothing(data) || !is_sc(data) || error("config dataset = pb but the data given holds single cells")
    panel = get(cfg, "panel", "hvg"); rule = topk_rule(cfg)
    topk || rule == "median" || error("topk_rule = $rule applies to top-k tokens / features only")
    isnothing(data) && return load_data(cfg["data_path"]; genes, panel, topk, topk_rule = rule)
    d = genes == "all" ? (is_all(data) ? data : load_data(data.path; genes = "all", panel, topk = !isempty(data.topk.ids), topk_rule = loaded_topk_rule(data))) : panel_data(data; panel)
    tk = topk && (isempty(d.topk.ids) || loaded_topk_rule(d) != rule) ? (ids = topk_ids(d, rule), x = topk_values(d, rule), rule = rule) : d.topk
    # a reload keeps the given data's split (callers may pass a subset split)
    (tk !== d.topk || d.split !== data.split) && (d = TahoeData(d.X, d.meta, d.genes, d.medians, d.gene_mean, d.gene_sd, data.split, d.gene_idx, d.panel,
                                                              d.panel_name, d.panels, d.full, tk, d.path))
    return d
end

# raw input of samples cols: panel genes x samples (log1p values), or their top-k tokens (ids, values); rank_shuffle permutes each sample's tokens once (seeded by the sample index)
function raw_input(d, cols, cfg::AbstractDict)
    shuf = get(cfg, "rank_shuffle", false) == true
    token_set(cfg) == "panel" && (shuf && error("rank_shuffle needs tokens = topk"); return d.X[:, cols])
    shuf && cfg["input"] != "rtf" && error("rank_shuffle is a control for input = rtf")
    K = Int(get(cfg, "top_k", 1024)); isempty(d.topk.ids) && error("top-k tokens not loaded (load_data(...; topk = true) or cfg_data)")
    loaded_topk_rule(d) == topk_rule(cfg) || error("top-k tokens loaded with topk_rule = $(loaded_topk_rule(d)), config asks $(topk_rule(cfg))")
    size(d.topk.ids, 1) >= K || error("top_k $K > stored top-k $(size(d.topk.ids, 1))")
    I = d.topk.ids[1:K, cols]; V = d.topk.x[1:K, cols]; len = topk_len(d, cols, K)
    shuf && shuffle_tokens!(I, V, len, is_sc(d) ? d.meta.cell[cols] : cols)
    return TopkInput(I, V, len)
end

# real top-k tokens of samples cols at top_k = K (single cells with fewer detected genes are padded); nothing = no padding stored
topk_len(d, cols, K::Integer) = (l = get(d.topk, :len, nothing); isnothing(l) ? nothing : min.(l[cols], Int32(K)))

# rank-shuffle control: each sample's real tokens permuted once, seeded by the sample id
function shuffle_tokens!(I::AbstractMatrix, V::AbstractMatrix, len, ids)
    for (j, s) in enumerate(ids)
        L = isnothing(len) ? size(I, 1) : Int(len[j]); p = randperm(Xoshiro(RANK_SHUFFLE_SEED + s), L); I[1:L, j] = I[p, j]; V[1:L, j] = V[p, j]
    end
    return I, V
end

# topk model input: ETF values (raw, or z of each token's own gene), RTF the rank positions 1..K
model_input(::ETF, X::TopkInput, d, cfg::AbstractDict) = TopkInput(X.ids, topk_scale(X, d, get(cfg, "etf_scale", "raw")), X.len)
model_input(::RTF, X::TopkInput, d, cfg::AbstractDict) = TopkInput(X.ids, repeat(Int32.(1:size(X, 1)), 1, size(X, 2)), X.len)

function topk_scale(X::TopkInput, d, how::AbstractString)
    how == "raw" && return Matrix{Float32}(X.v)
    how == "gene" || error("etf_scale for topk tokens must be raw or gene, got $how")
    return clamp.((X.v .- d.full.gene_mean[X.ids]) ./ d.full.gene_sd[X.ids], -10f0, 10f0)
end

# token helpers: number of tokens, the values fed to the ETF value channel, each token's rank (topk: its slot) and gene (panel: nothing = token t is gene t)
n_tokens(X) = size(X, 1)
token_values(X::AbstractMatrix) = X
token_values(X::TopkInput) = X.v
token_ranks(X::AbstractMatrix, d) = gene_ranks(X, d.medians)
token_ranks(X::TopkInput, d) = repeat(Int32.(1:size(X, 1)), 1, size(X, 2))
token_genes(X::AbstractMatrix) = nothing
token_genes(X::TopkInput) = X.ids

# topk rank baseline features (n x B): (K + 1 - r) / K for the sample's gene at rank r, 0 elsewhere; set = 1 for every top-k gene; shuffle = the graded values permuted inside the set
function topk_features(ids::AbstractMatrix{<:Integer}, n::Integer, cols::AbstractVector{<:Integer}; feat::AbstractString = "graded", len = nothing)
    feat in ("graded", "set", "shuffle") || error("rank_feat must be graded, set or shuffle, got $feat")
    K, B = size(ids); F = zeros(Float32, n, B)
    for j in 1:B
        L = isnothing(len) ? K : Int(len[j])   # padded samples: only their real tokens
        p = feat == "shuffle" ? randperm(Xoshiro(RANK_SHUFFLE_SEED + cols[j]), L) : (1:L)
        for r in 1:L
            F[ids[p[r], j], j] = feat == "set" ? 1f0 : Float32((K + 1 - r) / K)
        end
    end
    return F
end

# topk expression baseline features (n x B): each top-k token's value at its gene row, 0 elsewhere
function topk_value_features(ids::AbstractMatrix{<:Integer}, x::AbstractMatrix{<:Real}, n::Integer; len = nothing)
    F = zeros(Float32, n, size(ids, 2))
    for j in axes(ids, 2), r in 1:(isnothing(len) ? size(ids, 1) : Int(len[j])); F[ids[r, j], j] = x[r, j]; end
    return F
end

# panel / all-gene rank controls on rank features F of raw X: graded = F, set = detected-gene indicator, shuffle = F permuted among each sample's detected genes
function rank_control(F::AbstractMatrix, X::AbstractMatrix, cols::AbstractVector{<:Integer}, feat::AbstractString)
    feat == "graded" && return F
    feat == "set" && return Float32.(X .> 0)
    feat == "shuffle" || error("rank_feat must be graded, set or shuffle, got $feat")
    S = copy(F)
    for j in axes(F, 2)
        k = findall(>(0), view(X, :, j)); p = randperm(Xoshiro(RANK_SHUFFLE_SEED + cols[j]), length(k))
        S[k, j] = F[k[p], j]
    end
    return S
end
