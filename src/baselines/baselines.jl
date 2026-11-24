# no-pretraining baselines: e* = expression, r* = rank features; log = logistic regression, mlp = relu MLP, lda = shrinkage LDA

include("logistic.jl")
include("mlp.jl")
include("trivial.jl")

_flag(x) = x isa Bool ? x : x != 0

baseline_input_name(model::AbstractString) = model[1] == 'e' ? "etf" : model[1] == 'r' ? "rtf" : error("unknown baseline $model")

# gene set of a baseline: genes = panel | all | topk; default the primary set of the matching transformer input (expression panel, rank topk)
baseline_genes(cfg::Dict) = get(cfg, "genes", primary_tokens(baseline_input_name(cfg["model"])))

# name suffix of the position control: _posproj (project, the default) or _pos (subtract, the original rows)
pos_suffix(cfg::Dict) = (pm = pos_method(get(cfg, "pos_correct", false)); isnothing(pm) ? "" : pm == "subtract" ? "_pos" : "_posproj")

# name: model, _z, then the gene set (topkraw for topk_rule = raw) / rank control / position correction when not the primary setting, _wh when whitened, then the level
function baseline_name(cfg::Dict)
    m = cfg["model"]; g = baseline_genes(cfg); rf = get(cfg, "rank_feat", "graded"); raw = g == "topk" && topk_rule(cfg) == "raw"
    s = (g == primary_tokens(baseline_input_name(m)) && !raw ? "" : "_$g" * (raw ? "raw" : "")) * (rf == "graded" ? "" : "_$rf") * pos_suffix(cfg) * (haskey(cfg, "whiten") ? "_wh" : "")
    return "$(m)$(m[1] == 'e' && _flag(get(cfg, "standardize", false)) ? "_z" : "")$(s)_$(cfg["level"])"
end

# data a baseline needs: all genes for genes = all, the top-k tokens for genes = topk, else the panel
baseline_data(cfg::Dict, data) = (g = baseline_genes(cfg); cfg_data(cfg, data; genes = g == "all" ? "all" : "panel", topk = g == "topk"))

# train / val / test / reloc inputs: rank features (panel, all genes or top-k; graded, set or shuffle), or expression z-scored per gene with the stored train mean/sd (as ETF A) and clipped to +-clip (0 = no clip);
# expression on top-k genes: each sample's top-k values at their gene rows (0 elsewhere, 19020 dims), z-scored per feature with its train-column mean / sd; pos_correct removes the plate-position effect (src/eval/position.jl)
function baseline_inputs(d::TahoeData, idx, sp, model::AbstractString, cfg::Dict)
    genes = baseline_genes(cfg); rf = get(cfg, "rank_feat", "graded")
    genes in ("panel", "all", "topk") || error("genes must be panel, all or topk, got $genes")
    (genes == "all") == is_all(d) || error("baseline genes = $genes but the data holds $(is_all(d) ? "all genes" : "a panel")")
    model[1] == 'e' && rf != "graded" && error("expression baselines take no rank_feat")
    is_sc(d) && return cell_baseline_inputs(d, idx, sp, model, cfg)
    aux = (;)
    if model[1] == 'r' && genes == "topk"
        K = Int(get(cfg, "top_k", 1024))
        X = topk_features(d.topk.ids[1:K, idx], length(d.full.medians), idx; feat = rf, len = topk_len(d, idx, K)); info = "top-$K rank features ($rf, topk_rule $(loaded_topk_rule(d)))"
    elseif genes == "topk"
        K = Int(get(cfg, "top_k", 1024)); clip = Float32(get(cfg, "clip", 10))
        X = topk_value_features(d.topk.ids[1:K, idx], d.topk.x[1:K, idx], length(d.full.medians); len = topk_len(d, idx, K))
        info = "top-$K values (topk_rule $(loaded_topk_rule(d)))"
        if _flag(get(cfg, "standardize", false))
            Xt = X[:, sp.train]; mu = vec(mean(Xt, dims=2)); sd = vec(std(Xt, dims=2)); sd[sd .== 0] .= 1f0; Xt = nothing; aux = (mu = mu, sd = sd)
            X = (X .- mu) ./ sd; frac = mean(abs.(X[:, sp.train]) .> clip)
            clip > 0 && (X = clamp.(X, -clip, clip))
            info *= @sprintf(", z-scored per feature on train, clip %g (%.4f%% of train values)", clip, 100 * (clip > 0 ? frac : 0.0))
        end
    elseif model[1] == 'r'
        X = d.X[:, idx]
        X = rank_control(rank_features(gene_ranks(X, d.medians), X), X, idx, rf); info = "rank features ($genes, $rf)"
    elseif _flag(get(cfg, "standardize", false))
        X = d.X[:, idx]
        clip = Float32(get(cfg, "clip", 10))
        X = (X .- d.gene_mean) ./ d.gene_sd
        frac = mean(abs.(X[:, sp.train]) .> clip)
        clip > 0 && (X = clamp.(X, -clip, clip))
        info = @sprintf("z-scored (%s), clip %g (%.4f%% of train values)", genes, clip, 100 * (clip > 0 ? frac : 0.0))
    else
        X = d.X[:, idx]; info = "raw expression ($genes)"
    end
    pm = pos_method(get(cfg, "pos_correct", false))
    isnothing(pm) || (X = pos_correct(X, d.meta, idx, sp.train; method = pm); info *= ", position-corrected ($pm)")
    return X[:, sp.train], X[:, sp.val], X[:, sp.test], X[:, sp.reloc], info, aux
end

# the same features for raw stream batches (single cells): rank / value features of top-k tokens, rank features or z-scores of the panel; aux = baseline_inputs' train mean / sd of the standardised top-k values
function stream_features(d::TahoeData, model::AbstractString, cfg::Dict, raw, aux::NamedTuple)
    genes = baseline_genes(cfg); rf = get(cfg, "rank_feat", "graded"); n = length(d.full.medians); z = _flag(get(cfg, "standardize", false)); clip = Float32(get(cfg, "clip", 10))
    rf == "shuffle" && error("rank_feat = shuffle is not built for the cell stream (sc_stream = false trains on the in-memory cells)")
    isnothing(pos_method(get(cfg, "pos_correct", false))) || error("pos_correct is not built for the cell stream (sc_stream = false trains on the in-memory cells)")
    cl(X) = clip > 0 ? clamp.(X, -clip, clip) : X
    model[1] == 'r' && genes == "topk" && return topk_features(raw.ids, n, 1:size(raw, 2); feat = rf, len = raw.len)
    if genes == "topk"
        X = topk_value_features(raw.ids, raw.v, n; len = raw.len)
        return z ? cl((X .- aux.mu) ./ aux.sd) : X
    end
    model[1] == 'r' && return rank_control(rank_features(gene_ranks(raw, d.medians), raw), raw, 1:size(raw, 2), rf)
    return z ? cl((raw .- d.gene_mean) ./ d.gene_sd) : raw
end

# single cells: the same features built split by split (one dense matrix per split, no copy of all cells); top-k values are standardised with the train split's mean / sd
function cell_baseline_inputs(d::TahoeData, idx, sp, model::AbstractString, cfg::Dict)
    c = merge(cfg, Dict("tokens" => baseline_genes(cfg) == "topk" ? "topk" : "panel", "input" => baseline_input_name(model), "rank_shuffle" => false))
    raw(v) = raw_input(d, idx[v], c); aux = (;)
    if baseline_genes(cfg) == "topk" && model[1] == 'e' && _flag(get(cfg, "standardize", false))
        Xt = stream_features(d, model, merge(cfg, Dict("standardize" => false)), raw(sp.train), aux)
        mu = vec(mean(Xt, dims=2)); sd = vec(std(Xt, dims=2)); sd[sd .== 0] .= 1f0; aux = (mu = mu, sd = sd); Xt = nothing
    end
    f(v) = stream_features(d, model, cfg, raw(v), aux)
    return f(sp.train), f(sp.val), f(sp.test), f(sp.reloc), "$(baseline_genes(cfg)) features of $(model[1] == 'r' ? "ranks" : "expression"), built per split", aux
end

# train one logistic / MLP baseline with the finetune() loop conventions (train/classify.jl)
function baseline(cfg::Dict; data::Union{Nothing,TahoeData} = nothing)
    set_math!(cfg)
    model_name = cfg["model"]; level = cfg["level"]
    model_name in ("elda", "rlda") && return lda_baseline(cfg; data)
    model_name in ("elog", "rlog", "emlp", "rmlp") || error("model must be elog, rlog, emlp, rmlp, elda or rlda, got $model_name")
    is_log = endswith(model_name, "log")
    seed = cfg["seed"]; bs = cfg["batch_size"]; total = cfg["max_steps"]
    lr = Float64(is_log ? get(cfg, "lr", 1e-3) : cfg["lr"])
    drop = Float64(get(cfg, "drop_prob", 0.0)); wd = Float64(get(cfg, "weight_decay", 0.0))
    cfg = merge(cfg, Dict("lr" => lr, "job_type" => get(cfg, "job_type", "baseline")))
    d = baseline_data(cfg, data)
    idx, y, names = labels(d, level; reloc = true)
    sp = finetune_split(d, idx)
    C = length(names)
    Xtr, Xva, Xte, Xrl, info, aux = baseline_inputs(d, idx, sp, model_name, cfg)
    if haskey(cfg, "whiten")
        is_sc(d) && error("whiten is pseudobulk only")
        (Xtr, Xva, Xte, Xrl), wi = lda_whiten(Xtr, y[sp.train], C, Float64(cfg["whiten"]), (Xtr, Xva, Xte, Xrl)); info *= wi
    end
    G = size(Xtr, 1)
    ytr, yva, yte = y[sp.train], y[sp.val], y[sp.test]

    # model
    seed_all!(seed)
    if is_log
        model = logistic_model(G, C); arch = "linear $G => $C"
    else
        sizes = mlp_sizes(G, C, cfg["n_layers"], get(cfg, "mlp_hidden_dim", 0); shape=get(cfg, "mlp_shape", "const"))
        model = mlp_model(sizes, drop); arch = "MLP $(join(sizes, "-"))"
    end
    model = to_device(model)
    opt = Flux.setup(Optimisers.AdamW(lr, (0.9, 0.999), wd), model)

    name = get(cfg, "run_name", baseline_name(cfg))
    dir = run_dir(cfg, "baseline", name)
    cfg = merge(cfg, Dict("wandb_name" => "$(name)_s$(seed)"))
    @printf("baseline %s | %d classes | train %d val %d test %d reloc %d | %s | %s | lr %.3g | dir %s\n",
            name, C, length(ytr), length(yva), length(yte), length(sp.reloc), arch, info, lr, dir)

    grad(m, x, t) = (r = Flux.withgradient(m -> Flux.logitcrossentropy(m(x), t), m); (r.val, r.grad[1]))
    # single cells: train batches from every train cell of the level's classes, as the transformer finetunes
    stream = cell_stream(d, cfg; level, seed = seed + 3000, tokens = baseline_genes(cfg) == "topk" ? "topk" : "panel", balanced = get(cfg, "balanced", level == "lvl2") == true)
    feed = isnothing(stream) ? nothing : bs -> ((raw, yb) = next_batch!(stream, bs); (stream_features(d, model_name, cfg, raw, aux), yb))
    res = train_classifier!(model, opt, grad, (Xtr, ytr, Xva, yva, Xte, yte), cfg, C; extra = (reloc = (Xrl, y[sp.reloc]),), feed, groups = is_sc(d) ? pb_groups(d, idx[sp.test]) : nothing)
    return save_classifier_run(dir, cfg, res, d, idx, sp, y, names,
                               (name = name, input = baseline_input_name(model_name), tokens = token_label(baseline_genes(cfg), cfg), mode = "baseline", objective = model_name), NaN)
end
