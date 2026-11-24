# end-to-end finetuning on lvl1 (cell line) or lvl2 (drug); mode scratch (T0), random (T1) or pretrained (T2); test, reloc and per-dose accuracy; saves the encoders as checkpoints

function finetune(cfg::Dict; data::Union{Nothing,TahoeData} = nothing)
    set_math!(cfg)
    seed = cfg["seed"]; bs = cfg["batch_size"]; total = cfg["max_steps"]; lr = Float64(cfg["lr"])
    mode = cfg["mode"]; input = parse_input(cfg["input"]); level = cfg["level"]
    drop = Float64(cfg["drop_prob"]); micro = get(cfg, "micro_bs", 0)
    # a pretrained / random encoder keeps its pretraining token set, norm and ETF input scaling
    if mode in ("pretrained", "random")
        pcfg = ckpt_config(cfg["ckpt"])
        haskey(cfg, "tokens") && cfg["tokens"] != token_set(pcfg) && error("checkpoint tokens $(token_set(pcfg)) != finetune tokens $(cfg["tokens"])")
        haskey(cfg, "topk_rule") && topk_rule(cfg) != topk_rule(pcfg) && error("checkpoint topk_rule $(topk_rule(pcfg)) != finetune topk_rule $(cfg["topk_rule"])")
        # top-k tokens index the same gene rows in both datasets; panel rows do not (each dataset has its own panel genes)
        get(pcfg, "dataset", "pb") == get(cfg, "dataset", "pb") || token_set(pcfg) == "topk" || error("checkpoint dataset $(get(pcfg, "dataset", "pb")) != finetune dataset $(cfg["dataset"]): panel tokens hold different genes")
        cfg = merge(cfg, Dict("norm" => get(pcfg, "norm", "pre"), "etf_scale" => get(pcfg, "etf_scale", "raw")), delete!(ckpt_input_cfg(pcfg), "input"))
    end
    cfg = merge(cfg, Dict("tokens" => token_set(cfg)))
    d = cfg_data(cfg, data)
    idx, y, names = labels(d, level; reloc = true)   # train / val / test plus the reloc wells of the same classes
    sp = finetune_split(d, idx)
    nid, G = token_dims(d, cfg); C = length(names)

    # model
    obj_name = "none"; pex = Dict{String,Any}("init_scheme" => get(cfg, "init", "default"), "x_mean" => 0.0, "x_sd" => 1.0)
    if mode == "scratch"
        seed_all!(seed)
        enc = Encoder(input; n_genes=nid, n_ranks=G, embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"], n_heads=cfg["n_heads"],
                      n_layers=cfg["n_layers"], drop_prob=drop, norm=cfg["norm"])
    elseif mode in ("pretrained", "random")
        enc, pcfg, pex = load_encoder(cfg["ckpt"]; random=(mode == "random"), init_seed=get(cfg, "init_seed", 0))
        pcfg["input"] == cfg["input"] || error("checkpoint input $(pcfg["input"]) != finetune input $(cfg["input"])")
        enc = set_dropout(enc, drop); obj_name = pcfg["objective"]
        seed_all!(seed)
    else
        error("mode must be scratch, pretrained or random")
    end

    Xtr, Xva, Xte, Xrl = finetune_inputs(cfg, d, idx, sp)
    # scratch encoder init scheme (default, or tok_norm as in erecon / lrecon pretraining, with the train input's global mean / sd)
    if mode == "scratch" && get(cfg, "init", "default") != "default"
        pex["x_mean"], pex["x_sd"] = Float64(mean(real_values(Xtr))), Float64(std(real_values(Xtr)))
        init_encoder!(enc, Symbol(cfg["init"]); seed, x_mean=pex["x_mean"], x_sd=pex["x_sd"])
    end
    ytr, yva, yte = y[sp.train], y[sp.val], y[sp.test]
    model = Classifier(enc, C; hidden_dim=cfg["hidden_dim"], drop_prob=drop)
    model = to_device(model)
    hsd = get(cfg, "head_init", false) ? head_init!(model, batches(dev, Xtr[:, 1:min(5000, size(Xtr, 2))], bs)) : NaN

    opt = Flux.setup(Optimisers.AdamW(lr, (0.9, 0.999), Float64(get(cfg, "weight_decay", 0.0))), model)
    # lr x s (warmup)
    lr_scale!(s) = Optimisers.adjust!(opt, lr * s)

    name = get(cfg, "run_name", obj_name == "none" ? "$(mode)_$(arm_name(cfg))_$(level)" : "$(mode)_$(obj_name)_$(arm_name(cfg))_$(level)")
    dir = run_dir(cfg, "finetune", name)
    cfg = merge(cfg, Dict("wandb_name" => "$(name)_s$(seed)"))
    @printf("finetune %s | %s tokens | %d classes | train %d val %d test %d reloc %d | head_init dim-sd %.3g | dir %s\n",
            name, cfg["tokens"], C, length(ytr), length(yva), length(yte), length(sp.reloc), hsd, dir)

    # single cells: train batches from every train cell of the level's classes (class-balanced as cfg balanced), test cells grouped by pseudobulk
    stream = cell_stream(d, cfg; level, seed = seed + 3000, balanced = get(cfg, "balanced", level == "lvl2") == true)
    feed = isnothing(stream) ? nothing : bs -> ((X, yb) = next_batch!(stream, bs); (model_input(input, X, d, cfg), yb))
    r = train_classifier!(model, opt, (m, x, t) -> _ft_grad(m, x, t, micro), (Xtr, ytr, Xva, yva, Xte, yte), cfg, C; lr_scale!,
                          extra = (reloc = (Xrl, y[sp.reloc]),), feed, groups = is_sc(d) ? pb_groups(d, idx[sp.test]) : nothing,
                          val_groups = is_sc(d) ? pb_groups(d, idx[sp.val]) : nothing)
    # finetuned encoders as checkpoints that load_encoder and the probe read (<dir>/final, <dir>/best); init keys from the encoder's own init
    if get(cfg, "save_model", true)
        ex = Dict{String,Any}("n_genes" => nid, "n_ranks" => G, "tokens" => cfg["tokens"], "init_scheme" => string(pex["init_scheme"]), "x_mean" => pex["x_mean"],
                              "x_sd" => pex["x_sd"], "stage" => "finetune", "level" => level, "mode" => mode, "git" => git_hash(), "gpu" => gpu_name())
        for (k, st) in (("final", r.final_state), ("best", r.best_state))
            Flux.loadmodel!(model, st); save_checkpoint(joinpath(dir, k), model.encoder, merge(cfg, Dict("objective" => obj_name)); extra = merge(ex, Dict("step" => k == "final" ? r.steps : r.best.step)))
        end
    end
    return save_classifier_run(dir, cfg, r, d, idx, sp, y, names, (name = name, input = cfg["input"], tokens = token_label(cfg["tokens"], cfg), mode = mode, objective = obj_name), hsd)
end

# batch gradient, optionally accumulated over micro-batches (same gradient as one pass)
function _ft_grad(model, x, y, micro::Integer)
    B = size(x, 2)
    if micro <= 0 || micro >= B
        r = Flux.withgradient(m -> Flux.logitcrossentropy(m(x), y), model)
        return r.val, r.grad[1]
    end
    lv = 0f0; g = nothing
    for s in 1:micro:B
        c = s:min(s + micro - 1, B); w = Float32(length(c) / B)
        xb = x[:, c]; yb = y[:, c]
        r = Flux.withgradient(m -> Flux.logitcrossentropy(m(xb), yb) * w, model)
        lv += r.val; g = isnothing(g) ? r.grad[1] : Flux.Zygote.accum(g, r.grad[1])
    end
    return lv, g
end

# finetune inputs (train, val, test, reloc): the model input of the labelled samples (cfg tokens; etf_scale for ETF)
function finetune_inputs(cfg::Dict, d::TahoeData, idx, sp)
    X = model_input(parse_input(cfg["input"]), raw_input(d, idx, cfg), d, cfg)
    return X[:, sp.train], X[:, sp.val], X[:, sp.test], X[:, get(sp, :reloc, Int[])]
end
