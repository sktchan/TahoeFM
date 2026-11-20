# pretraining loop for every objective, input and token set

function pretrain(cfg::Dict; data::Union{Nothing,TahoeData} = nothing)
    set_math!(cfg)
    seed = cfg["seed"]; bs = cfg["batch_size"]; total = cfg["max_steps"]
    eval_every = get(cfg, "eval_every", 1000); max_val = get(cfg, "max_val_samples", 5000)
    wd = Float64(get(cfg, "weight_decay", 0.0)); clip = Float64(get(cfg, "clip", 1.0))
    input = parse_input(cfg["input"]); obj = objective_from(cfg)
    cfg = merge(cfg, Dict("tokens" => token_set(cfg))); topk = cfg["tokens"] == "topk"
    d = cfg_data(cfg, data)
    nid, G = token_dims(d, cfg)   # gene table rows; tokens per sample (= rank table rows)
    sp = pretrain_split(d)
    Xtr = raw_input(d, sp.train, cfg)
    nv = min(max_val, length(sp.val)); Xva = raw_input(d, sp.val[1:nv], cfg)
    # single cells: Xtr = the in-memory train cells (statistics, trivials, warm-up); the training batches come from every train cell through the stream
    stream = cell_stream(d, cfg; seed = seed + 3000)

    # per-gene train stats (erecon target): panel from the train columns, topk the stored all-gene train stats
    mu_g, sd_g = topk ? (d.full.gene_mean, max.(d.full.gene_sd, 1f-6)) : (vec(mean(Xtr, dims=2)), max.(vec(std(Xtr, dims=2)), 1f-6))
    # global stats of the ETF model input (tok_norm init), over the values actually fed (padding excluded)
    xv = real_values(Xtr)
    x_mean, x_sd = cfg["etf_scale"] == "raw" ? (Float64(mean(xv)), Float64(std(xv))) :
                   (xs = real_values(model_input(ETF(), Xtr, d, cfg)); (Float64(mean(xs)), Float64(std(xs))))
    ctx0 = (G = G, n_ids = nid, topk = topk, medians = d.medians, mu_g = mu_g, sd_g = sd_g, input = input)

    # model: encoder + objective heads, built right after the seed
    seed_all!(seed)
    enc = Encoder(input; n_genes=nid, n_ranks=G, embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"], n_heads=cfg["n_heads"],
                  n_layers=cfg["n_layers"], drop_prob=Float64(cfg["drop_prob"]), norm=cfg["norm"])
    scheme = Symbol(get(cfg, "init", string(init_scheme(obj))))
    init_encoder!(enc, scheme; seed, x_mean, x_sd)
    heads = make_heads(obj, cfg["embed_dim"], cfg["hidden_dim"], mlm_classes(input, ctx0))
    model = (encoder = enc, heads = heads)

    name = "$(cfg["objective"])_$(arm_name(cfg))"
    dir = run_dir(cfg, "pretrain", get(cfg, "run_name", name))
    cfg = merge(cfg, Dict("wandb_name" => "$(get(cfg, "run_name", name))_s$(seed)"))
    extra = Dict{String,Any}("n_genes" => nid, "n_ranks" => G, "tokens" => cfg["tokens"], "init_scheme" => string(scheme), "x_mean" => x_mean, "x_sd" => x_sd,
                             "git" => git_hash(), "gpu" => gpu_name())
    save_checkpoint(joinpath(dir, "init"), enc, cfg; heads, extra)
    cn = channel_norms(enc, model_input(input, Xtr[:, 1:bs], d, cfg))
    @printf("pretrain %s%s | init %s: gene %.3f value %.3f mask %.3f | train %d val %d (eval on %d) | dir %s\n",
            name, isnothing(stream) ? "" : " (single cells, stream)", scheme, cn.gene, cn.value, cn.mask, length(sp.train), length(sp.val), nv, dir)
    flush(stdout)

    model = to_device(model)
    teacher = uses_teacher(obj) ? Flux.testmode!(deepcopy(model.encoder)) : nothing
    opt = Flux.setup(OptimiserChain(ClipNorm(clip), Optimisers.AdamW(Float64(cfg["lr"]), (0.9, 0.999), wd)), model)
    warmup = max(1, div(total, 10))

    # fixed val masks and inputs; same masks for ETF and RTF
    hv = hide_mask(G, nv, obj.mask_ratio, MersenneTwister(1001), pad_len(Xva))
    xin_va = model_input(input, Xva, d, cfg)
    zero_mse = obj isa ERecon ? Float64(mean(abs2, targets(obj, input, Xva, xin_va, hv, ctx0).z)) : NaN
    ctx = merge(ctx0, (zero_mse = zero_mse,), topk_trivials(obj, input, Xtr, Xva, xin_va, hv, ctx0))
    triv = trivial_loss(obj, ctx); triv_alt = trivial_alt(obj, ctx)
    # lrecon topk: per-gene running means of the teacher states, warmed on train PBs (never val)
    if uses_teacher(obj) && topk && obj.topk_norm == :gene
        init_centre!(obj, teacher, nid)
        wc = randperm(MersenneTwister(seed + 4000), size(Xtr, 2))[1:min(LRECON_WARM, size(Xtr, 2))]
        for c in Iterators.partition(wc, bs); update_centre!(obj, teacher, dev(model_input(input, Xtr[:, c], d, cfg))); end
    end

    # val loss as the mean over hidden tokens, with per-gene / per-rank / per-sample sums (diag_*.jld2)
    R_va = token_ranks(Xva, d); ids_va = token_genes(Xva)
    function val_eval()
        Flux.testmode!(model)
        D = new_diag(nid, nv; n_rank = G)
        for s in 1:bs:nv
            c = s:min(s + bs - 1, nv)
            xc = dev(xin_va[:, c])
            t = prepare_targets(obj, teacher, xc, map(dev, targets(obj, input, Xva[:, c], xin_va[:, c], hv[:, c], ctx)))
            st = token_stats(obj, input, model, xc, hv[:, c], t; teacher)
            add_diag!(D, st, t.idx, c, R_va, G; ids = ids_va)
        end
        Flux.trainmode!(model)
        return sum(D["gene_loss"]) / sum(D["gene_n"]), D
    end
    # per-eval scale checks: pooled sd of the downstream encoder, raw teacher target sd (lrecon)
    xin_sd = xin_va[:, 1:min(nv, 1000)]
    function scale_stats()
        Flux.testmode!(model)
        r = (pool_sd = pooled_sd(_ckpt_encoder(model, teacher), xin_sd, bs),
             tgt_raw_sd = isnothing(teacher) ? NaN : teacher_raw_sd(teacher, dev(xin_va[:, 1:min(nv, bs)])))
        Flux.trainmode!(model)
        return r
    end
    val_idx = sp.val[1:nv]

    println("model on $(gpu_name()) ($(CUDA.functional() ? "$(CUDA.math_mode()) $(CUDA.math_precision())" : "cpu")), first eval"); flush(stdout)
    log = pretrain_log(dir, cfg)
    crng = MersenneTwister(seed + 1000)          # hidden genes (shared by ETF and RTF)
    brng = MersenneTwister(seed + 3000)          # batch order (shared by ETF and RTF)
    perm = randperm(brng, size(Xtr, 2)); pos = 1
    best = Inf; step = 0; t0 = time(); Dv = nothing
    v0, D0 = val_eval()
    save_diag(joinpath(dir, "diag_init.jld2"), D0, 0, val_idx)
    log_eval!(log, 0, NaN, v0, triv, 0.0, 0.0; extra=merge((lr = 0.0, trivial_alt = triv_alt), scale_stats()))
    while step < total
        losses = Float64[]; gmax = 0.0
        for _ in 1:min(eval_every, total - step)
            if isnothing(stream)
                pos + bs - 1 > length(perm) && (perm = randperm(brng, size(Xtr, 2)); pos = 1)
                cols = perm[pos:pos+bs-1]; pos += bs
                Xb = Xtr[:, cols]
            else
                Xb = first(next_batch!(stream, bs))
            end
            xin = model_input(input, Xb, d, cfg)
            hide = hide_mask(G, bs, obj.mask_ratio, crng, pad_len(Xb))
            x = dev(xin)
            t = prepare_targets(obj, teacher, x, map(dev, targets(obj, input, Xb, xin, hide, ctx)); update = true)
            step += 1
            Optimisers.adjust!(opt, cosine_lr(step, total, Float64(cfg["lr"]), warmup))
            l, g = Flux.withgradient(m -> pretext_loss(obj, input, m, x, hide, t; teacher), model)
            isfinite(l) || error("non-finite loss at step $step")
            gmax = max(gmax, grad_norm(g[1]))
            Flux.update!(opt, model, g[1])
            isnothing(teacher) || ema_update!(teacher, model.encoder, obj.tau)
            push!(losses, l)
        end
        vl, Dv = val_eval()
        log_eval!(log, step, mean(losses), vl, triv, gmax, time() - t0; extra=merge((lr = cosine_lr(step, total, Float64(cfg["lr"]), warmup), trivial_alt = triv_alt), scale_stats()))
        if vl < best
            best = vl
            save_diag(joinpath(dir, "diag_best.jld2"), Dv, step, val_idx)
            save_checkpoint(joinpath(dir, "best"), _ckpt_encoder(model, teacher), cfg; heads=model.heads, extra=merge(extra, Dict("step" => step), centre_extra(obj)))
        end
    end
    save_diag(joinpath(dir, "diag_final.jld2"), Dv, step, val_idx)
    save_checkpoint(joinpath(dir, "final"), _ckpt_encoder(model, teacher), cfg; heads=model.heads,
                    extra=merge(extra, Dict("step" => step, "s_per_step" => (time() - t0) / max(step, 1)), centre_extra(obj)))
    isnothing(teacher) || save_checkpoint(joinpath(dir, "final_student"), model.encoder, cfg; heads=model.heads, extra=extra)
    final = joinpath(dir, "final")
    @printf("done %s: best val %.4f (trivial %.4f), %.3f s/step, final ckpt %s\n", name, best, triv, (time() - t0) / max(step, 1), final)
    rname = get(cfg, "run_name", name) * (seed == 1 ? "" : "_s$seed")
    ptr = ckpt_pointer(cfg["results_dir"], rname); mkpath(dirname(ptr)); write(ptr, final)
    println("checkpoint pointer $ptr (ckpt=@$rname)")
    model = nothing; teacher = nothing; opt = nothing; GC.gc(); CUDA.functional() && CUDA.reclaim()
    post_eval(cfg, final, "$(get(cfg, "run_name", name))_s$seed", d)
    close_log!(log)
    return final
end

# frozen lvl2 / moa probe and health check of a finished pretrain checkpoint (cfg probe / health = false skips); failures only warn
function post_eval(cfg::Dict, ck::AbstractString, name::AbstractString, d::TahoeData)
    is_sc(d) && return println("probe and health check are not ported to single cells yet: skipped")
    for (key, f) in (("probe", run_probe), ("health", run_health))
        get(cfg, key, true) || continue
        try
            row = f(merge(cfg, Dict{String,Any}("ckpt" => ck, "name" => name)); data=d)
            key == "probe" && log_wandb(cfg, Dict("probe_drug_h2" => row.drug_h2_test, "probe_drug_clc_h2" => row.drug_clc_h2_test, "probe_cell_h2" => row.cell_h2_test,
                                                  "probe_moa_h2" => row.moa_h2))
            key == "health" && log_wandb(cfg, Dict("health_verdict" => row.verdict))
        catch e
            @warn "$key failed for $ck" exception=(e, catch_backtrace())
        end
    end
end

# the encoder that downstream uses: the EMA teacher for objectives with a teacher, else the student
_ckpt_encoder(model, teacher) = isnothing(teacher) ? model.encoder : teacher
