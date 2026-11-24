# supervised training loop shared by finetune() and baseline(): sampling, periodic val eval, best-val tracking, test eval

# mean loss, accuracy and predictions over all columns of Xs; full = also the true-class probability and the top-10 classes / probabilities
# groups = a group id per column (single cells: their pseudobulk, 0 = none): also the class with the highest mean log-probability per group
function evaluate_classifier(m, Xs, ys, C::Int, bs::Int; full::Bool = false, groups::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    Flux.testmode!(m); loss = 0.0; preds = Int[]
    k = min(10, C); n = size(Xs, 2)
    gsum = isnothing(groups) ? nothing : zeros(Float64, C, isempty(groups) ? 0 : maximum(groups))
    p_true = full ? zeros(Float32, n) : nothing
    topk = full ? zeros(Int16, k, n) : nothing; topk_p = full ? zeros(Float16, k, n) : nothing
    for s in 1:bs:n
        c = s:min(s + bs - 1, n)
        lg = m(dev(Xs[:, c]))
        loss += Float64(Flux.logitcrossentropy(lg, dev(Flux.onehotbatch(ys[c], 1:C)))) * length(c)
        L = Array(lg)
        append!(preds, Flux.onecold(L))
        if !isnothing(gsum)
            Pg = logsoftmax(L; dims=1)
            for (j, i) in enumerate(c); groups[i] > 0 && (gsum[:, groups[i]] .+= view(Pg, :, j)); end
        end
        if full
            P = softmax(L; dims=1)
            for (j, i) in enumerate(c)
                p_true[i] = P[ys[i], j]
                o = partialsortperm(view(P, :, j), 1:k; rev=true)
                topk[:, i] .= o; topk_p[:, i] .= P[o, j]
            end
        end
    end
    Flux.trainmode!(m)
    return loss / n, mean(preds .== ys), preds, (p_true = p_true, top10 = topk, top10_p = topk_p, group_pred = isnothing(gsum) ? nothing : Int[argmax(view(gsum, :, g)) for g in axes(gsum, 2)])
end

# grad(model, x, y) -> (loss, gradient); cfg balanced (default: lvl2) samples class-balanced (uniform class, then uniform sample), else shuffled epochs
# eval_every (steps, default one epoch); best = highest val acc, ties by lower val loss; always runs max_steps
# cfg select = "val" (default, as LincsFM) keeps that best-val model as "best"; "final" uses no validation selection ("best" = the final-step model)
# lr_scale!(s) sets every lr to s x its base value; with it, cfg warmup_steps gives a linear warmup then a constant lr
# extra = named (X, y) evaluation sets (e.g. reloc), predicted with the final and the best-val model next to test
# feed(bs) -> (model input, class ids) replaces the in-memory train batches (single cells: the stream over every train cell); groups = pseudobulk id per test column
function train_classifier!(model, opt, grad, data, cfg::Dict, C::Int; lr_scale! = nothing, extra::NamedTuple = (;), feed = nothing, groups = nothing, val_groups = nothing)
    Xtr, ytr, Xva, yva, Xte, yte = data
    bs = cfg["batch_size"]; total = cfg["max_steps"]; balanced = get(cfg, "balanced", cfg["level"] == "lvl2") == true
    sel = get(cfg, "select", "val"); sel in ("val", "final") || error("select must be val or final, got $sel")
    wu = isnothing(lr_scale!) ? 0 : Int(get(cfg, "warmup_steps", 0))
    wu > 0 && lr_scale!(1 / wu)
    bpe = div(length(ytr), bs); eval_every = get(cfg, "eval_every", bpe)
    cidx = Dict{Int,Vector{Int}}(); for (i, c) in enumerate(ytr); push!(get!(cidx, c, Int[]), i); end
    classes = collect(keys(cidx))
    perm = randperm(length(ytr)); pos = 1
    function next_batch()
        balanced && return [rand(cidx[rand(classes)]) for _ in 1:bs]
        pos + bs - 1 > length(perm) && (perm = randperm(length(ytr)); pos = 1)
        b = perm[pos:pos+bs-1]; pos += bs
        return b
    end

    curve = NamedTuple[]; best = (acc = -Inf, loss = Inf, step = 0); best_state = nothing
    step = 0; losses = Float64[]; t0 = time()
    while step < total
        xb, yb = isnothing(feed) ? (b = next_batch(); (Xtr[:, b], ytr[b])) : feed(bs)
        l, g = grad(model, dev(xb), dev(Flux.onehotbatch(yb, 1:C)))
        Flux.update!(opt, model, g)
        push!(losses, l); step += 1
        wu > 0 && step < wu && lr_scale!((step + 1) / wu)
        (step % eval_every == 0 || step == total) || continue
        vl, va, _, ve = evaluate_classifier(model, Xva, yva, C, bs; groups = val_groups)
        ep = step / bpe
        # single cells: val accuracy at the pseudobulk unit, logged only (selection stays on val_acc)
        vpb = isnothing(val_groups) ? (;) : (val_acc_pb = group_acc(ve.group_pred, val_groups, yva),)
        push!(curve, merge((step = step, epoch = ep, train = mean(losses), val_loss = vl, val_acc = va), vpb))
        @printf("step %d (epoch %.2f) train %.4f val %.4f val_acc %.4f%s | %.0f s\n", step, ep, mean(losses), vl, va, isempty(vpb) ? "" : @sprintf(" val_acc_pb %.4f", vpb.val_acc_pb), time() - t0); flush(stdout)
        log_wandb(cfg, Dict("step" => step, "epoch" => ep, "train_loss" => mean(losses), "val_loss" => vl, "val_acc" => va, "best_val_acc" => max(best.acc, va)))
        losses = Float64[]
        if va > best.acc || (va == best.acc && vl < best.loss)
            best = (acc = va, loss = vl, step = step); best_state = Flux.state(cpu(model))
        end
    end
    wall = time() - t0
    _, acc_final, p_final, e_final = evaluate_classifier(model, Xte, yte, C, bs; full=true, groups)
    x_final = map(s -> evaluate_classifier(model, s[1], s[2], C, bs)[3], extra)
    final_state = Flux.state(cpu(model))
    final_val = (acc = curve[end].val_acc, loss = curve[end].val_loss, step = step)   # the last eval is at step max_steps
    sel == "final" && (best = final_val; best_state = final_state)
    Flux.loadmodel!(model, best_state)
    _, acc_best, p_best, e_best = evaluate_classifier(model, Xte, yte, C, bs; full=true, groups)
    x_best = map(s -> evaluate_classifier(model, s[1], s[2], C, bs)[3], extra)
    return (curve = curve, steps = step, wall = wall, best = best, final_val = final_val, final_state = final_state, best_state = best_state,
            acc_final = acc_final, p_final = p_final, acc_best = acc_best, p_best = p_best, e_final = e_final, e_best = e_best,
            x_final = x_final, x_best = x_best)
end

const DOSE_LEVELS = ("0.05 uM", "0.5 uM", "5.0 uM")
const DOSE_TAGS = ("005", "05", "5")

# accuracy columns next to test: reloc wells (final / best model) and the test set per dose level; and the reloc / group data kept in predstrues.jld2
function eval_extras(d, idx, sp, y, p_final, p_best, rl_final, rl_best)
    acc(p, yy) = isempty(yy) ? NaN : mean(p .== yy)
    yte = y[sp.test]; yrl = y[sp.reloc]; dz = d.meta.dose[idx[sp.test]]
    ks = Symbol[:final_reloc, :best_reloc]; vs = Any[acc(rl_final, yrl), acc(rl_best, yrl)]
    for (w, p) in (("final", p_final), ("best", p_best)), (tag, lv) in zip(DOSE_TAGS, DOSE_LEVELS)
        k = dz .== lv; push!(ks, Symbol("$(w)_test_$tag")); push!(vs, acc(p[k], yte[k]))
    end
    push!(ks, :n_reloc); push!(vs, length(yrl))
    cols = NamedTuple{Tuple(ks)}(Tuple(vs))
    keep = (test_groups = d.meta.group[idx[sp.test]], test_dose = dz, trues_reloc = yrl, preds_reloc_final = rl_final, preds_reloc_best = rl_best,
            reloc_sample_idx = idx[sp.reloc], reloc_groups = d.meta.group[idx[sp.reloc]])
    return cols, keep
end

# (block, position) lookup on the val set (lvl2; NaN for lvl1): majority train drug at the PB's block and plate position, no expression
function lookup_val_acc(d, idx, sp, y, level)
    (level == "lvl2" && !isempty(sp.val)) || return NaN
    m = d.meta; bp = collect(zip(m.block[idx], m.pos[idx])); ps = m.pos[idx]; ytr = y[sp.train]
    pv, _ = key_majority(ytr, ps[sp.train], ps[sp.val])
    p, _ = key_majority(ytr, bp[sp.train], bp[sp.val]; fallback = pv)
    return mean(p .== y[sp.val])
end

# accuracy of group predictions gp (group ids g per column, 0 = none) against the column labels y
function group_acc(gp, g, y)
    (isnothing(gp) || isempty(gp)) && return NaN
    yg = zeros(Int, length(gp)); for (j, k) in enumerate(g); k > 0 && (yg[k] = y[j]); end
    return mean(gp .== yg)
end

# single cells: accuracy at the pseudobulk unit (the class with the highest mean log-probability over the test cells of a (well, cell line) that is a pseudobulk of the table), and what predstrues.jld2 keeps
pb_extras(d, idx, sp, y, r) = pb_extras(d, idx, sp, y, r.e_final.group_pred, r.e_best.group_pred)
function pb_extras(d, idx, sp, y, gp_final, gp_best)
    is_sc(d) || return (;), (;)
    g = pb_groups(d, idx[sp.test]); ng = isempty(g) ? 0 : maximum(g); yg = zeros(Int, ng); n = zeros(Int, ng)
    for (j, k) in enumerate(g); k > 0 && (yg[k] = y[sp.test[j]]; n[k] += 1); end
    acc(p) = isnothing(p) || ng == 0 ? NaN : mean(p .== yg)
    cols = (final_test_pb = acc(gp_final), best_test_pb = acc(gp_best), n_test_pb = ng, cells_per_pb = ng == 0 ? NaN : Float64(median(n)), n_test_cells = length(g))
    return cols, (test_cell = d.meta.cell[idx[sp.test]], test_pb = g, pb_trues = yg, pb_cells = n, pb_preds_final = gp_final, pb_preds_best = gp_best)
end

# results row of a classifier run (finetune.tsv): run head, test accuracies, extras (reloc, per dose), val selection (select, final / best val acc, the lookup's val acc), run info
result_row(cfg::Dict, head::NamedTuple, steps, acc_final, acc_best, extras::NamedTuple, best_step, best_val_acc, final_val_acc, lookup_val, hsd, gpu, wall, dir) =
    merge((dataset = get(cfg, "dataset", "pb"),), head, (level = cfg["level"], seed = get(cfg, "seed", 1), steps = steps, final_test = acc_final, best_test = acc_best),
          extras, (select = get(cfg, "select", "val"), best_step = best_step, best_val_acc = best_val_acc, final_val_acc = final_val_acc, lookup_val_acc = lookup_val,
                   head_init_sd = hsd, gpu = gpu, job = get(ENV, "SLURM_JOB_ID", "local"), wall_s = round(Int, wall), git = git_hash(), dir = dir))

# save model states, test (and reloc) predictions, curve, config and the results row; returns the row
function save_classifier_run(dir::String, cfg::Dict, r, d::TahoeData, idx, sp, y, names, row_head::NamedTuple, hsd::Real)
    yte = y[sp.test]
    xc, xk = eval_extras(d, idx, sp, y, r.p_final, r.p_best, get(r.x_final, :reloc, Int[]), get(r.x_best, :reloc, Int[]))
    pc, pk = pb_extras(d, idx, sp, y, r); xc = merge(xc, pc); xk = merge(xk, pk)
    get(cfg, "save_model", true) && jldsave(joinpath(dir, "model.jld2"); final=r.final_state, best=r.best_state, config=cfg)
    jldsave(joinpath(dir, "predstrues.jld2"); preds_final=r.p_final, preds_best=r.p_best, trues=yte,
            ptrue_final=r.e_final.p_true, ptrue_best=r.e_best.p_true, top10_final=r.e_final.top10, top10_best=r.e_best.top10,
            top10p_final=r.e_final.top10_p, top10p_best=r.e_best.top10_p,
            test_sample_idx=idx[sp.test], label_names=names, xk...)
    write_curve(joinpath(dir, "curve.tsv"), r.curve)
    open(io -> TOML.print(io, Dict(k => v for (k, v) in cfg if v isa Union{Number,String,Bool})), joinpath(dir, "config.toml"), "w")
    lv = lookup_val_acc(d, idx, sp, y, cfg["level"])
    row = result_row(cfg, row_head, r.steps, r.acc_final, r.acc_best, xc, r.best.step, r.best.acc, r.final_val.acc, lv, hsd, gpu_name(), r.wall, dir)
    append_row(joinpath(cfg["results_dir"], "finetune.tsv"), row)
    log_wandb_summary(cfg, Dict("final_test" => r.acc_final, "best_test" => r.acc_best, "best_step" => r.best.step, "final_val_acc" => r.final_val.acc, "lookup_val_acc" => lv,
                                "final_reloc" => xc.final_reloc, "best_reloc" => xc.best_reloc))
    @printf("test final %.4f best %.4f (step %d, select %s) | reloc final %.4f best %.4f | val final %.4f best %.4f, (block, position) lookup %.4f | %.0f s\n",
            r.acc_final, r.acc_best, r.best.step, get(cfg, "select", "val"), xc.final_reloc, xc.best_reloc, r.final_val.acc, r.best.acc, lv, r.wall)
    is_sc(d) && @printf("per pseudobulk (mean log-probability over its test cells): final %.4f best %.4f over %d pseudobulks (median %g of their cells in the %d test cells)\n",
                        xc.final_test_pb, xc.best_test_pb, xc.n_test_pb, xc.cells_per_pb, xc.n_test_cells)
    return row
end
