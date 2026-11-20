# trivial label baselines on the test (and reloc) set: chance, majority class, majority train class per plate / cell line / (plate, dose), and for lvl2 the (block, position) lookup

# most frequent value, ties -> smallest class id
function _majority(v::AbstractVector{<:Integer})
    cnt = countmap(v)
    return minimum(c for (c, k) in cnt if k == maximum(values(cnt)))
end

# per-key majority train class for keys k; unseen keys -> fallback prediction (default: the global train majority); returns predictions and the fraction of keys seen in train
function key_majority(ytr, ktr, k; fallback = fill(_majority(ytr), length(k)))
    groups = Dict{eltype(ktr),Vector{Int}}()
    for (y, kk) in zip(ytr, ktr); push!(get!(groups, kk, Int[]), y); end
    best = Dict(kk => _majority(v) for (kk, v) in groups)
    return [get(best, kk, f) for (kk, f) in zip(k, fallback)], (isempty(k) ? NaN : mean(haskey(best, kk) for kk in k))
end

# cfg = a config whose dataset is loaded when no data is given (dataset = sc: the in-memory cell sample)
function trivial_baselines(level::AbstractString = "lvl2"; data::Union{Nothing,TahoeData} = nothing, cfg::Union{Nothing,AbstractDict} = nothing,
                           results_dir::AbstractString = joinpath(PKG_DIR, "results"), data_path::AbstractString = default_data_path())
    d = !isnothing(data) ? data : isnothing(cfg) ? load_data(data_path) : cfg_data(merge(Dict{String,Any}("input" => "etf"), cfg))
    ds = dataset_name(d)
    idx, y, names = labels(d, level; reloc = true)
    sp = finetune_split(d, idx)
    m = d.meta
    plate = m.plate[idx]; cell = m.cell_line[idx]; pd = collect(zip(m.plate[idx], m.dose[idx])); bp = collect(zip(m.block[idx], m.pos[idx])); ps = m.pos[idx]
    ytr, yte, yrl = y[sp.train], y[sp.test], y[sp.reloc]
    dz = m.dose[idx[sp.test]]
    C = length(names); n = length(yte); g = _majority(ytr)
    rows = NamedTuple[]
    acc(p, yy) = isempty(yy) ? NaN : mean(p .== yy)
    function add(b, pte, prl, seen)
        r = (dataset = ds, level = level, baseline = b, n_classes = C, n_train = length(ytr), n_test = n, acc = acc(pte, yte), test_key_in_train = seen)
        dc = NamedTuple{Tuple(Symbol.("acc_" .* DOSE_TAGS))}(Tuple(acc(pte[dz .== lv], yte[dz .== lv]) for lv in DOSE_LEVELS))
        push!(rows, merge(r, dc, (n_reloc = length(yrl), reloc_acc = acc(prl, yrl))))
    end
    push!(rows, (dataset = ds, level = level, baseline = "chance", n_classes = C, n_train = length(ytr), n_test = n, acc = 1 / C, test_key_in_train = NaN,
                 acc_005 = 1 / C, acc_05 = 1 / C, acc_5 = 1 / C, n_reloc = length(yrl), reloc_acc = 1 / C))
    add("majority", fill(g, n), fill(g, length(yrl)), NaN)
    for (b, k) in (("plate_majority", plate), ("cell_majority", cell), ("plate_dose_majority", pd))
        pte, seen = key_majority(ytr, k[sp.train], k[sp.test]); prl, _ = key_majority(ytr, k[sp.train], k[sp.reloc])
        add(b, pte, prl, seen)
    end
    # (block, position) lookup: majority train drug at the sample's block and plate position; unseen (e.g. plate 13) -> majority at the position over all blocks
    if level == "lvl2"
        pos_te, _ = key_majority(ytr, ps[sp.train], ps[sp.test]); pos_rl, _ = key_majority(ytr, ps[sp.train], ps[sp.reloc])
        pte, seen = key_majority(ytr, bp[sp.train], bp[sp.test]; fallback = pos_te); prl, _ = key_majority(ytr, bp[sp.train], bp[sp.reloc]; fallback = pos_rl)
        add("block_pos_lookup", pte, prl, seen)
    end
    path = joinpath(results_dir, "trivial.tsv")
    foreach(r -> append_row(path, r), rows)
    foreach(r -> @printf("%s %-20s acc %.4f (test keys in train %.3f) | by dose %.4f %.4f %.4f | reloc %.4f\n", r.level, r.baseline, r.acc, r.test_key_in_train,
                         r.acc_005, r.acc_05, r.acc_5, r.reloc_acc), rows)
    return rows
end
