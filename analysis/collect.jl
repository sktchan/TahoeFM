# main results table: mean ± sd test accuracy per arm and level, with reloc and per-dose accuracy (Tahoe) and the trivial baselines
using Statistics, Printf

const TAHOE = dirname(@__DIR__)
rdir = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(TAHOE, "results")
out = length(ARGS) >= 2 ? ARGS[2] : joinpath(rdir, "main_table.tsv")

# tsv -> vector of Dict(column => string)
function read_tsv(path)
    isfile(path) || return Dict{String,String}[]
    lines = filter(!isempty, readlines(path))
    isempty(lines) && return Dict{String,String}[]
    head = split(lines[1], '\t')
    rows = Dict{String,String}[]
    for l in lines[2:end]
        f = split(l, '\t')
        f == head && continue                      # repeated header
        length(f) == length(head) || (@warn "skipping malformed row in $path: $l"; continue)
        push!(rows, Dict(String(h) => String(v) for (h, v) in zip(head, f)))
    end
    return rows
end

num(s) = something(tryparse(Float64, s), NaN)
col(r, k) = num(get(r, k, "NaN"))   # NaN when the column is absent
sd(v) = length(v) > 1 ? std(v) : NaN
pm(m, s) = isnan(s) ? @sprintf("%.4f", m) : @sprintf("%.4f ± %.4f", m, s)

runs = read_tsv(joinpath(rdir, "finetune.tsv"))
isempty(runs) && @warn "no runs in $(joinpath(rdir, "finetune.tsv"))"

# a rerun of the same arm and seed replaces the earlier row
latest = Dict{Tuple{String,String,String},Dict{String,String}}()
# select = final rows (no validation selection) are their own arm, <name>_selfinal
armname(r) = r["name"] * (get(r, "select", "val") == "final" ? "_selfinal" : "")
for r in runs
    latest[(get(r, "dataset", "pb") * "|" * r["level"], armname(r), r["seed"])] = r
end

arms = Dict{Tuple{String,String},Vector{Dict{String,String}}}()
for ((lvl, name, _), r) in latest
    push!(get!(arms, (lvl, name), Dict{String,String}[]), r)
end

const MODE_ORDER = Dict("baseline" => 1, "scratch" => 2, "random" => 3, "pretrained" => 4)
table = NamedTuple[]
for ((lvl, name), rs) in arms
    fin = num.(getindex.(rs, "final_test")); best = num.(getindex.(rs, "best_test")); bva = num.(getindex.(rs, "best_val_acc"))
    r1 = rs[1]
    push!(table, (level = lvl, arm = name, mode = r1["mode"], objective = r1["objective"], input = r1["input"], tokens = get(r1, "tokens", "-"), n_seeds = length(rs),
                  seeds = join(sort(getindex.(rs, "seed")), ","), steps = join(unique(getindex.(rs, "steps")), ","),
                  final_mean = mean(fin), final_sd = sd(fin), best_mean = mean(best), best_sd = sd(best), best_val_acc = mean(bva),
                  final_val_acc = mean(col.(rs, "final_val_acc")), lookup_val_acc = mean(col.(rs, "lookup_val_acc")),
                  final_reloc = mean(col.(rs, "final_reloc")), final_005 = mean(col.(rs, "final_test_005")), final_05 = mean(col.(rs, "final_test_05")),
                  final_5 = mean(col.(rs, "final_test_5"))))
end

# trivial label baselines (one row each; latest wins)
triv = Dict{Tuple{String,String},Dict{String,String}}()
for r in read_tsv(joinpath(rdir, "trivial.tsv"))
    triv[(get(r, "dataset", "pb") * "|" * r["level"], r["baseline"])] = r
end
for ((lvl, b), r) in triv
    acc = num(r["acc"])
    push!(table, (level = lvl, arm = b, mode = "trivial", objective = b, input = "-", tokens = "-", n_seeds = 1, seeds = "-", steps = "-",
                  final_mean = acc, final_sd = NaN, best_mean = acc, best_sd = NaN, best_val_acc = NaN, final_val_acc = NaN, lookup_val_acc = NaN,
                  final_reloc = col(r, "reloc_acc"), final_005 = col(r, "acc_005"), final_05 = col(r, "acc_05"), final_5 = col(r, "acc_5")))
end

isempty(table) && (println("nothing to collect in $rdir"); exit())
sort!(table, by = t -> (t.level, get(MODE_ORDER, t.mode, 0), t.objective, t.input, t.arm))

open(out, "w") do io
    println(io, join(string.(keys(table[1])), '\t'))
    for t in table
        println(io, join([v isa AbstractFloat ? @sprintf("%.6g", v) : string(v) for v in values(t)], '\t'))
    end
end

for lvl in unique(getfield.(table, :level))
    println("\n== $lvl")
    @printf("%-34s %-10s %-9s %-5s %-5s %5s  %-18s %-18s %-7s %-20s\n", "arm", "mode", "objective", "input", "toks", "seeds", "final test", "best-val test", "reloc", "dose .05/.5/5")
    for t in filter(t -> t.level == lvl, table)
        @printf("%-34s %-10s %-9s %-5s %-5s %5d  %-18s %-18s %-7.4f %.3f/%.3f/%.3f\n", t.arm, t.mode, t.objective, t.input, t.tokens, t.n_seeds,
                pm(t.final_mean, t.final_sd), pm(t.best_mean, t.best_sd), t.final_reloc, t.final_005, t.final_05, t.final_5)
    end
end
println("\nwrote $out")
