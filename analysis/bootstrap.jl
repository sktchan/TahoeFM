# paired bootstrap comparison of finished runs (clusters = drug|dose groups, i.e. wells, by default)
using Pkg; Pkg.activate(joinpath(dirname(@__DIR__), "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
using TahoeFM
out = isabspath(ARGS[1]) ? ARGS[1] : joinpath(TahoeFM.PKG_DIR, "results", "bootstrap", ARGS[1])
arms = Pair{String,String}[]; contrasts = Tuple{Vector{String},Bool}[]; seed_gap = 0.0; all_pairs = false; B = 2000; cluster = "group"
i = 2
while i <= length(ARGS)
    a = ARGS[i]
    if a == "--contrast" || a == "--did"
        p = String.(split(ARGS[i+1], ':')); k = a == "--did" ? 4 : 2
        length(p) >= k || error("$a needs $(k == 2 ? "A:B" : "A:B:C:D")")
        push!(contrasts, (p[1:k], length(p) > k && p[k+1] == "pt")); global i += 2
    elseif a == "--seed_gap"
        global seed_gap = parse(Float64, ARGS[i+1]); global i += 2
    elseif a == "--all_pairs"
        global all_pairs = ARGS[i+1] == "1"; global i += 2
    elseif a == "--B"
        global B = parse(Int, ARGS[i+1]); global i += 2
    elseif a == "--cluster"
        global cluster = ARGS[i+1]; global i += 2
    else
        k = findfirst('=', a); isnothing(k) && error("arm must be name=source: $a")
        push!(arms, a[1:k-1] => a[k+1:end]); global i += 1
    end
end
TahoeFM.run_bootstrap(out, arms; contrasts, seed_gap, all_pairs, B, cluster)
