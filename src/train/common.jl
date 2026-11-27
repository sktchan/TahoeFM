# config, schedules and run directories shared by pretraining and finetuning

const DEFAULTS = Dict{String,Any}(
    "embed_dim" => 128, "hidden_dim" => 128, "n_heads" => 2, "n_layers" => 2, "norm" => "pre_final", "etf_scale" => "raw",
    "batch_size" => 100, "seed" => 1, "wandb_project" => "", "wandb_group" => "", "results_dir" => joinpath(PKG_DIR, "results"),
    "data_path" => joinpath(PKG_DIR, "data", "tahoe_pb.jld2"),
    # Tahoe: dataset (pb = pseudobulks, sc = single cells), gene panel of the panel token set, top-k token count; tokens default by input (inputs.jl token_set)
    "dataset" => "pb", "panel" => "hvg", "top_k" => 1024, "rank_shuffle" => false,
    # dataset = sc: raw files read in place, the cell index (scripts/prepare_cells.jl), the in-memory cell sample (shards per plate, cells per split side; the test side = sc_test_per_pb cells of every test pseudobulk, 0 = sc_test_cells as the other sides),
    # and the training stream over every train cell (shards per plate and cells per pool; sc_stream = false trains on the in-memory train cells)
    "raw_dir" => RAW_DIR, "sc_path" => joinpath(PKG_DIR, "data", "tahoe_sc.jld2"), "sc_shards" => 2, "sc_train_cells" => 50000, "sc_val_cells" => 20000,
    "sc_test_cells" => 100000, "sc_test_per_pb" => 50, "sc_reloc_cells" => 20000, "sc_stream" => true, "sc_pool_shards" => 3, "sc_pool_cells" => 30000, "sc_prefetch" => 1)

# toml file merged over its base = "<file>" (path relative to the file), if set
function read_toml(path::AbstractString)
    t = TOML.parsefile(path)
    b = pop!(t, "base", nothing)
    return isnothing(b) ? t : merge(read_toml(joinpath(dirname(path), b)), t)
end

# config = defaults < base toml < toml file < overrides (key = value pairs)
function load_config(path::Union{Nothing,AbstractString} = nothing, overrides::AbstractDict = Dict{String,Any}(); kw...)
    cfg = copy(DEFAULTS)
    isnothing(path) || merge!(cfg, read_toml(path))
    for (k, v) in Iterators.flatten((overrides, kw))
        cfg[string(k)] = v
    end
    # single-cell runs keep their own results tree (rows have extra columns; checkpoint pointers must not collide with the pseudobulk ones)
    cfg["dataset"] == "sc" && cfg["results_dir"] == DEFAULTS["results_dir"] && (cfg["results_dir"] = joinpath(PKG_DIR, "results", "sc"))
    ck = get(cfg, "ckpt", nothing)
    ck isa AbstractString && startswith(ck, "@") && (cfg["ckpt"] = resolve_ckpt(ck[2:end], cfg["results_dir"]))
    return cfg
end

# parse "key=value" command-line overrides (numbers and bools parsed, else strings)
function parse_overrides(args)
    out = Dict{String,Any}()
    for a in args
        k, v = split(a, "="; limit=2)
        b = lowercase(v)
        out[k] = something(tryparse(Int, v), tryparse(Float64, v), b == "true" ? true : b == "false" ? false : String(v))
    end
    return out
end

# script config from the command line: [<config.toml>] key=value ...; prints the trial line under a wandb sweep agent
function cli_config(args)
    haskey(ENV, "WANDB_SWEEP_ID") && println("sweep trial $(ENV["WANDB_SWEEP_ID"]) run $(get(ENV, "WANDB_RUN_ID", "?")) | $(join(args, ' '))")
    file = !isempty(args) && !occursin('=', args[1])
    return load_config(file ? args[1] : nothing, parse_overrides(file ? args[2:end] : args))
end

# checkpoint pointers: <results_dir>/ckpt/<name> holds the final checkpoint of a finished pretrain run (ckpt=@name)
ckpt_pointer(results_dir, name) = joinpath(results_dir, "ckpt", name)
function resolve_ckpt(name::AbstractString, results_dir::AbstractString)
    f = ckpt_pointer(results_dir, name)
    isfile(f) || error("no checkpoint pointer $f (written when pretrain run $name finishes)")
    return String(strip(read(f, String)))
end

# linear warmup then cosine to 0
cosine_lr(step::Int, total::Int, base::Float64, warmup::Int) = step <= warmup ? base * step / warmup :
    base * 0.5 * (1 + cos(pi * min(1.0, (step - warmup) / max(1, total - warmup))))

# global gradient norm
function grad_norm(g)
    s = Ref(0.0)
    fmap(x -> (x isa AbstractArray{<:Number} && (s[] += Float64(sum(abs2, x))); x), g; exclude = x -> x isa AbstractArray)
    return sqrt(s[])
end

seed_all!(seed::Int) = (Random.seed!(seed); CUDA.functional() && CUDA.seed!(seed); nothing)

# results/<stage>/<name>/<timestamp>_s<seed>_j<job>
function run_dir(cfg::Dict, stage::String, name::String)
    ts = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    job = get(ENV, "SLURM_JOB_ID", "local")
    d = joinpath(cfg["results_dir"], stage, name, "$(ts)_s$(cfg["seed"])_j$(job)")
    mkpath(d)
    return d
end

gpu_name() = CUDA.functional() ? CUDA.name(CUDA.device()) : "cpu"
git_hash() = (try readchomp(pipeline(`git -C $PKG_DIR rev-parse --short HEAD`; stderr=devnull)) catch; "nogit" end)

# columns of X in batches of bs, as device inputs
batches(f, X, bs) = (f(X[:, s:min(s + bs - 1, size(X, 2))]) for s in 1:bs:size(X, 2))

# TF32 matmuls (as the Nov 2026 fastobj runs); cfg tf32 = false keeps full Float32; no-op without a functional GPU (CPU runs)
set_math!(cfg::Dict) = (CUDA.functional() && (get(cfg, "tf32", true) ? CUDA.math_mode!(CUDA.FAST_MATH; precision=:TensorFloat32) : CUDA.math_mode!(CUDA.DEFAULT_MATH)); nothing)
