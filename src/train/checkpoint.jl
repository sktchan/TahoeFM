# checkpoints: encoder state + heads + the config needed to rebuild them

function save_checkpoint(dir::String, encoder::Encoder, cfg::Dict; heads = nothing, extra = Dict{String,Any}())
    mkpath(dir)
    jldsave(joinpath(dir, "checkpoint.jld2"); encoder=Flux.state(cpu(encoder)),
            heads=isnothing(heads) ? nothing : Flux.state(cpu(heads)), config=cfg, extra=extra)
    open(io -> TOML.print(io, Dict(k => v for (k, v) in cfg if v isa Union{Number,String,Bool})), joinpath(dir, "config.toml"), "w")
    return dir
end

# rebuild the encoder of a pretraining run (weights loaded unless random = true; random uses the run's init scheme)
function load_encoder(dir::String; random::Bool = false, init_seed::Int = 0)
    ck = load(joinpath(dir, "checkpoint.jld2"))
    cfg = ck["config"]; ex = ck["extra"]
    input = parse_input(cfg["input"])
    random && Random.seed!(init_seed)
    enc = Encoder(input; n_genes=ex["n_genes"], n_ranks=get(ex, "n_ranks", ex["n_genes"]), embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
                  n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], drop_prob=Float64(cfg["drop_prob"]), norm=get(cfg, "norm", "pre"))
    if random
        init_encoder!(enc, Symbol(ex["init_scheme"]); seed=init_seed, x_mean=ex["x_mean"], x_sd=ex["x_sd"])
    else
        Flux.loadmodel!(enc, ck["encoder"])
    end
    return enc, cfg, ex
end

# config of a pretraining checkpoint, and the input settings that finetune / probe / health take from it
ckpt_config(dir::AbstractString) = load(joinpath(dir, "checkpoint.jld2"), "config")
ckpt_input_cfg(pcfg::AbstractDict) = merge(Dict{String,Any}("input" => pcfg["input"], "tokens" => token_set(pcfg)),
                                           Dict{String,Any}(k => pcfg[k] for k in ("top_k", "topk_rule", "panel", "rank_shuffle", "etf_scale", "norm") if haskey(pcfg, k)))
