# tsv logging and optional Weights & Biases (cfg wandb_project = "" turns it off)

# append one NamedTuple row to a tsv (header written once)
function append_row(path::String, row::NamedTuple)
    mkpath(dirname(path))
    new = !isfile(path)
    open(path, "a") do io
        new && println(io, join(string.(keys(row)), '\t'))
        println(io, join([v isa AbstractFloat ? @sprintf("%.6g", v) : string(v) for v in values(row)], '\t'))
    end
end

function write_curve(path::String, rows)
    isfile(path) && rm(path)
    foreach(r -> append_row(path, r), rows)
end

# pretraining log: tsv per eval + stdout + wandb
struct PretrainLog
    path::String
    cfg::Dict
end
pretrain_log(dir::String, cfg::Dict) = (log_wandb_init(cfg, "pretrain"); PretrainLog(joinpath(dir, "log.tsv"), cfg))

function log_eval!(log::PretrainLog, step, train, val, trivial, gmax, elapsed; extra::NamedTuple = (;))
    row = merge((step = step, train = train, val = val, trivial = trivial, ratio = val / trivial, grad_norm_max = gmax, elapsed_s = elapsed), extra)
    append_row(log.path, row)
    @printf("step %d | train %.4f | val %.4f | trivial %.4f | ratio %.4f | grad max %.2f | %.0f s\n", step, train, val, trivial, val / trivial, gmax, elapsed)
    flush(stdout)
    log_wandb(log.cfg, Dict(string(k) => v for (k, v) in pairs(row)))
end
close_log!(log::PretrainLog) = log_wandb_finish()

# wandb through PyCall (python env from the PYTHON / venv on PATH)
const WANDB_RUN = Ref{Any}(nothing)
_wandb_on(cfg) = get(cfg, "wandb_project", "") != ""

function log_wandb_init(cfg::Dict, job_type::String)
    _wandb_on(cfg) || return nothing
    wandb = PyCall.pyimport("wandb")
    WANDB_RUN[] = wandb.init(project=cfg["wandb_project"], group=get(cfg, "wandb_group", ""), job_type=job_type, name=get(cfg, "wandb_name", nothing),
                             config=Dict(k => v for (k, v) in cfg if v isa Union{Number,String,Bool}), reinit=true)
    return WANDB_RUN[]
end

function log_wandb(cfg::Dict, d::Dict)
    _wandb_on(cfg) || return nothing
    isnothing(WANDB_RUN[]) && log_wandb_init(cfg, get(cfg, "job_type", "finetune"))
    WANDB_RUN[].log(d)
end

function log_wandb_summary(cfg::Dict, d::Dict)
    (_wandb_on(cfg) && !isnothing(WANDB_RUN[])) || return nothing
    for (k, v) in d
        WANDB_RUN[].summary[k] = v
    end
    log_wandb_finish()
end

log_wandb_finish() = (isnothing(WANDB_RUN[]) || (WANDB_RUN[].finish(); WANDB_RUN[] = nothing); nothing)
