# writes a wandb sweep yaml from configs/sweeps/<stage>.yaml and creates the sweep
using Pkg; Pkg.activate(joinpath(dirname(@__DIR__), "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
using TahoeFM, Printf

name = ARGS[1]
fixed = Dict{String,String}(String.(split(a, "="; limit=2)) for a in ARGS[2:end])
dry = lowercase(pop!(fixed, "dry", "false")) == "true"
# W&B project for the sweep and its trials (finals go to tahoe-pb-lvl1-runs / tahoe-pb-lvl2-runs, tests to tahoe-test); dataset=sc sweeps single cells (tahoe-sc-*)
project = pop!(fixed, "project", "tahoe-$(get(fixed, "dataset", "pb"))-lvl2-sweeps")
# trial length: sweep_steps=<n>; default = the LincsFM protocol (lvl1 10k, else 40k), TBD on Tahoe by the sweep-length check
steps = pop!(fixed, "sweep_steps", "")

stage, arm = String.(split(name, '_'; limit=2))
level = stage == "pt" ? "" : String(split(arm, '_')[end])
# ft names: <mode>_<input>[_<tokens>][_<objective>]_<level>; ft_pretrained_rtf_mlm_lvl2 -> configs/finetune/pretrained_rtf.toml, ckpt=@mlm_rtf;
# ft_pretrained_rtf_panel_lrecon_lvl2 -> pretrained_rtf_panel.toml, ckpt=@lrecon_rtf_panel (no objective: erecon)
ftparts = stage == "ft" ? String.(split(chopsuffix(arm, "_" * level), '_')) : String[]
hasobj = !isempty(ftparts) && ftparts[end] in ("mlm", "erecon", "lrecon")
objective = hasobj ? ftparts[end] : "erecon"
ftbase = hasobj ? ftparts[1:end-1] : ftparts
ftarm = join(ftbase, '_')
config = stage == "pt" ? "configs/pretrain/$arm.toml" :
         stage == "ft" ? "configs/finetune/$ftarm.toml" :
         stage == "bl" ? "configs/baselines/$arm.toml" : error("sweep name must start with pt_, ft_ or bl_")
stage == "pt" || level in ("lvl1", "lvl2") || error("sweep name must end in _lvl1 or _lvl2")
stage == "bl" && !occursin("mlp", arm) && !occursin("log", arm) && error("bl sweeps are for the mlp and logistic baselines")
isfile(joinpath(TahoeFM.PKG_DIR, config)) || error("no config $config")

# pretrained / random finetunes default to the checkpoint of the named objective (erecon if none) and the same input and token set
stage == "ft" && !startswith(ftarm, "scratch_") && get!(fixed, "ckpt", "@$(objective)_" * join(ftbase[2:end], '_'))

lr = TahoeFM.read_toml(joinpath(TahoeFM.PKG_DIR, config))["lr"]
vals = Dict("NAME" => name, "CONFIG" => config, "PROJECT" => project, "LEVEL" => level,
            "MAX_STEPS" => isempty(steps) ? (level == "lvl1" ? "10000" : "40000") : steps, "EVAL_EVERY" => level == "lvl1" ? "1000" : "500",
            "LR_GRID" => @sprintf("[%.3g, %.3g, %.3g]", lr / 3, lr, lr * 3),
            "FIXED" => join(["  $k:\n    value: \"$v\"\n" for (k, v) in sort(collect(fixed))]))
template = stage == "bl" && occursin("log", arm) ? "bl_log" : stage
yaml = replace(read(joinpath(TahoeFM.PKG_DIR, "configs", "sweeps", "$template.yaml"), String), ("\${$k}" => v for (k, v) in vals)...)
out = joinpath(TahoeFM.PKG_DIR, "logs", "sweeps", "$name.yaml")
mkpath(dirname(out)); write(out, yaml)
println(out)
dry || cd(() -> run(`wandb sweep $out`), TahoeFM.PKG_DIR)
