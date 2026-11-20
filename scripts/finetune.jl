# finetune a transformer classifier from a config
using Pkg; Pkg.activate(joinpath(dirname(@__DIR__), "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
using TahoeFM
finetune(TahoeFM.cli_config(ARGS))
