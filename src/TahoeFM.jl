module TahoeFM

# Tahoe-100M pseudobulk and single-cell transformer experiments: expression (ETF) vs rank (RTF) input, pretraining objectives, finetuning, baselines

using CUDA, Flux, Functors, JLD2, LinearAlgebra, NNlib, Optimisers, Printf, Random, Statistics, StatsBase, TOML, Dates
import PyCall, Distributed

const PKG_DIR = dirname(@__DIR__)   # thesis/tahoe

include("models/fastattn.jl")
using .FastAttn: attend

include("data/data.jl")
include("data/prepare.jl")
include("inputs.jl")
include("data/cells.jl")
include("models/transformer.jl")
include("models/encoder.jl")
include("models/heads.jl")
include("models/classifier.jl")
include("objectives/objective.jl")
include("objectives/mlm.jl")
include("objectives/erecon.jl")
include("objectives/lrecon.jl")
include("train/common.jl")
include("train/checkpoint.jl")
include("train/diag.jl")
include("train/pretrain.jl")
include("train/classify.jl")
include("train/finetune.jl")
include("log.jl")
include("eval/eval.jl")
include("baselines/baselines.jl")

export TahoeData, load_data, panel_data, panel_matrix, all_matrix, topk_ids, topk_values, labels, pretrain_split, finetune_split
export CellSource, CellStream, cell_source, cell_data, cell_stream, next_batch!, read_cells
export Input, ETF, RTF, TopkInput, parse_input, gene_ranks, rank_features, token_set, cfg_data, raw_input, model_input
export Encoder, Classifier, MLM, ERecon, LRecon, objective_from
export pretrain, finetune, load_config, load_encoder
export baseline, lda_baseline, trivial_baselines
export run_probe, run_health, run_bootstrap, pos_correct

end
