# encoder shared by ETF and RTF: token = value channel (ETF Dense(1 => D), RTF rank table) + gene embedding; a mask vector replaces a hidden channel; panel or topk tokens

struct Encoder{V,E,M<:AbstractVector,O<:Flux.Dropout,T<:Flux.Chain,F}
    value::V
    gene::E
    mask::M
    dropout::O
    layers::T
    final::F     # LayerNorm on the last layer output (norm = pre_final), else nothing
end
Flux.@layer Encoder

# n_genes = gene table rows (panel G, topk all genes); n_ranks = RTF rank table rows (panel G, topk K)
function Encoder(input::Input; n_genes::Int, n_ranks::Int = n_genes, embed_dim::Int = 128, hidden_dim::Int = 128, n_heads::Int = 2,
                 n_layers::Int = 2, drop_prob::Float64 = 0.1, norm::AbstractString = "pre")
    norm in ("pre", "pre_final", "post") || error("norm must be pre, pre_final or post, got $norm")
    value = input isa ETF ? Flux.Dense(1 => embed_dim) : Flux.Embedding(n_ranks => embed_dim)
    layers = Flux.Chain([Transf(embed_dim, hidden_dim; n_heads, dropout_prob=drop_prob, post=(norm == "post")) for _ in 1:n_layers]...)
    # mask scale as the original models: ETF mask_emb 0.02 randn, RTF spare embedding row randn
    mask = input isa ETF ? randn(Float32, embed_dim) .* 0.02f0 : randn(Float32, embed_dim)
    return Encoder(value, Flux.Embedding(n_genes => embed_dim), mask, Flux.Dropout(drop_prob), layers,
                   norm == "pre_final" ? Flux.LayerNorm(embed_dim) : nothing)
end

embed_dim(e::Encoder) = length(e.mask)
n_genes(e::Encoder) = size(e.gene.weight, 2)

_gids(x) = Flux.ChainRulesCore.ignore_derivatives(() -> x isa CuArray ? cu(Int32.(1:size(x, 1))) : Int32.(1:size(x, 1)))
_hide3(x, hide) = Flux.ChainRulesCore.ignore_derivatives(() -> reshape(x isa CuArray ? CuArray(hide) : Array(hide), 1, size(hide)...))

value_channel(e::Encoder, x::AbstractMatrix{<:AbstractFloat}) = e.value(reshape(x, 1, size(x)...))
value_channel(e::Encoder, r::AbstractMatrix{<:Integer}) = e.value(r)
gene_channel(e::Encoder, x) = e.gene(_gids(x))
# topk tokens: gene channel from each token's gene id, value channel from its value / rank position
value_channel(e::Encoder, x::TopkInput) = value_channel(e, x.v)
gene_channel(e::Encoder, x::TopkInput) = e.gene(x.ids)
_hide3(x::TopkInput, hide) = _hide3(x.ids, hide)

tokens(e::Encoder, x) = value_channel(e, x) .+ gene_channel(e, x)
# hide = Bool G x B (true = hidden); :value hides the value and keeps the gene, :gene hides the gene and keeps the value
tokens(e::Encoder, x, hide::AbstractMatrix{Bool}, what::Symbol = :value) = what == :value ?
    ifelse.(_hide3(x, hide), e.mask, value_channel(e, x)) .+ gene_channel(e, x) :
    value_channel(e, x) .+ ifelse.(_hide3(x, hide), e.mask, gene_channel(e, x))

# dropout + blocks (+ final LayerNorm); returns all block outputs when states = true (last = encoder output); mask = attention mask of padded tokens
function run_layers(e::Encoder, h0; states::Bool = false, mask = nothing)
    h = e.dropout(h0)
    hs = states ? [h0] : nothing
    for l in e.layers.layers
        h = l(h, mask)
        states && push!(hs, h)
    end
    if !isnothing(e.final)
        h = e.final(h)
        states && (hs[end] = h)
    end
    return states ? hs : h
end

encode(e::Encoder, x) = run_layers(e, tokens(e, x); mask = attn_mask(x))
encode(e::Encoder, x, hide, what::Symbol = :value) = run_layers(e, tokens(e, x, hide, what); mask = attn_mask(x))
# every block output (h0..hL) of the unmasked input
token_layers(e::Encoder, x) = run_layers(e, tokens(e, x); states = true, mask = attn_mask(x))
# mean-pooled encoder output (real tokens only)
pooled(e::Encoder, x) = meanpool(encode(e, x), pool_weights(x))

# inits (cpu, before moving to the gpu); dedicated rngs so ETF and RTF get the same vectors at the same seed

# mask vector 0.02 randn
init_mask!(e::Encoder; seed::Int = 1) = (e.mask .= randn(MersenneTwister(seed + 2001), Float32, length(e.mask)) .* 0.02f0; e)

# RTF rank table linear in rank: table[:, r] = w * z_r, z_r = -(r - (G+1)/2) / sd(1:G), rank 1 largest
function rank_init_lin!(e::Encoder; seed::Int = 1)
    D, G = size(e.value.weight)
    w = vec(Flux.glorot_uniform(MersenneTwister(seed + 2002), D, 1))
    e.value.weight .= w * Float32.(-((1:G) .- (G + 1) / 2) ./ std(1:G))'
    return e
end

# tok_norm: gene table and mask at norm ~1, ETF value projection acts on globally standardised x
function tok_norm!(e::Encoder; x_mean::Real = 0, x_sd::Real = 1)
    e.gene.weight .*= 1f0 / sqrt(Float32(size(e.gene.weight, 1)))
    e.mask ./= sqrt(sum(abs2, e.mask))
    if e.value isa Flux.Dense
        e.value.weight ./= Float32(x_sd)
        e.value.bias .-= vec(e.value.weight) .* Float32(x_mean)
    end
    return e
end

# init schemes: :default (original models) or :tok_norm (erecon, lrecon)
function init_encoder!(e::Encoder, scheme::Symbol; seed::Int = 1, x_mean::Real = 0, x_sd::Real = 1)
    scheme == :default && return e
    scheme == :tok_norm || error("unknown init scheme $scheme")
    init_mask!(e; seed)
    e.value isa Flux.Embedding && rank_init_lin!(e; seed)
    return tok_norm!(e; x_mean, x_sd)
end

# mean column norms of the gene and value channels and the mask (init health)
function channel_norms(e::Encoder, x)
    cn(A) = mean(sqrt.(vec(sum(abs2, reshape(Array(A), size(A, 1), :), dims=1))))
    return (gene = cn(gene_channel(e, x)), value = cn(value_channel(e, x)), mask = sqrt(sum(abs2, Array(e.mask))))
end
