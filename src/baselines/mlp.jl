# relu MLP baselines (emlp / rmlp), as the old emlp.jl / rmlp.jl

# layer sizes: mlp_hidden_dim = 0 -> tapered G -> C over n_layers hidden layers; else n_layers hidden layers of width h (const) or halving (funnel, floor C)
function mlp_sizes(G::Integer, C::Integer, n_layers::Integer, h::Integer; shape::AbstractString = "const")
    h == 0 && return [round(Int, G + (C - G) * i / (n_layers + 1)) for i in 0:n_layers+1]
    hidden = shape == "funnel" ? [max(h ÷ 2^(i-1), C) for i in 1:n_layers] : fill(h, n_layers)
    return [G; hidden; C]
end

# Dense(relu) + Dropout per hidden layer, linear output, no input dropout
function mlp_model(sizes::Vector{Int}, drop::Real)
    layers = Any[]
    for i in 1:length(sizes)-1
        last = i == length(sizes) - 1
        push!(layers, Flux.Dense(sizes[i] => sizes[i+1], last ? identity : relu))
        last || push!(layers, Flux.Dropout(Float64(drop)))
    end
    return Flux.Chain(layers...)
end
