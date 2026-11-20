# plate-position control: "project" (default) removes the span of the train plate-position means from every column; "subtract" is I_split.md's per-(position, block) subtraction (injects a class code, see DECISIONS.md)

# pos_correct config value -> method: false / 0 -> nothing, true -> "project", "project" | "subtract"
function pos_method(v)
    (v === false || v == 0) && return nothing
    v === true && return "project"
    v in ("project", "subtract") || error("pos_correct must be false, true, \"project\" or \"subtract\", got $v")
    return String(v)
end

# residuals of F centred per (plate, cell line) over the train columns tr, averaged per train well of blocks 1-4: (W features x wells, well positions, well blocks); cols = sample indices of F's columns
function well_residuals(F::AbstractMatrix, meta::NamedTuple, cols::AbstractVector{<:Integer}, tr::AbstractVector{<:Integer})
    p = size(F, 1); key = [meta.plate[c] * "|" * meta.cell_line[c] for c in cols]
    km = Dict{String,Vector{Float64}}(); kn = Dict{String,Int}()
    for j in tr
        v = get!(() -> zeros(p), km, key[j]); v .+= view(F, :, j); kn[key[j]] = get(kn, key[j], 0) + 1
    end
    for (k, v) in km; v ./= kn[k]; end
    wells = Dict{String,Vector{Int}}()
    for j in tr; meta.block[cols[j]] > 0 && push!(get!(wells, meta.well[cols[j]], Int[]), j); end
    wl = sort(collect(keys(wells))); W = zeros(p, length(wl))
    for (i, w) in enumerate(wl)
        for j in wells[w]; W[:, i] .+= view(F, :, j) .- km[key[j]]; end
        W[:, i] ./= length(wells[w])
    end
    return W, [meta.pos[cols[wells[w][1]]] for w in wl], [meta.block[cols[wells[w][1]]] for w in wl]
end

# position effect P (features x 96 positions x blocks 0-4): for each (position, block) the median train-well residual at that position in the other blocks (>= min_wells wells, else 0)
function position_effect(F::AbstractMatrix, meta::NamedTuple, cols::AbstractVector{<:Integer}, tr::AbstractVector{<:Integer}; min_wells::Int = 3)
    W, wq, wb = well_residuals(F, meta, cols, tr)
    P = zeros(Float32, size(F, 1), 96, 5)
    for b in 0:4, q in 0:95
        k = findall((wq .== q) .& (wb .!= b))
        length(k) >= min_wells && (P[:, q+1, b+1] .= vec(median(W[:, k], dims=2)))
    end
    return P
end

# orthonormal basis (features x r) of the train-well residual means per plate position, centred over positions (r <= positions with a train well - 1)
function position_basis(F::AbstractMatrix, meta::NamedTuple, cols::AbstractVector{<:Integer}, tr::AbstractVector{<:Integer}; rtol::Float64 = 1e-6)
    W, wq, _ = well_residuals(F, meta, cols, tr)
    Q = reduce(hcat, [vec(mean(W[:, wq .== q], dims=2)) for q in 0:95 if any(wq .== q)]); Q .-= mean(Q, dims=2)
    s = svd(Q); r = count(>(rtol * max(s.S[1], eps())), s.S)
    return s.U[:, 1:r]
end

# F corrected for plate position, estimated on the train columns tr; project: the same linear map for every column (no metadata at application),
# subtract: F minus the effect at each column's own (position, block), a vector that is constant within a drug and differs between drugs
function pos_correct(F::AbstractMatrix, meta::NamedTuple, cols::AbstractVector{<:Integer}, tr::AbstractVector{<:Integer}; method::AbstractString = "project", min_wells::Int = 3)
    if method == "project"
        U = Float32.(position_basis(F, meta, cols, tr)); Ff = Matrix{Float32}(F)
        return Ff .- U * (U' * Ff)
    end
    method == "subtract" || error("position correction method must be project or subtract, got $method")
    P = position_effect(F, meta, cols, tr; min_wells)
    Y = similar(F, Float32)
    for (j, c) in enumerate(cols); Y[:, j] .= view(F, :, j) .- view(P, :, meta.pos[c] + 1, meta.block[c] + 1); end
    return Y
end
