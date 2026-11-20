# shared readout helpers: shrinkage LDA (uniform priors, alpha picked on val), PCA, frozen token states

const LDA_ALPHAS = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 0.6, 0.9]

# per-class means (p x K) and counts
function lda_group_means(Z, g, K)
    p, n = size(Z); S = zeros(p, K); c = zeros(Int, K)
    @inbounds for j in 1:n
        k = g[j]; c[k] += 1
        for i in 1:p; S[i, k] += Z[i, j]; end
    end
    return S ./ max.(c', 1), c
end

# z-score with train-column stats, drop constant dims; Float64
function lda_zprep(F, tr)
    Ftr = Float64.(F[:, tr]); mu = mean(Ftr, dims=2); sd = std(Ftr, dims=2); k = vec(sd .> 1e-8 * maximum(sd))
    return (Float64.(F[k, :]) .- mu[k]) ./ sd[k]
end

# shrinkage LDA fit on train, alpha by val accuracy; classes absent from train are never predicted; returns (val_acc, test_acc, alpha, test_preds)
function shrink_lda(Ztr, ytr, Zva, yva, Zte, yte, K; alphas = LDA_ALPHAS)
    p, n = size(Ztr); M, cn = lda_group_means(Ztr, ytr, K); present = count(>(0), cn)
    Rw = Ztr .- M[:, ytr]
    F = eigen(Symmetric(Rw * Rw' ./ (n - present))); U, sv = F.vectors, max.(F.values, 0); tau = sum(sv) / p
    UM = U' * M; best = (-1.0, NaN, NaN, Int[])
    for a in alphas
        W = U * (UM ./ ((1 - a) .* sv .+ a * tau .+ 1e-12 * tau)); b = -0.5 .* vec(sum(M .* W, dims=1)); b[cn .== 0] .= -Inf
        prd(X) = vec(getindex.(argmax(W' * X .+ b, dims=1), 1))
        v = mean(prd(Zva) .== yva)
        if v > best[1]
            pt = prd(Zte); best = (v, mean(pt .== yte), a, pt)
        end
    end
    return best
end

# shrinkage LDA fit on train, predictions of Zte for every alpha (n_te x n_alpha), same maths as shrink_lda
function lda_alpha_preds(Ztr, ytr, Zte, K; alphas = LDA_ALPHAS)
    p, n = size(Ztr); M, cn = lda_group_means(Ztr, ytr, K); present = count(>(0), cn)
    Rw = Ztr .- M[:, ytr]
    F = eigen(Symmetric(Rw * Rw' ./ (n - present))); U, sv = F.vectors, max.(F.values, 0); tau = sum(sv) / p
    UM = U' * M; P = zeros(Int, size(Zte, 2), length(alphas))
    for (i, a) in enumerate(alphas)
        W = U * (UM ./ ((1 - a) .* sv .+ a * tau .+ 1e-12 * tau)); b = -0.5 .* vec(sum(M .* W, dims=1)); b[cn .== 0] .= -Inf
        P[:, i] = vec(getindex.(argmax(W' * Zte .+ b, dims=1), 1))
    end
    return P
end

# z-score on the train columns then shrinkage LDA; TR / VA / TE = column ranges of F
lda_readout(F, y, K, TR, VA, TE) = (Z = lda_zprep(F, TR); shrink_lda(Z[:, TR], y[TR], Z[:, VA], y[VA], Z[:, TE], y[TE], K))

# PCA projection (k comps, at most the feature count) fit on train columns tr, applied to all columns
function pca_project(F, tr, k)
    k = min(k, size(F, 1))
    mu = mean(Float64.(F[:, tr]), dims=2); C = Float64.(F[:, tr]) .- mu
    U = eigen(Symmetric(C * C')).vectors[:, end:-1:end-k+1]
    return Float32.(U' * (Float64.(F) .- mu))
end

# encoder on the device in test mode (dropout off)
eval_device(m) = Flux.testmode!(to_device(m))

# token states h0 (tokens), h1..hL on unmasked input
token_states(enc::Encoder, x) = token_layers(enc, x)

