module FastAttn

# fused column softmax (one warp per column) for the attention scores + the same attention math as
# NNlib.dot_product_attention (Flux.MultiHeadAttention), so checkpoints and outputs are unchanged

using CUDA, Flux, NNlib, Random
using Flux: ChainRulesCore
using Flux.ChainRulesCore: NoTangent, ZeroTangent, unthunk

export fast_softmax, attend, USE_FAST_ATTN

# false -> Flux.MultiHeadAttention (reference path)
const USE_FAST_ATTN = Ref(true)

const WARP = Int32(32)

function _sm_fwd_kernel!(y, x, L::Int32, N::Int32)
    t = threadIdx().x - Int32(1)
    lane = t % WARP
    col = (blockIdx().x - Int32(1)) * (blockDim().x ÷ WARP) + t ÷ WARP
    if col < N
        base = Int64(col) * L
        m = -Inf32
        i = lane
        while i < L
            @inbounds m = max(m, x[base + i + 1])
            i += WARP
        end
        off = Int32(16)
        while off > Int32(0)
            m = max(m, CUDA.shfl_xor_sync(0xffffffff, m, off))
            off >>= Int32(1)
        end
        s = 0f0
        i = lane
        while i < L
            @inbounds s += exp(x[base + i + 1] - m)
            i += WARP
        end
        off = Int32(16)
        while off > Int32(0)
            s += CUDA.shfl_xor_sync(0xffffffff, s, off)
            off >>= Int32(1)
        end
        r = 1f0 / s
        i = lane
        while i < L
            @inbounds y[base + i + 1] = exp(x[base + i + 1] - m) * r
            i += WARP
        end
    end
    return nothing
end

# dx = y .* (dy .- sum(dy .* y, dims=1))
function _sm_bwd_kernel!(dx, dy, y, L::Int32, N::Int32)
    t = threadIdx().x - Int32(1)
    lane = t % WARP
    col = (blockIdx().x - Int32(1)) * (blockDim().x ÷ WARP) + t ÷ WARP
    if col < N
        base = Int64(col) * L
        s = 0f0
        i = lane
        while i < L
            @inbounds s += dy[base + i + 1] * y[base + i + 1]
            i += WARP
        end
        off = Int32(16)
        while off > Int32(0)
            s += CUDA.shfl_xor_sync(0xffffffff, s, off)
            off >>= Int32(1)
        end
        i = lane
        while i < L
            @inbounds dx[base + i + 1] = y[base + i + 1] * (dy[base + i + 1] - s)
            i += WARP
        end
    end
    return nothing
end

const THREADS = 256

function _launch!(kernel, out, args...)
    L = size(out, 1); N = length(out) ÷ L
    N == 0 && return out
    @cuda threads=THREADS blocks=cld(N, THREADS ÷ Int(WARP)) kernel(out, args..., Int32(L), Int32(N))
    return out
end

# softmax over dim 1
fast_softmax(x::AbstractArray) = NNlib.softmax(x; dims=1)
fast_softmax(x::CuArray{Float32}) = _launch!(_sm_fwd_kernel!, similar(x), x)

function ChainRulesCore.rrule(::typeof(fast_softmax), x::CuArray{Float32})
    y = fast_softmax(x)
    function fast_softmax_pullback(dy)
        g = unthunk(dy)
        g isa ZeroTangent && return (NoTangent(), ZeroTangent())
        g = g isa CuArray{Float32} ? g : CuArray{Float32}(g)
        return (NoTangent(), _launch!(_sm_bwd_kernel!, similar(y), g, y))
    end
    return y, fast_softmax_pullback
end

# attention core on (Lk, dh, nh, B) keys, (dh, Lq, nh, B) scaled queries, (dh, Lk, nh, B) values:
# o = v * dropout(softmax(mask(k'q))), keep = dropout keep mask (Bool, (Lk, Lq, nh*B)) or nothing
function attn_core(kt, qt, vt, mask, keep, scale)
    α = fast_softmax(NNlib.apply_attn_mask(NNlib.batched_mul(kt, qt), mask))
    isnothing(keep) || (α = α .* reshape(keep, size(α)) .* scale)
    return NNlib.batched_mul(vt, α)
end

# GPU forward on 3D views: S = k'q, mask, in-place softmax -> P, o = v * (P .* keep * scale)
function _attn_fwd(kt, qt, vt, mask, keep, scale)
    Lk, dh, nh, B = size(kt); Lq = size(qt, 2)
    k3 = reshape(kt, Lk, dh, nh * B); q3 = reshape(qt, dh, Lq, nh * B); v3 = reshape(vt, dh, Lk, nh * B)
    S = NNlib.batched_mul(k3, q3)                                   # (Lk, Lq, nh*B)
    if !isnothing(mask)
        S4 = reshape(S, Lk, Lq, nh, B)
        S4 .= ifelse.(mask, S4, typemin(Float32))
        CUDA.unsafe_free!(S4)                                       # drops the view's reference only
    end
    _launch!(_sm_fwd_kernel!, S, S)                                 # in place: S = P
    if isnothing(keep)
        o = NNlib.batched_mul(v3, S)
    else
        Pd = S .* keep .* Float32(scale)
        o = NNlib.batched_mul(v3, Pd)
        CUDA.unsafe_free!(Pd)
    end
    return o, S, k3, q3, v3
end

# GPU primal (no gradient, e.g. eval): frees the (L x L) scores at once
function attn_core(kt::CuArray{Float32,4}, qt::CuArray{Float32,4}, vt::CuArray{Float32,4}, mask, keep, scale)
    o, P, _, _, _ = _attn_fwd(kt, qt, vt, mask, keep, scale)
    CUDA.unsafe_free!(P)
    return reshape(o, size(vt, 1), size(qt, 2), size(kt, 3), size(kt, 4))
end

# GPU gradient: keeps only softmax output + Bool keep mask for the backward, frees every other (L x L) temporary at once
function ChainRulesCore.rrule(::typeof(attn_core), kt::CuArray{Float32,4}, qt::CuArray{Float32,4}, vt::CuArray{Float32,4},
                              mask, keep, scale)
    Lk, dh, nh, B = size(kt); Lq = size(qt, 2)
    o, P, k3, q3, v3 = _attn_fwd(kt, qt, vt, mask, keep, scale)
    used = Ref(false)
    function attn_core_pullback(do4)
        used[] && error("attn_core pullback called twice (its buffers are freed after the first call)")
        used[] = true
        g = unthunk(do4)
        if g isa ZeroTangent
            CUDA.unsafe_free!(P)
            return (NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), NoTangent(), NoTangent(), NoTangent())
        end
        dO = reshape(g isa CuArray{Float32} ? g : CuArray{Float32}(g), dh, Lq, nh * B)
        keep3 = keep   # flat (Lk, Lq, nh*B)
        # dV = dO * Pd'
        if isnothing(keep3)
            dv = NNlib.batched_mul(dO, NNlib.batched_transpose(P))
        else
            Pd = P .* keep3 .* Float32(scale)
            dv = NNlib.batched_mul(dO, NNlib.batched_transpose(Pd))
            CUDA.unsafe_free!(Pd)
        end
        # dP = (V' * dO) .* keep * scale, then softmax backward in place (masked positions -> 0, P = 0 there)
        dP = NNlib.batched_mul(NNlib.batched_transpose(v3), dO)    # (Lk, Lq, nh*B)
        isnothing(keep3) || (dP .= dP .* keep3 .* Float32(scale))
        _launch!(_sm_bwd_kernel!, dP, dP, P)
        CUDA.unsafe_free!(P)
        isnothing(keep) || CUDA.unsafe_free!(keep)
        dk = NNlib.batched_mul(dP, NNlib.batched_transpose(q3))     # (Lk, dh, nh*B)
        dq = NNlib.batched_mul(NNlib.batched_transpose(k3), dP)     # (dh, Lq, nh*B)
        CUDA.unsafe_free!(dP)
        return (NoTangent(), reshape(dk, Lk, dh, nh, B), reshape(dq, dh, Lq, nh, B), reshape(dv, dh, Lk, nh, B),
                NoTangent(), NoTangent(), NoTangent())
    end
    return reshape(o, dh, Lq, nh, B), attn_core_pullback
end

_drop_active(d::Flux.Dropout, x) = d.p > 0 && (isnothing(d.active) ? NNlib.within_gradient(x) : d.active)

# self-attention, same ops as Flux.MultiHeadAttention / NNlib.dot_product_attention (bias = nothing); returns y only
function attend(mha::Flux.MultiHeadAttention, x::AbstractArray{T,3}, mask) where T
    USE_FAST_ATTN[] || return mha(x, x, x; mask=mask)[1]
    q = NNlib.split_heads(mha.q_proj(x), mha.nheads)   # (dh, nh, L, B)
    k = NNlib.split_heads(mha.k_proj(x), mha.nheads)
    v = NNlib.split_heads(mha.v_proj(x), mha.nheads)
    kt = permutedims(k, (3, 1, 2, 4))
    qt = permutedims(q, (1, 3, 2, 4)) ./ √T(size(q, 1))
    vt = permutedims(v, (1, 3, 2, 4))
    d = mha.attn_drop
    keep = _drop_active(d, x) ? Flux.ChainRulesCore.ignore_derivatives(() -> _keep_mask(d, (size(kt, 1), size(qt, 2), size(kt, 3) * size(kt, 4)), x)) : nothing
    o = attn_core(kt, qt, vt, mask, keep, T(1 / (1 - d.p)))
    o = permutedims(o, (1, 3, 2, 4))
    return mha.out_proj(NNlib.join_heads(o))
end

# Bool keep mask with the dropout layer's rng (u > p, as Flux's dropout kernel)
function _keep_mask(d::Flux.Dropout, sz, x)
    u = rand!(d.rng, similar(x, Float32, sz))
    keep = u .> Float32(d.p)
    x isa CuArray && CUDA.unsafe_free!(u)
    return keep
end

end  # module FastAttn
