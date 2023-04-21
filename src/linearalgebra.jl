@inline dot(x::NTuple{N, T}, y::NTuple{N, T}) where {N, T <: Union{Float16, ComplexF16}} = dot(x, y, N < 128 ? Val(8) : Val(16))
@inline dot(x::NTuple{N, T}, y::NTuple{N, T}) where {N, T <: Union{Float32, ComplexF32}} = dot(x, y, N < 64 ? Val(4) : Val(8))
@inline dot(x::NTuple{N, T}, y::NTuple{N, T}) where {N, T <: Union{Float64, ComplexF64}} = dot(x, y, N < 64 ? Val(2) : Val(4))

@inline function dot(x::NTuple{N, T}, y::NTuple{N, T}, ::Val{M}) where {N, T, M}
    V = T <: Real ? Vec : ComplexVec
    unroll = M
    R = N % (2*unroll)

    s1 = V(ntuple(i -> zero(T), Val(M)))
    s2 = V(ntuple(i -> zero(T), Val(M)))

    for i in 1:2*unroll:N-R
        a = ntuple(k -> x[i + k - 1], Val(unroll))
        b = ntuple(k -> y[i + k - 1], Val(unroll))
        c = ntuple(k -> x[i + k - 1 + unroll], Val(unroll))
        d = ntuple(k -> y[i + k - 1 + unroll], Val(unroll))

        s1 = fmadd(conj(V(a)), V(b), s1)
        s2 = fmadd(conj(V(c)), V(d), s2)
    end
   
    s = fhadd(fadd(s1, s2))
    if !iszero(R)
        _x = ntuple(i -> x[N - R + i], Val(R))
        _y = ntuple(i -> y[N - R + i], Val(R))
        s3 = fmul(conj(V(_x)), V(_y))
        s += fhadd(s3)
    end
    return s
end
