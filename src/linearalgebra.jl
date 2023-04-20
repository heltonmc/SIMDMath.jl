@inline dot(x::NTuple{N, T}, y::NTuple{N, T}) where {N, T <: Union{Float16, ComplexF16}} = dot(x, y, Val(8))
@inline dot(x::NTuple{N, T}, y::NTuple{N, T}) where {N, T} = dot(x, y, Val(4))

@inline function dot(x::NTuple{N, T}, y::NTuple{N, T}, ::Val{M}) where {N, T, M}
    V = T <: Real ? Vec : ComplexVec
    unroll = M
    R = N % unroll

    _x = (ntuple(i -> x[i], Val(R))..., ntuple(i -> zero(T), Val(unroll-R))...)
    _y = (ntuple(i -> y[i], Val(R))..., ntuple(i -> zero(T), Val(unroll-R))...)
  
    out = fmul((V(_x)), V(_y))

    for i in R+1:unroll:length(x)
        a = ntuple(k -> x[i + k - 1], Val(unroll))
        b = ntuple(k -> y[i + k - 1], Val(unroll))
        out = fmadd((V(a)), V(b), out)
    end
    return fhadd(out)
end
