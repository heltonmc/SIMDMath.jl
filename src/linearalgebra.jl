function dot(x::NTuple{N, Complex{T}}, y::NTuple{N, Complex{T}}) where {N, T}
    R = N % 4

    if R == 0
        _x = constantvector(zero(Complex{T}), ComplexVec{4, T})
        _y = constantvector(zero(Complex{T}), ComplexVec{4, T})
    elseif R == 1
        _x = ComplexVec((x[1], zero(Complex{T}), zero(Complex{T}), zero(Complex{T})))
        _y = ComplexVec((y[1], zero(Complex{T}), zero(Complex{T}), zero(Complex{T})))
    elseif R == 2
        _x = ComplexVec((x[1], x[2], zero(Complex{T}), zero(Complex{T})))
        _y = ComplexVec((y[1], y[2], zero(Complex{T}), zero(Complex{T})))
    elseif R == 3
        _x = ComplexVec((x[1], x[2], x[3], zero(Complex{T})))
        _y = ComplexVec((y[1], y[2], y[3], zero(Complex{T})))
    end

    out = fmul(conj(_x), _y)

    for i in R+1:4:length(x)
        a = ComplexVec((x[i], x[i+1], x[i+2], x[i+3]))
        b = ComplexVec((y[i], y[i+1], y[i+2], y[i+3]))
        out = fmadd(conj(a), b, out)
    end
    return fhadd(out)
end

function dot(x::NTuple{N, T}, y::NTuple{N, T}) where {N, T}
    unroll = 32
    R = N % unroll

    _x = Vec((ntuple(i -> x[i], Val(R))..., ntuple(i -> zero(T), Val(unroll-R))...))
    _y = Vec((ntuple(i -> y[i], Val(R))..., ntuple(i -> zero(T), Val(unroll-R))...))
  
    out = fmul(_x, _y)

    for i in R+1:unroll:length(x)
        a = Vec(ntuple(k -> x[i + k - 1], Val(unroll)))
        b = Vec(ntuple(k -> y[i + k - 1], Val(unroll)))
        out = fmadd(a, b, out)
    end
    return fhadd(out)
end
