# Create tuples of VecElements filled with a constant value of x

@inline constantvector(x::T, y) where T <: FloatTypes = constantvector(VE(x), y)
@inline constantvector(x::T, ::Type{Vec{N, T}}) where {N, T <: FloatTypes} = Vec{N, T}(constantvector(VE(x), LVec{N, T}))
@inline constantvector(z::Complex{T}, ::Type{ComplexVec{N, T}}) where {N, T <: FloatTypes} = ComplexVec{N, T}(constantvector(VE(z.re), LVec{N, T}), constantvector(VE(z.im), LVec{N, T}))

# Generic interface to fused operations for scalar and vector types

for f in (:fmadd, :fmsub, :fnmadd, :fnmsub)
    @eval begin
        @inline $f(x::Vec{N, T}, y::Vec{N, T}, z::Vec{N, T}) where {N, T <: FloatTypes} = Vec($f(x.data, y.data, z.data))
        @inline $f(x::Vec{N, T}, y::Vec{N, T}, z::T) where {N, T <: FloatTypes} = $f(promote(x, y, z)...)
        @inline $f(x::Vec{N, T}, y::T, z::T) where {N, T <: FloatTypes} = $f(promote(x, y, z)...)
        @inline $f(x::T, y::Vec{N, T}, z::Vec{N, T}) where {N, T <: FloatTypes} = $f(promote(x, y, z)...)
        @inline $f(x::T, y::Vec{N, T}, z::T) where {N, T <: FloatTypes} = $f(promote(x, y, z)...)
        @inline $f(x::Vec{N, T}, y::T, z::Vec{N, T}) where {N, T <: FloatTypes} = $f(promote(x, y, z)...)
        @inline $f(x::T, y::T, z::Vec{N, T}) where {N, T <: FloatTypes} = $f(promote(x, y, z)...)
    end
end

for f in (:fadd, :fsub, :fmul, :fdiv)
    @eval begin
        @inline $f(x::Vec{N, T}, y::Vec{N, T}) where {N, T <: FloatTypes} = Vec($f(x.data, y.data))
        @inline $f(x::Vec{N, T}, y::T) where {N, T <: FloatTypes} = $f(promote(x, y)...)
        @inline $f(x::T, y::Vec{N, T}) where {N, T <: FloatTypes} = $f(promote(x, y)...)
    end
end

@inline Base.checkbounds(v::ComplexOrRealVec{N, T}, i::IntegerTypes) where {N, T} = (i < 1 || i > N) && Base.throw_boundserror(v, i)

Base.@propagate_inbounds function Base.getindex(v::Vec{N, T}, i::IntegerTypes) where {N, T}
    @boundscheck checkbounds(v, i)
    return extractelement(v.data, i-1)
end

Base.@propagate_inbounds function Base.getindex(v::ComplexVec{N, T}, i::IntegerTypes) where {N, T}
    @boundscheck checkbounds(v, i)
    return complex(extractelement(v.re, i-1), extractelement(v.im, i-1))
end

# horizontal reduction
@inline fhadd(z::Vec{1, FloatTypes}) where FloatTypes = z[1]
@inline fhmul(z::Vec{1, FloatTypes}) where FloatTypes = z[1]
@inline fhadd(z::Vec{2, FloatTypes}) where FloatTypes = z[1] + z[2]
@inline fhmul(z::Vec{2, FloatTypes}) where FloatTypes = z[1] * z[2]

@inline function fhadd(z::Vec{N, FloatTypes}) where {N, FloatTypes}
    if ispow2(N)
        a = Vec(shufflevector(z.data, Val(0:N÷2-1)))
        b = Vec(shufflevector(z.data, Val(N÷2:N-1)))
        c = fadd(a, b)
        return fhadd(c)
    else
        return reduce(+, ntuple(i -> z[i], Val(N)))
    end
end
@inline function fhmul(z::Vec{N, FloatTypes}) where {N, FloatTypes}
    if ispow2(N)
        a = Vec(shufflevector(z.data, Val(0:N÷2-1)))
        b = Vec(shufflevector(z.data, Val(N÷2:N-1)))
        c = fmul(a, b)
        return fhmul(c)
    else
        return reduce(*, ntuple(i -> z[i], Val(N)))
    end
end
