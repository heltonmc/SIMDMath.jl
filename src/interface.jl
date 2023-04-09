# Create tuples of VecElements filled with a constant value of x

@inline constantvector(x::T, y) where T <: FloatTypes = constantvector(VE(x), y)
@inline constantvector(x::T, ::Type{Vec{N, T}}) where {N, T <: FloatTypes} = Vec{N, T}(constantvector(VE(x), LVec{N, T}))
@inline constantvector(z::Complex{T}, ::Type{ComplexVec{N, T}}) where {N, T <: FloatTypes} = ComplexVec{N, T}(constantvector(VE(z.re), LVec{N, T}), constantvector(VE(z.im), LVec{N, T}))

# Generic interface to fused operations for scalar and vector types

for f in (:fmadd, :fmsub, :fnmadd, :fnmsub)
    @eval begin
        @inline $f(x::Vec{N, T}, y::Vec{N, T}, z::Vec{N, T}) where {N, T <: FloatTypes} = Vec($f(x.data, y.data, z.data))
        @inline $f(x::ScalarOrVec{N, T}, y::ScalarOrVec{N, T}, z::ScalarOrVec{N, T}) where {N, T <: FloatTypes} = $f(promote(x, y, z)...)
    end
end

for f in (:fadd, :fsub, :fmul, :fdiv)
    @eval begin
        @inline $f(x::Vec{N, T}, y::Vec{N, T}) where {N, T <: FloatTypes} = Vec($f(x.data, y.data))
        @inline $f(x::ScalarOrVec{N, T}, y::ScalarOrVec{N, T}) where {N, T <: FloatTypes} = $f(promote(x, y)...)
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
