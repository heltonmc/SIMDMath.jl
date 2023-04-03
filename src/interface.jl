# Create tuples of VecElements filled with a constant value of x

@inline constantvector(x::T, y) where T <: FloatTypes = constantvector(VE(x), y)
@inline constantvector(x::T, ::Type{Vec{N, T}}) where {N, T <: FloatTypes} = Vec{N, T}(constantvector(VE(x), LVec{N, T}))
@inline constantvector(z::Complex{T}, ::Type{ComplexVec{N, T}}) where {N, T <: FloatTypes} = ComplexVec{N, T}(constantvector(VE(z.re), LVec{N, T}), constantvector(VE(z.im), LVec{N, T}))

# Generic interface to fused operations for scalar and vector types

for f in (:fmadd, :fmsub, :fnmadd, :fnmsub)
    @eval begin
        @inline $f(x::Vec{N, T}, y::Vec{N, T}, z::Vec{N, T}) where {N, T <: FloatTypes} = Vec($f(x.data, y.data, z.data))
        @inline $f(x::ScalarTypes, y::Vec{N, T}, z::Vec{N, T}) where {N, T <: FloatTypes} = $f(constantvector(x, Vec{N, T}), y, z)
        @inline $f(x::Vec{N, T}, y::ScalarTypes, z::Vec{N, T}) where {N, T <: FloatTypes} = $f(x, constantvector(y, Vec{N, T}), z)
        @inline $f(x::ScalarTypes, y::ScalarTypes, z::Vec{N, T}) where {N, T <: FloatTypes} = $f(constantvector(x, Vec{N, T}), constantvector(y, Vec{N, T}), z)
        @inline $f(x::Vec{N, T}, y::Vec{N, T}, z::ScalarTypes) where {N, T <: FloatTypes} = $f(x, y, constantvector(z, Vec{N, T}))
        @inline $f(x::ScalarTypes, y::Vec{N, T}, z::ScalarTypes) where {N, T <: FloatTypes} = $f(constantvector(x, Vec{N, T}), y, constantvector(z, Vec{N, T}))
        @inline $f(x::Vec{N, T}, y::ScalarTypes, z::ScalarTypes) where {N, T <: FloatTypes} = $f(x, constantvector(y, Vec{N, T}), constantvector(z, Vec{N, T}))
    end
end

for f in (:fadd, :fsub, :fmul, :fdiv)
    @eval begin
        @inline $f(x::Vec{N, T}, y::Vec{N, T}) where {N, T <: FloatTypes} = Vec($f(x.data, y.data))
        @inline $f(x::Vec{N, T}, y::ScalarTypes) where {N, T <: FloatTypes} = $f(x, constantvector(y, Vec{N, T}))
        @inline $f(x::ScalarTypes, y::Vec{N, T}) where {N, T <: FloatTypes} = $f(constantvector(x, Vec{N, T}), y)
    end
end