# Implementations are based on the Block Interleaved format using a split data layout ((re, re), (im, im))
# where real and imaginary components stored separately
# This format appears to be best for SIMD multiply (compared to interleaving vectors as (re, im, re, im))
# Therefore, it may be better for muladd operations. For add/subtract it is probalby better for interleaved format
# due to less shuffling. Though, this format is based on the fact that the vector packing can be done at compile time
# The approach is based on the algorithm described in [1]
#
# [1] Popovici, Doru T., Franz Franchetti, and Tze Meng Low. "Mixed data layout kernels for vectorized complex arithmetic." 
#     2017 IEEE High Performance Extreme Computing Conference (HPEC). IEEE, 2017.
#

# complex multiply
@inline function fmul(x::ComplexVec{N, FloatTypes}, y::ComplexVec{N, FloatTypes}) where {N, FloatTypes}
    r = mulsub(x.re, y.re, fmul(x.im, y.im))
    i = muladd(x.re, y.im, fmul(x.im, y.re))
    return ComplexVec(r, i)
end

@inline function fmul(x::ComplexVec{N, FloatTypes}, y::Vec{N, FloatTypes}) where {N, FloatTypes}
    r = fmul(x.re, y.data)
    i = fmul(x.im, y.data)
    return ComplexVec(r, i)
end

# Complex add / subtract
for f in (:fadd, :fsub)
    @eval begin
        @inline function $f(x::ComplexVec{N, FloatTypes}, y::ComplexVec{N, FloatTypes}) where {N, FloatTypes}
            re = $f(x.re, y.re)
            im = $f(x.im, y.im)
            return ComplexVec(re, im)
        end
        @inline function $f(x::ComplexVec{N, FloatTypes}, y::Vec{N, FloatTypes}) where {N, FloatTypes}
            re = $f(x.re, y.data)
            return ComplexVec(re, x.im)
        end
    end
end

# Argument symmetry
for f in (:fmul, :fadd, :fsub)
    @eval @inline $f(x::Vec{N, T}, y::ComplexVec{N, T}) where {N, T <: FloatTypes} = $f(y, x)
end

# complex multiply-add
@inline muladd(x::ComplexorRealVec{N, T}, y::ComplexorRealVec{N, T}, z::ComplexorRealVec{N, T}) where {N, T <: FloatTypes} = fadd(fmul(x, y), z)

# complex multiply-subtract
@inline mulsub(x::ComplexorRealVec{N, T}, y::ComplexorRealVec{N, T}, z::ComplexorRealVec{N, T}) where {N, T <: FloatTypes} = fsub(fmul(x, y), z)
