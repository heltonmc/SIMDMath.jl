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
    r = fmsub(x.re, y.re, fmul(x.im, y.im))
    i = fmadd(x.re, y.im, fmul(x.im, y.re))
    return ComplexVec(r, i)
end

@inline function fmul(x::ComplexVec{N, FloatTypes}, y::Vec{N, FloatTypes}) where {N, FloatTypes}
    r = fmul(x.re, y.data)
    i = fmul(x.im, y.data)
    return ComplexVec(r, i)
end
@inline fmul(x::Vec{N, FloatTypes}, y::ComplexVec{N, FloatTypes}) where {N, FloatTypes} = fmul(y, x)

# Complex add / subtract
for f in (:fadd, :fsub)
    @eval begin
        @inline function $f(x::ComplexVec{N, FloatTypes}, y::ComplexVec{N, FloatTypes}) where {N, FloatTypes}
            r = $f(x.re, y.re)
            i = $f(x.im, y.im)
            return ComplexVec(r, i)
        end
        @inline function $f(x::ComplexVec{N, FloatTypes}, y::Vec{N, FloatTypes}) where {N, FloatTypes}
            r = $f(x.re, y.data)
            return ComplexVec(r, x.im)
        end
    end
end

@inline fadd(x::Vec{N, FloatTypes}, y::ComplexVec{N, FloatTypes}) where {N, FloatTypes} = fadd(y, x)

@inline function fsub(x::Vec{N, FloatTypes}, y::ComplexVec{N, FloatTypes}) where {N, FloatTypes}
    r = fsub(x.data, y.re)
    return ComplexVec(r, fneg(y.im))
end

for f in (:fmul, :fadd, :fsub)
    # promote complex numbers to constant complex vectors
    @eval @inline $f(x::Complex{T}, y::ComplexVec{N, T}) where {N, T <: FloatTypes} = $f(promote(x, y)...)
    @eval @inline $f(x::ComplexVec{N, T}, y::Complex{T}) where {N, T <: FloatTypes} = $f(promote(x, y)...)
    @eval @inline $f(x::Complex{T}, y::Vec{N, T}) where {N, T <: FloatTypes} = $f(convert(ComplexVec{N, T}, x), y)
    @eval @inline $f(x::Vec{N, T}, y::Complex{T}) where {N, T <: FloatTypes} = $f(x, convert(ComplexVec{N, T}, y))

    # promote real numbers to constant real vectors
    @eval @inline $f(x::T, y::ComplexVec{N, T}) where {N, T <: FloatTypes} = $f(convert(Vec{N, T}, x), y)
    @eval @inline $f(x::ComplexVec{N, T}, y::T) where {N, T <: FloatTypes} = $f(x, convert(Vec{N, T}, y))
end

@inline fneg(x::ComplexVec{N, T}) where {N, T <: FloatTypes} = ComplexVec{N, T}(fneg(x.re), fneg(x.im))

# complex multiply-add
# a*b + c
@inline fmadd(x, y, z) = fadd(fmul(x, y), z)

# complex multiply-subtract
# a*b - c
@inline fmsub(x, y, z) = fsub(fmul(x, y), z)

# complex negated multiply-add
# -a*b + c
@inline fnmadd(x, y, z) = fsub(z, fmul(x, y))

# -a*b - c
@inline fnmsub(x, y, z) = fneg(fmadd(x, y, z))

# scalar fallbacks
@inline fmul(x::Union{T, Complex{T}}, y::Union{T, Complex{T}}) where T = x * y
@inline fadd(x::Union{T, Complex{T}}, y::Union{T, Complex{T}}) where T = x + y
@inline fsub(x::Union{T, Complex{T}}, y::Union{T, Complex{T}}) where T = x - y
@inline fneg(x::Union{T, Complex{T}}) where T = -x

# conjugate
@inline Base.conj(z::ComplexVec{N, FloatTypes}) where {N, FloatTypes} = ComplexVec{N, FloatTypes}(z.re, fneg(z.im))
@inline Base.conj(z::Vec{N, FloatTypes}) where {N, FloatTypes} = z

# complex horizontal reduction
@inline fhadd(z::ComplexVec{2, FloatTypes}) where FloatTypes = z[1] + z[2]
@inline fhmul(z::ComplexVec{2, FloatTypes}) where FloatTypes = z[1] * z[2]

@inline function fhadd(z::ComplexVec{N, FloatTypes}) where {N, FloatTypes}
    if ispow2(N)
        a = ComplexVec(shufflevector(z.re, Val(0:N÷2-1)), shufflevector(z.im, Val(0:N÷2-1)))
        b = ComplexVec(shufflevector(z.re, Val(N÷2:N-1)), shufflevector(z.im, Val(N÷2:N-1)))
        c = fadd(a, b)
        return fhadd(c)
    else
        return reduce(+, ntuple(i -> z[i], Val(N)))
    end
end

@inline function fhmul(z::ComplexVec{N, FloatTypes}) where {N, FloatTypes}
    if ispow2(N)
        a = ComplexVec(shufflevector(z.re, Val(0:N÷2-1)), shufflevector(z.im, Val(0:N÷2-1)))
        b = ComplexVec(shufflevector(z.re, Val(N÷2:N-1)), shufflevector(z.im, Val(N÷2:N-1)))
        c = fmul(a, b)
        return fhmul(c)
    else
        return reduce(*, ntuple(i -> z[i], Val(N)))
    end
end
