# Implementations are based on the Block Interleaved format using a split data layout
# Vectors of reals and complex values are stored separately
# This format is more efficient for multiplies (as compared to interleaving vectors as (re, im, re, im))
# The approach is based on the algorithm described in [1]
# In general I've found this scheme best for multiplication and therefore better for muladd operations
# For straight addition and subtraction it is in general better for the interleaved format due to less shuffling

#
# [1] Popovici, Doru T., Franz Franchetti, and Tze Meng Low. "Mixed data layout kernels for vectorized complex arithmetic." 
#     2017 IEEE High Performance Extreme Computing Conference (HPEC). IEEE, 2017.
#

# complex multiply
@inline function fmul(x::ComplexVec{N, FloatTypes}, y::ComplexVec{N, FloatTypes}) where {N, FloatTypes}
    r = _mulsub(x.re, y.re, fmul(x.im, y.im))
    i = _muladd(x.re, y.im, fmul(x.im, y.re))
    return ComplexVec(r, i)
end

# complex add
@inline function fadd(x::ComplexVec{N, FloatTypes}, y::ComplexVec{N, FloatTypes}) where {N, FloatTypes}
    re = fadd(x.re, y.re)
    im = fadd(x.im, y.im)
    return ComplexVec(re, im)
end

# complex add
@inline function fsub(x::ComplexVec{N, FloatTypes}, y::ComplexVec{N, FloatTypes}) where {N, FloatTypes}
    re = fsub(x.re, y.re)
    im = fsub(x.im, y.im)
    return ComplexVec{N, FloatTypes}((re, im))
end

# complex multiply-add
@inline muladd(x::ComplexVec{N, FloatTypes}, y::ComplexVec{N, FloatTypes}, z::ComplexVec{N, FloatTypes}) where {N, FloatTypes} = fadd(fmul(x, y), z)

# complex multiply-subtract
@inline mulsub(x::ComplexVec{N, FloatTypes}, y::ComplexVec{N, FloatTypes}, z::ComplexVec{N, FloatTypes}) where {N, FloatTypes} = fsub(fmul(x, y), z)
