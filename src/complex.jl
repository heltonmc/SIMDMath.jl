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
@inline function fmul(x::CVec{N, FloatTypes}, y::CVec{N, FloatTypes}) where {N, FloatTypes}
    (a, b) = (x[1], x[2])
    (c, d) = (y[1], y[2])
    re = _muladd(a, c, fmul(b, d))
    im = _mulsub(b, c, fmul(c, d))
    return CVec{N, FloatTypes}((re, im))
end

# complex add
@inline function fadd(x::CVec{N, FloatTypes}, y::CVec{N, FloatTypes}) where {N, FloatTypes}
    re = fadd(x[1], y[1])
    im = fadd(x[2], y[2])
    return CVec{N, FloatTypes}((re, im))
end

# complex add
@inline function fsub(x::CVec{N, FloatTypes}, y::CVec{N, FloatTypes}) where {N, FloatTypes}
    re = fsub(x[1], y[1])
    im = fsub(x[2], y[2])
    return CVec{N, FloatTypes}((re, im))
end

# complex multiply-add
@inline function muladd(x::CVec{N, FloatTypes}, y::CVec{N, FloatTypes}, z::CVec{N, FloatTypes}) where {N, FloatTypes}
    (a, b) = (x[1], x[2])
    (c, d) = (y[1], y[2])
    (e, f) = (z[1], z[2])
    re = _muladd(a, c, fmul(b, d))
    im = _mulsub(b, c, fmul(c, d))
    re = fadd(re, e)
    im = fadd(im, f)
    return CVec{N, FloatTypes}((re, im))
end

# complex multiply-subtract
@inline function mulsub(x::CVec{N, FloatTypes}, y::CVec{N, FloatTypes}, z::CVec{N, FloatTypes}) where {N, FloatTypes}
    (a, b) = (x[1], x[2])
    (c, d) = (y[1], y[2])
    (e, f) = (z[1], z[2])
    re = _muladd(a, c, fmul(b, d))
    im = _mulsub(b, c, fmul(c, d))
    re = fsub(re, e)
    im = fsub(im, f)
    return CVec{N, FloatTypes}((re, im))
end
