#
# Polynomial evaluation using Horner's scheme.
#
# File contains methods to compute several different polynomials at once using SIMD with Horner's scheme (horner_simd).
# Additional methods to evaluate a single polynomial using SIMD with different order Horner's scheme (horner2, horner4, horner8).
# TODO: Support complex numbers

#
# Parallel Horner's scheme to evaluate 2, 4, or 8 polynomials in parallel using SIMD.
#
# Usage:
# x = 1.1
# P1 = (1.1, 1.2, 1.4, 1.5)
# P2 = (1.3, 1.3, 1.6, 1.8)
# a = horner_simd(x, pack_poly((P1, P2)))
# a[1].value == evalpoly(x, P1)
# a[2].value == evalpoly(x, P2)
#
# Note the strict equality as this method doesn't alter the order of individual polynomial evaluations
#

@inline pack_poly(P::Tuple{Vararg{NTuple{M, T}, N}}) where {N, M, T} = ntuple(i -> Vec{N, T}((ntuple(j -> P[j][i], Val(N)))), Val(M))

# cast single element `x` to a width of the number of polynomials
@inline horner_simd(x::Union{T, VE{T}}, p::NTuple{N, Vec{M, T}}) where {N, M, T} = horner_simd(constantvector(x, Vec{M, T}), p)

@inline function horner_simd(x::Vec{M, T}, p::NTuple{N, Vec{M, T}}) where {N, M, T <: FloatTypes}
    a = p[end]
    for i in N-1:-1:1
        a = muladd(x, a, p[i])
    end
    return a
end

# Clenshaw recurrence scheme to evaluate Chebyshev polynomials
# Assumes arguments x is prescaled
@inline function clenshaw_simd(x::T, c::NTuple{N, Vec{M, T}}) where {N, M, T <: FloatTypes}
    x2 = constantvector(2*x, Vec{M, T})
    c0 = c[end-1]
    c1 = c[end]
    for i in length(c)-2:-1:1
        c0, c1 = fsub(c[i], c1), muladd(x2, c1, c0)
    end
    return muladd(x, c1, c0)
end

#
# Horner scheme for evaluating single polynomials
#
# In some cases we are interested in speeding up the evaluation of a single polynomial of high degree.
# It is possible to split the polynomial into pieces and use SIMD to convert each piece simultaneously
#
# Usage:
# x = 1.1
# poly = (1.0, 0.5, 0.1, 0.05, 0.1)
# evalpoly(x, poly) ≈ horner2(x, pack_horner2(poly)) ≈ horner4(x, pack_horner4(poly)) ≈ horner8(x, pack_horner8(poly))
#
# Note the approximative relation between each as floating point arithmetic is associative.
# Here, we are adding up the polynomial in different degrees so there will be a few ULPs difference
# 

# horner - default regular horner to base evalpoly
# horner2 - 2nd horner scheme that splits polynomial into even/odd powers
# horner4 - 4th order horner scheme that splits into a + ex^4 + ... & b + fx^5 ... & etc
# horner8 - 8th order horner scheme that splits into a + ix^8 + .... & etc

@inline horner(x::T, P::NTuple{N, T}) where {N, T <: FloatTypes} = evalpoly(x, P)
@inline horner2(x, P::NTuple{N, T}) where {N, T <: FloatTypes} = horner2(x, pack_horner2(P))
@inline horner4(x, P::NTuple{N, T}) where {N, T <: FloatTypes} = horner4(x, pack_horner4(P))
@inline horner8(x, P::NTuple{N, T}) where {N, T <: FloatTypes} = horner8(x, pack_horner8(P))

@inline function horner2(x, P::NTuple{N, Vec{2, T}}) where {N, T <: FloatTypes}
    a = horner_simd(x * x, P)
    return muladd(x, a.data[2].value, a.data[1].value)
end

@inline function horner4(x, P::NTuple{N, Vec{4, T}}) where {N, T <: FloatTypes}
    xx = x * x
    a = horner_simd(xx * xx, P)
    b = muladd(x, Vec((a.data[4], a.data[2])), Vec((a.data[3], a.data[1])))
    return muladd(xx, b.data[1].value, b.data[2].value) 
end

@inline function horner8(x, P::NTuple{N, Vec{8, T}}) where {N, T <: FloatTypes}
    x2 = x * x
    x4 = x2 * x2
    a = horner_simd(x4 * x4, P)

    # following computes
    # a[1].value + a[2].value*x + a[3].value*x^2 + a[4].value*x^3 + a[5].value*x^4 + a[6].value*x^5 + a[7].value*x^6 + a[8].value*x^7

    b = muladd(x, Vec((a.data[4], a.data[2], a.data[8], a.data[6])), Vec((a.data[3], a.data[1], a.data[7], a.data[5])))
    c = muladd(x2, Vec((b.data[1], b.data[3])), Vec((b.data[2], b.data[4])))
    return muladd(x4, c.data[2].value, c.data[1].value)
end

#
# Packs coefficients for arbitrary order horner evaluation.
# For example second order Horner packs a polynomial (0.1, 0.2, 0.3, 0.4) into (Vec{2, T}((0.1, 0.2)), Vec{2, T}((0.3, 0.4)))
# A similar scheme is used for larger degrees with tupel elngth equal to the Val type.
# Only suppored for second, fourth, and eight order horner evaluations, so Val should be restricted to 2, 4, 8

#
# Usage:
# x = 1.1
# poly = (1.0, 0.5, 0.1, 0.05, 0.1)
# evalpoly(x, poly) ≈ horner2(x, pack_horner(poly, Val(2))) ≈ horner4(x, pack_horner(poly, Val(4))) ≈ horner8(x, pack_horner(poly, Val(8)))
#
# Note: these functions will pack statically known polynomials for N <= 32 at compile time.
# If length(poly) > 32 they must be packed and declared as constants otherwise packing will allocate at runtime.
# Though this isn't consistent and should be investigated further.
#
# Therefore, it is always recommended (if you know the polynomial coefficients ahead of time) to pre-pack as constants
# const P = pack_horner(ntuple(i -> i*0.1, 35), Val(2))
# horner2(x, P)
#
#

@inline function pack_horner(p::NTuple{N, T}, ::Val{M}) where {N, T <: FloatTypes, M}
    rem = N % M
    pad = !iszero(rem) ? (M - rem) : 0
    P = (p..., ntuple(i -> zero(T), Val(pad))...)
    return ntuple(i -> Vec(ntuple(k -> P[M*i - (M - k)], Val(M))), Val((N + pad) ÷ M))
end
