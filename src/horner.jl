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
# evalpoly(x, poly) ≈ horner(x, pack_horner(poly, Val(2))) ≈ horner(x, pack_horner(poly, Val(4))) ≈ horner(x, pack_horner(poly, Val(8)))
#
# Note the approximative relation between each as floating point arithmetic is associative.
# Here, we are adding up the polynomial in different degrees so there will be a few ULPs difference
# 

# horner - default regular horner to base evalpoly
# horner2 - 2nd horner scheme that splits polynomial into even/odd powers
# horner4 - 4th order horner scheme that splits into a + ex^4 + ... & b + fx^5 ... & etc
# horner8 - 8th order horner scheme that splits into a + ix^8 + .... & etc

#
# Packs coefficients for arbitrary order horner evaluation.
# For example second order Horner packs a polynomial (0.1, 0.2, 0.3, 0.4) into (Vec{2, T}((0.1, 0.2)), Vec{2, T}((0.3, 0.4)))
# A similar scheme is used for larger degrees with tupel elngth equal to the Val type.
# Only suppored for second, fourth, eighth, sixteenth, and 32nd order horner evaluations, so Val should be restricted to 2, 4, 8


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

@inline horner(x, P::NTuple{N, T}) where {N, T <: FloatTypes} = evalpoly(x, P)

@inline function horner(x, P::NTuple{N, Vec{2, T}}) where {N, T <: FloatTypes}
    x2 = x * x
    p = horner_simd(x2, P)
    a0 = Vec(shufflevector(p.data, Val(0)))
    b0 = Vec(shufflevector(p.data, Val(1)))
    return muladd(x, b0, a0).data[1].value
end

@inline function horner(x, P::NTuple{N, Vec{4, T}}) where {N, T <: FloatTypes}
    x2 = x * x
    x4 = x2 * x2
    p = horner_simd(x4, P)
    a0 = Vec(shufflevector(p.data, Val(0:1)))
    b0 = Vec(shufflevector(p.data, Val(2:3)))
    p1 = horner_simd(x2, (a0, b0))
    a1 = Vec(shufflevector(p1.data, Val(0)))
    b1 = Vec(shufflevector(p1.data, Val(1)))
    return muladd(x, b1, a1).data[1].value
end

@inline function horner(x, P::NTuple{N, Vec{8, T}}) where {N, T <: FloatTypes}
    x2 = x * x
    x4 = x2 * x2
    x8 = x4 * x4

    p0 = horner_simd(x8, P)
    a0 = Vec(shufflevector(p0.data, Val(0:3)))
    b0 = Vec(shufflevector(p0.data, Val(4:7)))

    p1 = horner_simd(x4, (a0, b0))
    a1 = Vec(shufflevector(p1.data, Val(0:1)))
    b1 = Vec(shufflevector(p1.data, Val(2:3)))

    p2 = horner_simd(x2, (a1, b1))
    a2 = Vec(shufflevector(p2.data, Val(0)))
    b2 = Vec(shufflevector(p2.data, Val(1)))

    return muladd(x, b2, a2).data[1].value
end

@inline function horner(x, P::NTuple{N, Vec{16, T}}) where {N, T <: FloatTypes}
    x2 = x * x
    x4 = x2 * x2
    x8 = x4 * x4
    x16 = x8 * x8

    p0 = horner_simd(x16, P)
    a0 = Vec(shufflevector(p0.data, Val(0:7)))
    b0 = Vec(shufflevector(p0.data, Val(8:15)))

    p1 = horner_simd(x8, (a0, b0))
    a1 = Vec(shufflevector(p1.data, Val(0:3)))
    b1 = Vec(shufflevector(p1.data, Val(4:7)))

    p2 = horner_simd(x4, (a1, b1))
    a2 = Vec(shufflevector(p2.data, Val(0:1)))
    b2 = Vec(shufflevector(p2.data, Val(2:3)))

    p3 = horner_simd(x2, (a2, b2))
    a3 = Vec(shufflevector(p3.data, Val(0)))
    b3 = Vec(shufflevector(p3.data, Val(1)))

    return muladd(x, b3, a3).data[1].value
end

@inline function horner(x, P::NTuple{N, Vec{32, T}}) where {N, T <: FloatTypes}
    x2 = x * x
    x4 = x2 * x2
    x8 = x4 * x4
    x16 = x8 * x8
    x32 = x16 * x16

    p0 = horner_simd(x32, P)
    a0 = Vec(shufflevector(p0.data, Val(0:15)))
    b0 = Vec(shufflevector(p0.data, Val(16:31)))

    p1 = horner_simd(x16, (a0, b0))
    a1 = Vec(shufflevector(p1.data, Val(0:7)))
    b1 = Vec(shufflevector(p1.data, Val(8:15)))

    p2 = horner_simd(x8, (a1, b1))
    a2 = Vec(shufflevector(p2.data, Val(0:3)))
    b2 = Vec(shufflevector(p2.data, Val(4:7)))

    p3 = horner_simd(x4, (a2, b2))
    a3 = Vec(shufflevector(p3.data, Val(0:1)))
    b3 = Vec(shufflevector(p3.data, Val(2:3)))

    p4 = horner_simd(x2, (a3, b3))
    a4 = Vec(shufflevector(p4.data, Val(0)))
    b4 = Vec(shufflevector(p4.data, Val(1)))

    return muladd(x, b4, a4).data[1].value
end
