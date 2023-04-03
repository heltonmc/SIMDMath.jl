using SIMDMath
using Test

@time @testset "SIMDMath.jl" begin

# Test horner_simd
let
    NT = 12
    P = (
                ntuple(n -> rand()*(-1)^n / n, NT),
                ntuple(n -> rand()*(-1)^n / n, NT),
                ntuple(n -> rand()*(-1)^n / n, NT),
                ntuple(n -> rand()*(-1)^n / n, NT)
            )
    
    x = 0.9
    a = horner_simd(x, pack_poly(P))
    @test evalpoly(x, P[1]) == a.data[1].value
    @test evalpoly(x, P[2]) == a.data[2].value
    @test evalpoly(x, P[3]) == a.data[3].value
    @test evalpoly(x, P[4]) == a.data[4].value

    NT = 24
    P32 = (
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
            )
    
    x = 1.2f0
    a = horner_simd(x, pack_poly(P32))
    @test evalpoly(x, P32[1]) == a.data[1].value
    @test evalpoly(x, P32[2]) == a.data[2].value
    @test evalpoly(x, P32[3]) == a.data[3].value
    @test evalpoly(x, P32[4]) == a.data[4].value
    @test evalpoly(x, P32[5]) == a.data[5].value
    @test evalpoly(x, P32[6]) == a.data[6].value
    @test evalpoly(x, P32[7]) == a.data[7].value
    @test evalpoly(x, P32[8]) == a.data[8].value

    NT = 4
    P16 = (
                ntuple(n -> Float16(rand()*(-1)^n / n), NT),
                ntuple(n -> Float16(rand()*(-1)^n / n), NT),
                ntuple(n -> Float16(rand()*(-1)^n / n), NT),
            )
    
    x = Float16(0.8)
    a = horner_simd(x, pack_poly(P16))
    @test evalpoly(x, P16[1]) ≈ a.data[1].value
    @test evalpoly(x, P16[2]) ≈ a.data[2].value
    @test evalpoly(x, P16[3]) ≈ a.data[3].value
end

# test second, fourth, and eighth order horner schemes
# note that the higher order schemes are more prone to overflow when using lower precision
# because you have to directly compute x^2, x^4, x^8 before the routine
let
    for N in [2, 3, 4, 5, 6, 7, 10, 13, 17, 20, 52, 89], x in [0.1, 0.5, 1.5, 4.2, 45.0]
        poly = ntuple(n -> rand()*(-1)^n / n, N)
        @test evalpoly(x, poly) ≈ horner(x, pack_horner(poly, Val(2))) ≈ horner(x, pack_horner(poly, Val(4))) ≈ horner(x, pack_horner(poly, Val(8))) ≈ horner(x, pack_horner(poly, Val(16))) ≈ horner(x, pack_horner(poly, Val(32)))
    end
    for N in [2, 3, 4, 5, 6, 7, 10], x32 in [0.1f0, 0.8f0, 2.4f0, 8.0f0, 25.0f0]
        poly32 = ntuple(n -> Float32(rand()*(-1)^n / n), N)
        @test evalpoly(x32, poly32) ≈ horner(x32, pack_horner(poly32, Val(2))) ≈ horner(x32, pack_horner(poly32, Val(4))) ≈ horner(x32, pack_horner(poly32, Val(8))) ≈ horner(x32, pack_horner(poly32, Val(16))) ≈ horner(x32, pack_horner(poly32, Val(32)))
    end
    # Float16 is unreliable
    for N in [2, 3, 4, 5, 6], x in [0.1, 0.5, 1.1]
        poly16 = ntuple(n -> Float16(rand()*(-1)^n / n), N)
        x16 = Float16.(x)
        @test evalpoly(x16, poly16) ≈ horner(x16, pack_horner(poly16, Val(2))) ≈ horner(x16, pack_horner(poly16, Val(4))) ≈ horner(x16, pack_horner(poly16, Val(8))) ≈ horner(x16, pack_horner(poly16, Val(16))) skip=true
    end

end

# Tests for Clenshaw algorithm to evaluate Chebyshev polynomials

# non-SIMD version
function clen(x, c)
    x2 = 2x
    c0 = c[end-1]
    c1 = c[end]
    for i in length(c)-2:-1:1
        c0, c1 = c[i] - c1, c0 + c1 * x2
    end
    return c0 + c1 * x
end

let
    for N in [2, 6, 10], x in [0.1, 0.5, 1.5, 4.2, 45.0]
        P = (
            ntuple(n -> rand()*(-1)^n / n, N),
            ntuple(n -> rand()*(-1)^n / n, N),
            ntuple(n -> rand()*(-1)^n / n, N),
            ntuple(n -> rand()*(-1)^n / n, N)
        )
        # the native code generated between the SIMD and non-simd cases
        # is slightly different due to the SIMD case always using the fsub instruction
        # where the non-simd case sometimes chooses to reorder this in the native code generation
        # some small tests showed the SIMD case ordering was slightly more accurate
        # the SIMD case using this instruction is also faster than even a single evaluation
        a = clenshaw_simd(x, pack_poly(P))
        @test clen(x, P[1]) ≈ a.data[1].value
        @test clen(x, P[2]) ≈ a.data[2].value
        @test clen(x, P[3]) ≈ a.data[3].value
        @test clen(x, P[4]) ≈ a.data[4].value
    end
end


# test complex

let
    p = complex.(ntuple(i->rand(), 2), ntuple(i->rand(), 2))
    p2 = complex.(ntuple(i->rand(), 2), ntuple(i->rand(), 2))
    pr = ntuple(i->rand(), 2)

    pc = SIMDMath.ComplexVec(p)
    pc2 = SIMDMath.ComplexVec(p2)
    pr1 = SIMDMath.Vec(pr)

    # multiply

    pcmul = SIMDMath.fmul(pc, pc2)
    pmul = p .* p2
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    pcmul = SIMDMath.fmul(pc, pr1)
    @test pcmul == SIMDMath.fmul(pr1, pc)
    pmul = p .* pr
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im
    
    # add

    pcmul = SIMDMath.fadd(pc, pc2)
    pmul = p .+ p2
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    pcmul = SIMDMath.fadd(pc, pr1)
    @test pcmul == SIMDMath.fadd(pr1, pc)
    pmul = p .+ pr
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    # subtract

    pcmul = SIMDMath.fsub(pc, pc2)
    pmul = p .- p2
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    pcmul = SIMDMath.fsub(pc, pr1)
    @test pcmul == SIMDMath.fsub(pr1, pc)
    pmul = p .- pr
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    # multiply add

    pcmul = SIMDMath.muladd(pc, pc2, pc)
    pmul = muladd.(p, p2, p)
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    pcmul = SIMDMath.muladd(pc, pr1, pc)
    @test pcmul == SIMDMath.muladd(pr1, pc, pc)
    pmul = muladd.(p, pr, p)
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    pcmul = SIMDMath.muladd(pc, pr1, pr1)
    pmul = muladd.(p, pr, pr)
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    # multiply subtract

    pcmul = SIMDMath.mulsub(pc, pc2, pc)
    pmul = @. p*p2 - p
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    pcmul = SIMDMath.mulsub(pc, pr1, pc)
    @test pcmul == SIMDMath.mulsub(pr1, pc, pc)
    pmul = @. p*pr - p
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    pcmul = SIMDMath.mulsub(pc, pr1, pr1)
    pmul = @. p*pr - pr
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    P1 = (1.1, 1.2, 1.4, 1.5, 1.3, 1.4, 1.5, 1.6, 1.7, 1.2, 1.2, 2.1, 3.1, 1.4, 1.5)
    P2 = (1.1, 1.2, 1.4, 1.53, 1.32, 1.41, 1.52, 1.64, 1.4, 1.0, 1.6, 2.5, 3.1, 1.9, 1.2)
    pp3 = pack_poly((P1, P2))
    z = 1.2 + 1.1im
    s = horner_simd(z, pp3)
    e = evalpoly(z, P1)

    @test s.re[1].value == e.re
    @test s.im[1].value == e.im

    e = evalpoly(z, P2)
    @test s.re[2].value == e.re
    @test s.im[2].value == e.im
end

end
