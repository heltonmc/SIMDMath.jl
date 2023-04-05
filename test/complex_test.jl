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

    pcmul = SIMDMath.fmadd(pc, pc2, pc)
    pmul = muladd.(p, p2, p)
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    pcmul = SIMDMath.fmadd(pc, pr1, pc)
    @test pcmul == SIMDMath.fmadd(pr1, pc, pc)
    pmul = muladd.(p, pr, p)
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    pcmul = SIMDMath.fmadd(pc, pr1, pr1)
    pmul = muladd.(p, pr, pr)
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    # multiply subtract

    pcmul = SIMDMath.fmsub(pc, pc2, pc)
    pmul = @. p*p2 - p
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    pcmul = SIMDMath.fmsub(pc, pr1, pc)
    @test pcmul == SIMDMath.fmsub(pr1, pc, pc)
    pmul = @. p*pr - p
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    pcmul = SIMDMath.fmsub(pc, pr1, pr1)
    pmul = @. p*pr - pr
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    # complex negated multiply-add
    # -a*b + c
    pcmul = SIMDMath.fnmadd(pc, pc2, pc)
    pmul = @. -p*p2 + p
    @test pcmul.re[1].value ≈ pmul[1].re
    @test pcmul.im[1].value ≈ pmul[1].im
    @test pcmul.re[2].value ≈ pmul[2].re
    @test pcmul.im[2].value ≈ pmul[2].im

    # -a*b - c
    pcmul = SIMDMath.fnmsub(pc, pc2, pc)
    pmul = @. -p*p2 - p
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