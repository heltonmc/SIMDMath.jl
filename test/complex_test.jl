# test complex

using SIMDMath: fmul, fadd, fsub
using SIMDMath: fmadd, fmsub, fnmadd, fnmsub

p = complex.(ntuple(i->rand(), 2), ntuple(i->rand(), 2))
p2 = complex.(ntuple(i->rand(), 2), ntuple(i->rand(), 2))
pr = ntuple(i->rand(), 2)

pc = SIMDMath.ComplexVec(p)
pc2 = SIMDMath.ComplexVec(p2)
pr1 = SIMDMath.Vec(pr)

for (f, f2) in ((:fmul, :*), (:fadd, :+), (:fsub, :-))
    @eval begin
        # complex vec | complex vec
        pcmul = $f(pc, pc2)
        pmul = @. $f2(p, p2)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]

        # complex vec | real vec
        pcmul = $f(pc, pr1)
        pmul = @. $f2(p, pr)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]

        # real vec | complex vec
        pcmul = $f(pr1, pc)
        pmul = @. $f2(pr, p)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]

        # complex vec | real scalar
        pcmul = $f(pc, 1.2)
        pmul = @. $f2(p, 1.2)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]

        # real scalar | complex vec
        pcmul = $f(1.5, pc)
        pmul = @. $f2(1.5, p)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]

        # complex scalar | complex vec
        pcmul = $f(2.76 + 1.1im, pc)
        pmul = @. $f2(2.76 + 1.1im, p)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]

        # complex vec | complex scalar
        pcmul = $f(pc, 4.76 + 1.12im)
        pmul = @. $f2(p, 4.76 + 1.12im)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]
    end
end

mulsub(a, b, c) = a*b - c
nmuladd(a, b, c) = -a*b + c
nmulsub(a, b, c) = -a*b - c

for (f, f2) in ((:fmadd, :muladd), (:fmsub, :mulsub), (:fnmadd, :nmuladd), (:fnmsub, :nmulsub))
    @eval begin
        # complex vec | complex vec | complex vec
        pcmul =$f(pc, pc2, pc)
        pmul = $f2.(p, p2, p)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]
    
        # complex vec | real vec | complex vec
        pcmul = $f(pc, pr1, pc)
        pmul = $f2.(p, pr, p)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]
        
        # complex vec | real vec | real vec
        pcmul = $f(pc, pr1, pr1)
        pmul = $f2.(p, pr, pr)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]

        # real vec | complex vec | complex vec
        pcmul = $f(pr1, pc, pc)
        pmul = $f2.(pr, p, p)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]

        # real vec | real vec | complex vec
        pcmul = $f(pr1, pr1, pc)
        pmul = $f2.(pr, pr, p)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]

        # complex vec | complex vec | complex scalar
        pcmul = $f(pc, pc, 1.1 + 1.4im)
        pmul = $f2.(p, p, 1.1 + 1.4im)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]

        # complex vec | complex scalar | complex vec
        pcmul = $f(pc, 1.6 + 2.2im, pc)
        pmul = $f2.(p, 1.6 + 2.2im, p)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]

        # complex scalar | complex vec | complex vec
        pcmul = $f(-5.8 + 1.3im, pc, pc)
        pmul = $f2.(-5.8 + 1.3im, p, p)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]

        # complex scalar | complex scalar | complex vec
        pcmul = $f(9.2 - 1.1im, -1.3 + 1.5im, pc)
        pmul = $f2.(9.2 - 1.1im, -1.3 + 1.5im, p)
        @test pcmul[1] ≈ pmul[1]
        @test pcmul[2] ≈ pmul[2]
    end
end

P1 = (1.1, 1.2, 1.4, 1.5, 1.3, 1.4, 1.5, 1.6, 1.7, 1.2, 1.2, 2.1, 3.1, 1.4, 1.5)
P2 = (1.1, 1.2, 1.4, 1.53, 1.32, 1.41, 1.52, 1.64, 1.4, 1.0, 1.6, 2.5, 3.1, 1.9, 1.2)
pp3 = pack_poly((P1, P2))
z = 1.2 + 1.1im
s = horner_simd(z, pp3)
e = evalpoly(z, P1)

@test s[1].re == e.re
@test s[1].im == e.im

e = evalpoly(z, P2)
@test s[2].re == e.re
@test s[2].im == e.im
