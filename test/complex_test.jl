# test complex

using SIMDMath: fmul, fadd, fsub
using SIMDMath: fmadd, fmsub, fnmadd, fnmsub
using SIMDMath: ComplexVec, Vec

# define scalar functions
mulsub(a, b, c) = a*b - c
nmuladd(a, b, c) = -a*b + c
nmulsub(a, b, c) = -a*b - c

cvec1 = complex.(ntuple(i->rand(), 2), ntuple(i->rand(), 2))
cvec2 = complex.(ntuple(i->rand(), 2), ntuple(i->rand()*(-1)^i, 2))
cvec3 = complex.(ntuple(i->rand()*(-1)^i, 2), ntuple(i->rand()*(-1)^(2i), 2))

rvec1 = ntuple(i->rand(), 2)
rvec2 = ntuple(i->rand()*(-1)^(i), 2)
rvec3 = ntuple(i->rand()*(-1)^(2i), 2)

cscal1 = 1.2 + 1.3im
cscal2 = 2.1 - 1.9im
cscal3 = -3.1 - 3.4im

rscal1 = 4.5
rscal2 = -1.2
rscal3 = 6.5

for (f, f2) in ((:fmul, :*), (:fadd, :+), (:fsub, :-))
    @eval begin
        for a in ((cvec1, ComplexVec(cvec1)), (cvec2, ComplexVec(cvec2)), (cvec3, ComplexVec(cvec3)), (rvec1, Vec(rvec1)), (rvec2, Vec(rvec2)), (rvec3, Vec(rvec3)), (cscal1, cscal1), (cscal3, cscal3), (cscal3, cscal3), (rscal1, rscal1), (rscal2, rscal2), (rscal3, rscal3))
            for b in ((cvec1, ComplexVec(cvec1)), (cvec2, ComplexVec(cvec2)), (cvec3, ComplexVec(cvec3)), (rvec1, Vec(rvec1)), (rvec2, Vec(rvec2)), (rvec3, Vec(rvec3)), (cscal1, cscal1), (cscal3, cscal3), (cscal3, cscal3), (rscal1, rscal1), (rscal2, rscal2), (rscal3, rscal3))

                vec = $f(a[2], b[2])
                scal = @. $f2(a[1], b[1])
                @test vec[1] ≈ scal[1]
                if length(scal) == 2
                    @test vec[2] ≈ scal[2]
                end
            end
        end
    end
end

for (f, f2) in ((:fmadd, :muladd), (:fmsub, :mulsub), (:fnmadd, :nmuladd), (:fnmsub, :nmulsub))
    @eval begin
        for a in ((cvec1, ComplexVec(cvec1)), (cvec2, ComplexVec(cvec2)), (cvec3, ComplexVec(cvec3)), (rvec1, Vec(rvec1)), (rvec2, Vec(rvec2)), (rvec3, Vec(rvec3)), (cscal1, cscal1), (cscal3, cscal3), (cscal3, cscal3), (rscal1, rscal1), (rscal2, rscal2), (rscal3, rscal3))
            for b in ((cvec1, ComplexVec(cvec1)), (cvec2, ComplexVec(cvec2)), (cvec3, ComplexVec(cvec3)), (rvec1, Vec(rvec1)), (rvec2, Vec(rvec2)), (rvec3, Vec(rvec3)), (cscal1, cscal1), (cscal3, cscal3), (cscal3, cscal3), (rscal1, rscal1), (rscal2, rscal2), (rscal3, rscal3))
                for c in ((cvec1, ComplexVec(cvec1)), (cvec2, ComplexVec(cvec2)), (cvec3, ComplexVec(cvec3)), (rvec1, Vec(rvec1)), (rvec2, Vec(rvec2)), (rvec3, Vec(rvec3)), (cscal1, cscal1), (cscal3, cscal3), (cscal3, cscal3), (rscal1, rscal1), (rscal2, rscal2), (rscal3, rscal3))

                    vec = $f(a[2], b[2], c[2])
                    scal = @. $f2(a[1], b[1], c[1])
                    @test vec[1] ≈ scal[1]
                    if length(scal) == 2
                        @test vec[2] ≈ scal[2]
                    end

                end
            end
        end
    end
end

@test convert(ComplexVec{4, Float64}, 1.2) == ComplexVec{4, Float64}((1.2, 1.2, 1.2, 1.2), (0.0, 0.0, 0.0, 0.0))

# test conjugate
let
    c = (1.1 + 1.4im, 1.3 - 1.2im)
    cv = ComplexVec{2, Float64}((1.1, 1.3), (1.4, -1.2))
    cconj = conj(cv)
    @test cconj[1] == conj(c[1])
    @test cconj[2] == conj(c[2])
end

# test horizontal reduction
let 
    for N in (2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 16, 17)
        x = ntuple(i -> (complex(rand(), rand())), N)
        xvec = ComplexVec(x)
        @test sum(x) ≈ SIMDMath.fhadd(xvec)
        @test reduce(*, x) ≈ SIMDMath.fhmul(xvec)

        # test real case
        @test sum(real.(x)) ≈ SIMDMath.fhadd(Vec(xvec.re))
        @test reduce(*, real.(x)) ≈ SIMDMath.fhmul(Vec(xvec.re))
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
