using SIMDMath: LVec

let
    # test shufflevector
    using SIMDMath: shufflevector

    a = LVec{4, Float64}((1.2, 1.3, 1.4, 1.5))
    b = shufflevector(a, Val(1))
    @test b[1].value == 1.3
    b = shufflevector(a, Val(2:3))
    @test b[1].value == 1.4
    @test b[2].value == 1.5

    c = LVec{4, Float64}((2.2, 2.3, 2.4, 2.5))
    b = shufflevector(a, c, Val((0, 3, 5, 7)))
    @test (b[1].value, b[2].value, b[3].value, b[4].value) == (1.2, 1.5, 2.3, 2.5)    
end

let 
    # test constantvector
    using SIMDMath: constantvector
    x = 1.2
    a = constantvector(x, LVec{4, Float64})
    @test (a[1].value, a[2].value, a[3].value, a[4].value) == (1.2, 1.2, 1.2, 1.2)
end
let 
    # test fmadd / fmsub / fnmadd / fnmsub / fmaddsub / fmsubadd
    using SIMDMath: fmadd, fmsub, fnmadd, fnmsub, fmaddsub, fmsubadd
    a = (0.1, 0.3)
    b = (0.2, 0.5)
    c = (0.4, 0.9)
    avec = LVec{2, Float64}(a)
    bvec = LVec{2, Float64}(b)
    cvec = LVec{2, Float64}(c)

    r = muladd.(a, b, c)
    rvec = fmadd(avec, bvec, cvec)
    @test (rvec[1].value, rvec[2].value) == r

    r = @. a*b - c
    rvec = fmsub(avec, bvec, cvec)
    @test (rvec[1].value, rvec[2].value) == r

    r = @. -a*b + c
    rvec = fnmadd(avec, bvec, cvec)
    @test (rvec[1].value, rvec[2].value) == r

    r = @. -a*b - c
    rvec = fnmsub(avec, bvec, cvec)
    @test (rvec[1].value, rvec[2].value) == r

    r = (a[1]*b[1] - c[1], a[2]*b[2] + c[2])
    rvec = fmaddsub(avec, bvec, cvec)
    @test (rvec[1].value, rvec[2].value) == r

    r = (a[1]*b[1] + c[1], a[2]*b[2] - c[2])
    rvec = fmsubadd(avec, bvec, cvec)
    @test (rvec[1].value, rvec[2].value) == r

    a = SIMDMath.LVec{4, Float64}((1.1, 1.2, 1.3, 1.4))
    b = SIMDMath.LVec{4, Float64}((1.1, 1.3, 1.6, 1.5))
    c = SIMDMath.LVec{4, Float64}((1.3, 1.6, 1.9, 3.1))

    o = SIMDMath.fmaddsub(a, b, c)
    @test o[1].value ≈ 1.1*1.1 - 1.3
    @test o[2].value ≈ 1.2*1.3 + 1.6
    @test o[3].value ≈ 1.3*1.6 - 1.9
    @test o[4].value ≈ 1.4*1.5 + 3.1

    o = SIMDMath.fmsubadd(a, b, c)
    @test o[1].value ≈ 1.1*1.1 + 1.3
    @test o[2].value ≈ 1.2*1.3 - 1.6
    @test o[3].value ≈ 1.3*1.6 + 1.9
    @test o[4].value ≈ 1.4*1.5 - 3.1
end
let 
    # test faddsub / fsubadd / fadd / fsub / fmul / fdiv / fneg
    using SIMDMath: faddsub, fsubadd, fadd, fsub, fmul, fdiv, fneg
    a = (0.1, 0.3)
    b = (0.2, 0.5)
    avec = LVec{2, Float64}(a)
    bvec = LVec{2, Float64}(b)

    r = (a[1] - b[1], a[2] + b[2])
    rvec = faddsub(avec, bvec)
    @test (rvec[1].value, rvec[2].value) == r

    r = (a[1] + b[1], a[2] - b[2])
    rvec = fsubadd(avec, bvec)
    @test (rvec[1].value, rvec[2].value) == r
    
    r = @. a + b
    rvec = fadd(avec, bvec)
    @test (rvec[1].value, rvec[2].value) == r

    r = @. a - b
    rvec = fsub(avec, bvec)
    @test (rvec[1].value, rvec[2].value) == r

    r = @. a * b
    rvec = fmul(avec, bvec)
    @test (rvec[1].value, rvec[2].value) == r

    r = @. a / b
    rvec = fdiv(avec, bvec)
    @test (rvec[1].value, rvec[2].value) == r

    r = @. -a
    rvec = fneg(avec)
    @test (rvec[1].value, rvec[2].value) == r

end
let
    # test extractelement and getindex
    a = Vec{4, Float64}((1.2, 1.3, 1.4, 1.5))
    b = ComplexVec{4, Float64}((1.2, 1.3, 1.4, 1.5), (1.3, 1.4, 1.5, 1.6))

    @test a[1] == 1.2
    @test (@inbounds a[4] == 1.5)

    @test b[2] == 1.3 + 1.4im
    @test (@inbounds b[3] == 1.4 + 1.5im)
end