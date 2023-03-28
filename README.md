# SIMDMath.jl

[![Build Status](https://github.com/heltonmc/SIMDMath.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/heltonmc/SIMDMath.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/heltonmc/SIMDMath.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/heltonmc/SIMDMath.jl)


A lightweight module for explicit vectorization of simple math functions. The focus is mainly on vectorizing polynomial evaluation in two main cases: (1) evaluating many different polynomials of similar length and (2) evaluating a single large polynomial. It is primary used for vectorizing Bessel function evaluation in [Bessels.jl](https://github.com/JuliaMath/Bessels.jl).

This module is for statically known functions where the coefficients are unrolled and the size of the tuples is known at compilation. For more advanced needs it will be better to use SIMD.jl or LoopVectorization.jl.
[SIMDPoly.jl](https://github.com/augustt198/SIMDPoly.jl) is a similar package utilizing SIMD.jl.

Experimental support for complex numbers is provided. This package requires at least Julia v1.8.

### Case 1: Evaluating many different polynomials.

In the evaluation of special functions, we often need to compute many polynomials at the same `x`. An example structure would look like...
```julia
const NT = 12
const P = (
           ntuple(n -> rand()*(-1)^n / n, NT),
           ntuple(n -> rand()*(-1)^n / n, NT),
           ntuple(n -> rand()*(-1)^n / n, NT),
           ntuple(n -> rand()*(-1)^n / n, NT)
       )

function test(x)
           x2 = x * x
           x3 = x2 * x

           p1 = evalpoly(x3, P[1])
           p2 = evalpoly(x3, P[2])
           p3 = evalpoly(x3, P[3])
           p4 = evalpoly(x3, P[4])

           return muladd(x, -p2, p1), muladd(x2, p4, -p3)
       end
```
This structure is advantageous for vectorizing as `p1`, `p2`, `p3`, and `p4` are independent, require same number of evaluations, and coefficients are statically known.
However, we are relying on the auto-vectorizer to make sure this happens which is very fragile. In general, two polynomials might auto-vectorizer depending on how the values are used but is not reliable.
We can check that this function is not vectorizing (though it may on some architectures) by using `@code_llvm test(1.1)` and/or `@code_native(1.1)`.

Another way to test this is to benchmark this function and compare to the time to compute a single polynomial.
```julia
julia> @btime test(x) setup=(x=rand()*2)
  13.026 ns (0 allocations: 0 bytes)

julia> @btime evalpoly(x, P[1]) setup=(x=rand()*2)
  3.973 ns (0 allocations: 0 bytes)
```
In this case, `test` is almost 4x longer as all the polynomial evaluations are happening sequentially.

We can do much better by making sure these polynomials vectorize.
```julia
# using the same coefficients as above
julia> using SIMDMath

const pack_p = pack_poly(P)

@inline function test_simd(x)
       x2 = x * x
       x3 = x2 * x
       p = horner_simd(x3, pack_p)
       return muladd(x, -p.data[2].value, p.data[1].value), muladd(x2, p.data[4].value, -p.data[3].value)
end

julia> @btime test_simd(x) setup=(x=rand()*2)
  4.440 ns (0 allocations: 0 bytes)
```

### Case 2: Evaluating a single polynomial.

In some cases, we are interested in improving the performance when evaluating a single polynomial of larger degree. Horner's scheme is latency bound and for large polynomials (N>10) this can become a large part of the total runtime. We can test the performance of using a straight Horner scheme using the Base library function `evalpoly` against the higher order Horner schemes.
```julia
let
    horner_times = []
    horner2_times = []
    horner4_times = []
    horner8_times = []
    horner16_times = []
    horner32_times = []

    for N in [4, 8, 12, 16, 32, 64, 128, 256, 512]
        poly = ntuple(n -> rand()*(-1)^n / n, N)
        poly_packed2 = pack_horner(poly, Val(2))
        poly_packed4 = pack_horner(poly, Val(4))
        poly_packed8 = pack_horner(poly, Val(8))
        poly_packed16 = pack_horner(poly, Val(16))
        poly_packed32 = pack_horner(poly, Val(32))


        t1 = @benchmark evalpoly(x, $poly) setup=(x=rand())
        t2 = @benchmark horner(x, $poly_packed2) setup=(x=rand())
        t3 = @benchmark horner(x, $poly_packed4) setup=(x=rand())
        t4 = @benchmark horner(x, $poly_packed8) setup=(x=rand())
        t5 = @benchmark horner(x, $poly_packed16) setup=(x=rand())
        t6 = @benchmark horner(x, $poly_packed32) setup=(x=rand())


        push!(horner_times,  round(minimum(t1).time, digits=3))
        push!(horner2_times,  round(minimum(t2).time, digits=3))
        push!(horner4_times,  round(minimum(t3).time, digits=3))
        push!(horner8_times,  round(minimum(t4).time, digits=3))
        push!(horner16_times,  round(minimum(t5).time, digits=3))
        push!(horner32_times,  round(minimum(t6).time, digits=3))
    end


    
    
    using Plots
    plot([4, 8, 12, 16, 32, 64, 128, 256, 512], horner_times ./ horner2_times, lw=1.5, label="2nd Order", xlabel="N degree polynomial", ylabel="Relative speedup to evalpoly", legend=:topleft)
    plot!([4, 8, 12, 16, 32, 64, 128, 256, 512], horner_times ./ horner4_times, lw=1.5, label="4th Order")
    plot!([4, 8, 12, 16, 32, 64, 128, 256, 512], horner_times ./ horner8_times, lw=1.5, label="8th Order")
    plot!([4, 8, 12, 16, 32, 64, 128, 256, 512], horner_times ./ horner16_times, lw=1.5, label="16th Order")
    plot!([4, 8, 12, 16, 32, 64, 128, 256, 512], horner_times ./ horner32_times, lw=1.5, label="32nd Order")

end
```

![Alt text](/assets/horner_benchmark.png "Horner Benchmark")

As mentioned, Horner's scheme requires sequential multiply-add instructions that can't be performed in parallel. One way (another way is Estrin's method which we won't discuss) to improve this structure is to break the polynomial down into even and odd polynomials (a second order Horner's scheme) or into larger powers of `x^4` or `x^8` (a fourth and eighth order Horner's scheme) which allow for computing many different polynomials of similar length simultaneously. In some regard, we are just rearranging the coefficients and using the same method as we did in the first case with some additional arithmetic at the end to add all the different polynomials together. This method should be considered a fastmath approach as it rearranges the floating point arithmetic.

The last fact is important because we are actually increasing the total amount of arithmetic operations needed but increasing by a large amount the number of operations that can happen in parallel. The increased operations make the advantages of this approach less straightforward than the first case which is always superior. The second and perhaps most important point is that floating point arithmetic is not associative so these approaches will give slightly different results as we are adding and multiplying in slightly differnet order.

Asymptotically, we can see that the method approaches a 2, 4, and 8x increase respecitively for large degrees, however, for smaller degrees the advanges are more complex. Therefore, it is encouraged to test the performance for individual cases. Of course, this depends on statically knowing the polynomial size during compilation which allows for packing the coefficients in the most efficient way.

Which order of Horner's method to use will depend on the degree polynomial we want to evaluate. A second order scheme is the fastest for degrees N < 12 and is faster than the standard `evalpoly` even for small degrees N < 4. For 12 < N < 30, a 4th or even 8th degree polynomial will be preferred while a 16th and 32nd order scheme will be preferred for very large polynomials N > 75. The above benchmark should be run on the desired computer and measured for the static degree to see the fastest approach.

### Case 3: Evaluating a polynomial in Chebyshev basis.

Similar to the first case evaluating many different polynomials this is also important when using a Chebyshev basis particularly in 2D problems. A simple comparison is the following...

```julia
# define scalar version
function clenshaw_chebyshev(x, c)
    x2 = 2x
    c0 = c[end-1]
    c1 = c[end]
    for i in length(c)-2:-1:1
        c0, c1 = c[i] - c1, c0 + c1 * x2
    end
    return c0 + c1 * x
end

# scalar version evaluating single polynomial
julia> @benchmark clenshaw_chebyshev(x, (1.2, 1.2, 1.3, 1.5, 1.6, 1.8, 1.9, 2.1, 2.2, 2.3, 2.5, 1.3, 1.5, 1.6, 1.8, 1.9, 2.1, 2.2)) setup=(x=rand())
BenchmarkTools.Trial: 10000 samples with 1000 evaluations.
 Range (min … max):  7.041 ns … 24.583 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     7.166 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   7.173 ns ±  0.384 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

                         ▂          █                         
  ▂▁▁▁▁▁▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▃ ▂
  7.04 ns        Histogram: frequency by time        7.25 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.

# SIMD version evaluating two polynomials...
julia> const P2 =  SIMDMath.pack_horner(((1.2, 1.2, 1.3, 1.5, 1.6, 1.8, 1.9, 2.1, 2.2, 2.3, 2.5, 1.3, 1.5, 1.6, 1.8, 1.9, 2.1, 2.2), (2.4, 1.3, 1.5, 1.6, 1.8, 1.9, 2.1, 2.2, 2.1, 2.6, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8)))
((VecElement{Float64}(1.2), VecElement{Float64}(2.4)), (VecElement{Float64}(1.2), VecElement{Float64}(1.3)), (VecElement{Float64}(1.3), VecElement{Float64}(1.5)), (VecElement{Float64}(1.5), VecElement{Float64}(1.6)), (VecElement{Float64}(1.6), VecElement{Float64}(1.8)), (VecElement{Float64}(1.8), VecElement{Float64}(1.9)), (VecElement{Float64}(1.9), VecElement{Float64}(2.1)), (VecElement{Float64}(2.1), VecElement{Float64}(2.2)), (VecElement{Float64}(2.2), VecElement{Float64}(2.1)), (VecElement{Float64}(2.3), VecElement{Float64}(2.6)), (VecElement{Float64}(2.5), VecElement{Float64}(2.1)), (VecElement{Float64}(1.3), VecElement{Float64}(2.2)), (VecElement{Float64}(1.5), VecElement{Float64}(2.3)), (VecElement{Float64}(1.6), VecElement{Float64}(2.4)), (VecElement{Float64}(1.8), VecElement{Float64}(2.5)), (VecElement{Float64}(1.9), VecElement{Float64}(2.6)), (VecElement{Float64}(2.1), VecElement{Float64}(2.7)), (VecElement{Float64}(2.2), VecElement{Float64}(2.8)))

julia> @benchmark SIMDMath.clenshaw_simd(x, P2) setup=(x=rand())
BenchmarkTools.Trial: 10000 samples with 1000 evaluations.
 Range (min … max):  4.291 ns … 24.000 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     4.416 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   4.415 ns ±  0.368 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

             ▂           █          █           ▃          ▂ ▂
  ▅▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁█ █
  4.29 ns      Histogram: log(frequency) by time      4.5 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
 ```

 Computing two Chebyshev polynomials is actually faster than the single polynomial case. The coefficients are packed more efficiently and the operations are not done in the same order. This leads to a speed up, however, because of the non-associativity of floating point arithmetic they will slightly differ. One is not neccessarily more accurate and should be tested for your use case.
 
