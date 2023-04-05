using SIMDMath
using Test

@time @testset "instrinsics" include("intrinsics_test.jl")
@time @testset "complex" include("complex_test.jl")
@time @testset "horner" include("horner_test.jl")
