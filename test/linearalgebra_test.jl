let 
    using LinearAlgebra

    for n in (3, 5, 6, 12, 15, 18, 33, 45, 201)
        x = ntuple(i -> (complex(rand(), rand())), 200)
        y = ntuple(i -> (complex(rand(), rand())), 200)

        @test LinearAlgebra.dot(x, y) ≈ SIMDMath.dot(x, y)
        @test LinearAlgebra.dot(ComplexF32.(x), ComplexF32.(y)) ≈ SIMDMath.dot(ComplexF32.(x), ComplexF32.(y))
        @test LinearAlgebra.dot(ComplexF16.(x), ComplexF16.(y)) ≈ SIMDMath.dot(ComplexF16.(x), ComplexF16.(y))

        @test LinearAlgebra.dot(real.(x), real.(y)) ≈ SIMDMath.dot(real.(x), real.(y))
        @test LinearAlgebra.dot(Float32.(real.(x)), Float32.(real.(y))) ≈ SIMDMath.dot(Float32.(real.(x)), Float32.(real.(y)))
        @test LinearAlgebra.dot(Float16.(real.(x)), Float16.(real.(y))) ≈ SIMDMath.dot(Float16.(real.(x)), Float16.(real.(y)))

    end
    
end