const VE = Base.VecElement
const LVec{N, T} = NTuple{N, VE{T}}

const FloatTypes = Union{Float16, Float32, Float64}
const ScalarTypes = Union{VE{FloatTypes}, FloatTypes}

struct Vec{N, T <: FloatTypes}
    data::LVec{N, T}
end

@inline Vec(v::NTuple{N, T}) where {N, T <: FloatTypes} = Vec(VE.(v))

const LLVMType = Dict{DataType, String}(
    Float16  => "half",
    Float32  => "float",
    Float64  => "double",
)

# Complex Types

struct ComplexVec{N, T<:FloatTypes}
    re::LVec{N, T}
    im::LVec{N, T}
end

const ComplexorRealVec{N, T} = Union{Vec{N, T}, ComplexVec{N, T}}

ComplexVec(x::NTuple{N, T}, y::NTuple{N, T}) where {N, T <: FloatTypes} = ComplexVec(LVec{N, T}(x), LVec{N, T}(y))

ComplexVec(z::NTuple{N, Complex{T}}) where {N, T <: FloatTypes} = ComplexVec(real.(z), imag.(z))
