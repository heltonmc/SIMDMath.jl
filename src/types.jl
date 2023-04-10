const VE = Base.VecElement
const LVec{N, T} = NTuple{N, VE{T}}

const IntTypes      = Union{Int8, Int16, Int32, Int64}
const BIntTypes     = Union{IntTypes, Bool}
const UIntTypes     = Union{UInt8, UInt16, UInt32, UInt64}
const IntegerTypes  = Union{IntTypes, UIntTypes}
const FloatTypes = Union{Float16, Float32, Float64}
const ScalarTypes = Union{VE{FloatTypes}, FloatTypes}

struct Vec{N, T <: FloatTypes}
    data::LVec{N, T}
end

@inline Vec(v::NTuple{N, T}) where {N, T <: FloatTypes} = Vec(VE.(v))

const LLVMType = Dict{DataType, String}(
    Int8         => "i8",
    Int16        => "i16",
    Int32        => "i32",
    Int64        => "i64",
    Int128       => "i128",

    UInt8        => "i8",
    UInt16       => "i16",
    UInt32       => "i32",
    UInt64       => "i64",
    UInt128      => "i128",

    Float16  => "half",
    Float32  => "float",
    Float64  => "double",
)

const ScalarOrVec{N, T} = Union{ScalarTypes, Vec{N, T}}

Base.convert(::Type{Vec{N, T}}, x::Vec{N, T}) where {N, T <: FloatTypes} = x
Base.convert(::Type{Vec{N, T}}, x::T) where {N, T <: ScalarTypes} = constantvector(x, Vec{N, T})

Base.promote_rule(::Type{Vec{N, T}}, ::Type{T}) where {N, T <: FloatTypes} = Vec{N, T}

# Complex Types

struct ComplexVec{N, T<:FloatTypes}
    re::LVec{N, T}
    im::LVec{N, T}
end

const ComplexOrRealVec{N, T} = Union{Vec{N, T}, ComplexVec{N, T}}

ComplexVec(x::NTuple{N, T}, y::NTuple{N, T}) where {N, T <: FloatTypes} = ComplexVec(LVec{N, T}(x), LVec{N, T}(y))
ComplexVec(z::NTuple{N, Complex{T}}) where {N, T <: FloatTypes} = ComplexVec(real.(z), imag.(z))

Base.convert(::Type{ComplexVec{N, T}}, z::ComplexVec{N, T}) where {N, T <: FloatTypes} = z
Base.convert(::Type{ComplexVec{N, T}}, z::Complex{T}) where {N, T <: FloatTypes} = constantvector(z, ComplexVec{N, T})
Base.convert(::Type{ComplexVec{N, T}}, x::T) where {N, T <: ScalarTypes} = ComplexVec{N, T}(constantvector(x, LVec{N, T}), constantvector(zero(T), LVec{N, T}))

Base.promote_rule(::Type{ComplexVec{N, T}}, ::Type{Complex{T}}) where {N, T <: FloatTypes} = ComplexVec{N, T}
