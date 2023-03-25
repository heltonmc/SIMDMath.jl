import Base: muladd

## ShuffleVector

# create tuples of VecElements filled with a constant value of x
@inline constantvector(x::T, y) where T <: FloatTypes = constantvector(VE(x), y)
@inline constantvector(x::T, ::Type{Vec{N, T}}) where {N, T <: FloatTypes} = Vec{N, T}(constantvector(VE(x), LVec{N, T}))

@inline @generated function constantvector(x::VecElement{T}, y::Type{LVec{N, T}}) where {N, T <: FloatTypes}
    s = """
        %2 = insertelement <$N x $(LLVMType[T])> undef, $(LLVMType[T]) %0, i32 0
        %3 = shufflevector <$N x $(LLVMType[T])> %2, <$N x $(LLVMType[T])> undef, <$N x i32> zeroinitializer
        ret <$N x $(LLVMType[T])> %3
        """
    return :(
        llvmcall($s, LVec{N, T}, Tuple{VecElement{T}}, x)
        )
end

# Generic function types

for f in (:muladd, :mulsub)
    @eval begin
        @inline $f(x::Vec{N, T}, y::Vec{N, T}, z::Vec{N, T}) where {N, T <: FloatTypes} = Vec($f(x.data, y.data, z.data))
        @inline $f(x::ScalarTypes, y::Vec{N, T}, z::Vec{N, T}) where {N, T <: FloatTypes} = $f(constantvector(x, Vec{N, T}), y, z)
        @inline $f(x::Vec{N, T}, y::ScalarTypes, z::Vec{N, T}) where {N, T <: FloatTypes} = $f(x, constantvector(y, Vec{N, T}), z)
        @inline $f(x::ScalarTypes, y::ScalarTypes, z::Vec{N, T}) where {N, T <: FloatTypes} = $f(constantvector(x, Vec{N, T}), constantvector(y, Vec{N, T}), z)
        @inline $f(x::Vec{N, T}, y::Vec{N, T}, z::ScalarTypes) where {N, T <: FloatTypes} = $f(x, y, constantvector(z, Vec{N, T}))
        @inline $f(x::ScalarTypes, y::Vec{N, T}, z::ScalarTypes) where {N, T <: FloatTypes} = $f(constantvector(x, Vec{N, T}), y, constantvector(z, Vec{N, T}))
        @inline $f(x::Vec{N, T}, y::ScalarTypes, z::ScalarTypes) where {N, T <: FloatTypes} = $f(x, constantvector(y, Vec{N, T}), constantvector(z, Vec{N, T}))
    end
end

for f in (:fadd, :fsub, :fmul, :fdiv)
    @eval begin
        @inline $f(x::Vec{N, T}, y::Vec{N, T}) where {N, T <: FloatTypes} = Vec($f(x.data, y.data))
        @inline $f(x::Vec{N, T}, y::ScalarTypes) where {N, T <: FloatTypes} = $f(x, constantvector(y, Vec{N, T}))
        @inline $f(x::ScalarTypes, y::Vec{N, T}) where {N, T <: FloatTypes} = $f(constantvector(x, Vec{N, T}), y)
    end
end

# muladd llvm instructions

@inline @generated function muladd(x::LVec{N, T}, y::LVec{N, T}, z::LVec{N, T}) where {N, T <: FloatTypes}
    s = """
        %4 = fmul contract <$N x $(LLVMType[T])> %0, %1
        %5 = fadd contract <$N x $(LLVMType[T])> %4, %2
        ret <$N x $(LLVMType[T])> %5
        """
    return :(
        llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}, LVec{N, T}}, x, y, z)
        )
end

@inline @generated function mulsub(x::LVec{N, T}, y::LVec{N, T}, z::LVec{N, T}) where {N, T <: FloatTypes}
    s = """
        %4 = fmul contract <$N x $(LLVMType[T])> %0, %1
        %5 = fsub contract <$N x $(LLVMType[T])> %4, %2
        ret <$N x $(LLVMType[T])> %5
        """
    return :(
        llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}, LVec{N, T}}, x, y, z)
        )
end

for f in (:fadd, :fsub, :fmul, :fdiv)
    @eval @inline @generated function $f(x::LVec{N, T}, y::LVec{N, T}) where {N, T <: FloatTypes}
        ff = $(QuoteNode(f))
        s = """
        %3 = $ff <$N x $(LLVMType[T])> %0, %1
        ret <$N x $(LLVMType[T])> %3
        """
        return :(
        llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}}, x, y)
        )
    end
end
