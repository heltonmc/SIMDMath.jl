# Shuffle Vector

@inline @generated function shufflevector(x::LVec{N, T}, y::LVec{N, T}, ::Val{I}) where {N, T, I}
    shfl = _shuffle_vec(I)
    M = length(I)
    s = """
    %res = shufflevector <$N x $(LLVMType[T])> %0, <$N x $(LLVMType[T])> %1, <$M x i32> <$shfl>
    ret <$M x $(LLVMType[T])> %res
    """
    return :(
        llvmcall($s, LVec{$M, T}, Tuple{LVec{N, T}, LVec{N, T}}, x, y)
        )
end

@inline @generated function shufflevector(x::LVec{N, T}, ::Val{I}) where {N, T, I}
    shfl = _shuffle_vec(I)
    M = length(I)
    s = """
    %res = shufflevector <$(N) x $(LLVMType[T])> %0, <$N x $(LLVMType[T])> undef, <$M x i32> <$shfl>
    ret <$M x $(LLVMType[T])> %res
    """
    return :(
        llvmcall($s, LVec{$M, T}, Tuple{LVec{N, T}}, x)
        )
end

_shuffle_vec(I) = join((string("i32 ", i == :undef ? "undef" : Int32(i::Integer)) for i in I), ", ")

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

# Extract element

@inline @generated function extractelement(x::LVec{N, T}, i::I) where {N, T, I <: Integer}
    s = """
    %3 = extractelement <$N x $(LLVMType[T])> %0, $(LLVMType[I]) %1
    ret $(LLVMType[T]) %3
    """
    return :(
        llvmcall($s, T, Tuple{LVec{N, T}, $i}, x, i)
    )
end

# Fused multiply/add/subtract/negate intrinsics

# a*b + c
@inline @generated function fmadd(x::LVec{N, T}, y::LVec{N, T}, z::LVec{N, T}) where {N, T <: FloatTypes}
    s = """
        %4 = fmul contract <$N x $(LLVMType[T])> %0, %1
        %5 = fadd contract <$N x $(LLVMType[T])> %4, %2
        ret <$N x $(LLVMType[T])> %5
        """
    return :(
        llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}, LVec{N, T}}, x, y, z)
        )
end

# a*b - c
@inline @generated function fmsub(x::LVec{N, T}, y::LVec{N, T}, z::LVec{N, T}) where {N, T <: FloatTypes}
    s = """
        %4 = fmul contract <$N x $(LLVMType[T])> %0, %1
        %5 = fsub contract <$N x $(LLVMType[T])> %4, %2
        ret <$N x $(LLVMType[T])> %5
        """
    return :(
        llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}, LVec{N, T}}, x, y, z)
        )
end

# -a*b + c
@inline @generated function fnmadd(x::LVec{N, T}, y::LVec{N, T}, z::LVec{N, T}) where {N, T <: FloatTypes}
    s = """
        %4 = fmul contract <$N x $(LLVMType[T])> %0, %1
        %5 = fsub contract <$N x $(LLVMType[T])> %2, %4
        ret <$N x $(LLVMType[T])> %5
        """
    return :(
        llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}, LVec{N, T}}, x, y, z)
        )
end

# -a*b - c
@inline @generated function fnmsub(x::LVec{N, T}, y::LVec{N, T}, z::LVec{N, T}) where {N, T <: FloatTypes}
    s = """
        %4 = fmul contract <$N x $(LLVMType[T])> %0, %1
        %5 = fadd contract <$N x $(LLVMType[T])> %4, %2
        %6 = fneg contract <$N x $(LLVMType[T])> %5
        ret <$N x $(LLVMType[T])> %6
        """
    return :(
        llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}, LVec{N, T}}, x, y, z)
        )
end

# a*b + c for i = 1, 3, ...
# a*b - c for i = 0, 2, ...
@inline @generated function fmaddsub(x::LVec{N, T}, y::LVec{N, T}, z::LVec{N, T}) where {N, T <: FloatTypes}
    @assert iseven(N) "Vector length must be even"
    shfl = join((string("i32 ", Int32(i-1), ", i32 ", Int32(N+i)) for i in 1:2:N), ", ")
    s = """
        %4 = fmul contract <$N x $(LLVMType[T])> %0, %1
        %5 = fsub contract <$N x $(LLVMType[T])> %4, %2
        %6 = fadd contract <$N x $(LLVMType[T])> %4, %2
        %7 = shufflevector <$N x $(LLVMType[T])> %5, <$N x $(LLVMType[T])> %6, <$N x i32> <$shfl>
        ret <$N x $(LLVMType[T])> %7
        """
    return :(
        llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}, LVec{N, T}}, x, y, z)
        )
end

# a*b - c for i = 1, 3, ...
# a*b + c for i = 0, 2, ...
@inline @generated function fmsubadd(x::LVec{N, T}, y::LVec{N, T}, z::LVec{N, T}) where {N, T <: FloatTypes}
    @assert iseven(N) "Vector length must be even"
    shfl = join((string("i32 ", Int32(i-1), ", i32 ", Int32(N+i)) for i in 1:2:N), ", ")
    s = """
        %4 = fmul contract <$N x $(LLVMType[T])> %0, %1
        %5 = fadd contract <$N x $(LLVMType[T])> %4, %2
        %6 = fsub contract <$N x $(LLVMType[T])> %4, %2
        %7 = shufflevector <$N x $(LLVMType[T])> %5, <$N x $(LLVMType[T])> %6, <$N x i32> <$shfl>
        ret <$N x $(LLVMType[T])> %7
        """
    return :(
        llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}, LVec{N, T}}, x, y, z)
        )
end

# a + b for i = 1, 3, ...
# a - b for i = 0, 2, ...
@inline @generated function faddsub(x::LVec{N, T}, y::LVec{N, T}) where {N, T <: FloatTypes}
    @assert iseven(N) "Vector length must be even"
    shfl = join((string("i32 ", Int32(i-1), ", i32 ", Int32(N+i)) for i in 1:2:N), ", ")
    s = """
        %3 = fsub contract <$N x $(LLVMType[T])> %0, %1
        %4 = fadd contract <$N x $(LLVMType[T])> %0, %1
        %5 = shufflevector <$N x $(LLVMType[T])> %3, <$N x $(LLVMType[T])> %4, <$N x i32> <$shfl>
        ret <$N x $(LLVMType[T])> %5
        """
    return :(
        llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}}, x, y)
        )
end

# a - b for i = 1, 3, ...
# a + b for i = 0, 2, ...
@inline @generated function fsubadd(x::LVec{N, T}, y::LVec{N, T}) where {N, T <: FloatTypes}
    @assert iseven(N) "Vector length must be even"
    shfl = join((string("i32 ", Int32(i-1), ", i32 ", Int32(N+i)) for i in 1:2:N), ", ")
    s = """
        %3 = fsub contract <$N x $(LLVMType[T])> %0, %1
        %4 = fadd contract <$N x $(LLVMType[T])> %0, %1
        %5 = shufflevector <$N x $(LLVMType[T])> %4, <$N x $(LLVMType[T])> %3, <$N x i32> <$shfl>
        ret <$N x $(LLVMType[T])> %5
        """
    return :(
        llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}}, x, y)
        )
end

# Add/subtract/multiply/divide instinsics
for f in (:fadd, :fsub, :fmul, :fdiv)
    @eval @inline @generated function $f(x::LVec{N, T}, y::LVec{N, T}) where {N, T <: FloatTypes}
        ff = $(QuoteNode(f))
        s = """
        %3 = $ff contract <$N x $(LLVMType[T])> %0, %1
        ret <$N x $(LLVMType[T])> %3
        """
        return :(
        llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}}, x, y)
        )
    end
end

# negate vector
@inline @generated function fneg(x::LVec{N, T}) where {N, T <: FloatTypes}
    s = """
        %2 = fneg <$N x $(LLVMType[T])> %0
        ret <$N x $(LLVMType[T])> %2
        """
    return :(
        llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}}, x)
        )
end
