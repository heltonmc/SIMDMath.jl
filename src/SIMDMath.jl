#
# SIMDMath aims to provide vectorized basic math operations to be used in static fully unrolled functions such as computing special functions.
#
# The type system is heavily influenced by SIMD.jl (https://github.com/eschnett/SIMD.jl) licensed under the Simplified "2-clause" BSD License:
# Copyright (c) 2016-2020: Erik Schnetter, Kristoffer Carlsson, Julia Computing All rights reserved.
#
# This module is also influenced by VectorizationBase.jl (https://github.com/JuliaSIMD/VectorizationBase.jl) licensed under the MIT License: Copyright (c) 2018 Chris Elrod
#

module SIMDMath

using Base: llvmcall, VecElement
using Base.Cartesian: @ntuple, @nexprs

export horner_simd, pack_poly
export horner, pack_horner

export clenshaw_simd

include("types.jl")
include("intrinsics.jl")
include("interface.jl")
include("complex.jl")
include("horner.jl")

end
