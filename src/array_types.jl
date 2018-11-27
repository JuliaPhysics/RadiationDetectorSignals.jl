# This file is a part of LegendDataTypes.jl, licensed under the MIT License (MIT).

# TODO: Replace by true custom types to avoid type piracy on NamedTuple
# (even though the named tuple types used here are quite specific).


const ArrayOfDims{N} = Array{T,N} where T
const AosAOfDims{M,N} = ArrayOfSimilarArrays{T,M,N} where T
const SArrayOfDims{N} = SArray{TP,T,N,L} where {TP,L,T}


Base.@pure recursive_ndims(::Type{<:AbstractArray{T,N}}) where {T,N} = N + recursive_ndims(T)
Base.@pure recursive_ndims(::Type{<:RealQuantity}) = 0
