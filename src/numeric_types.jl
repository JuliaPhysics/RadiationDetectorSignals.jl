# This file is a part of LegendDataTypes.jl, licensed under the MIT License (MIT).

# TODO: Replace by true custom types to avoid type piracy on NamedTuple
# (even though the named tuple types used here are quite specific).


const MaybeWithUnits{T} = Union{T,Quantity{<:T}}
const RealQuantity = MaybeWithUnits{<:Real}
