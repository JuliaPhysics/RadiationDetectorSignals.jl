# This file is a part of RadiationDetectorSignals.jl, licensed under the MIT License (MIT).

__precompile__(true)

module RadiationDetectorSignals

using ArraysOfArrays
using ElasticArrays
using FillArrays
using IntervalSets
using RecipesBase
using StaticArrays
using StructArrays
using Tables
using TypedTables
using UnsafeArrays
using Unitful

import StatsBase

include("numeric_types.jl")
include("array_types.jl")
include("detector_hits.jl")
include("detector_waveforms.jl")
include("plots_recipes.jl")

end # module
