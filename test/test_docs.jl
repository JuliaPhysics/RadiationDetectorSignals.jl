# This file is a part of RadiationDetectorSignals.jl, licensed under the MIT License (MIT).

using Test
using RadiationDetectorSignals
import Documenter

Documenter.DocMeta.setdocmeta!(
    RadiationDetectorSignals,
    :DocTestSetup,
    :(using RadiationDetectorSignals);
    recursive=true,
)
Documenter.doctest(RadiationDetectorSignals)
