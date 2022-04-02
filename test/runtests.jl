# This file is a part of RadiationDetectorSignals.jl, licensed under the MIT License (MIT).

import Test
Test.@testset "Package RadiationDetectorSignals" begin
    include("test_detector_hits.jl")
    include("test_detector_waveforms.jl")
    include("test_docs.jl")
end # testset
