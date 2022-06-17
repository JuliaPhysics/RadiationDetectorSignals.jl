# This file is a part of RadiationDetectorSignals.jl, licensed under the MIT License (MIT).

using RadiationDetectorSignals
using Test

using ArraysOfArrays, FillArrays, StructArrays, Unitful


@testset "detector_waveforms" begin
    nwf = 50
    wfdata = nestedview(rand(128, nwf))
    timedata = Fill(0:0.1:12.7,nwf)

    @test @inferred(ArrayOfRDWaveforms((wfdata, timedata))) isa StructArray
    A = ArrayOfRDWaveforms((timedata, wfdata))

    @test A.time[1] == A[1].time == 0:0.1:12.7
    @test A.signal isa ArrayOfSimilarArrays
    @test A.signal[1] == A[1].signal
end # testset

@testset "detector_waveforms with units" begin
    nwf = 50
    wfdata = nestedview(rand(128, nwf) * u"eV")
    timedata = Fill((0:0.1:12.7) * u"ns",nwf)

    @test @inferred(ArrayOfRDWaveforms((wfdata, timedata))) isa StructArray
    A = ArrayOfRDWaveforms((timedata, wfdata))

    @test A.time[1] == A[1].time == (0:0.1:12.7) * u"ns"
    @test A.signal isa ArrayOfSimilarArrays
    @test A.signal[1] == A[1].signal
end # testset
