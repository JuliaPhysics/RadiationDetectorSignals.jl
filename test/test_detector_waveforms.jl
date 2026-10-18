# This file is a part of RadiationDetectorSignals.jl, licensed under the MIT License (MIT).

using RadiationDetectorSignals
using Test

using ArraysOfArrays, FillArrays, StructArrays, Unitful


@testset "detector_waveforms" begin
    nwf = 50
    wfdata = VectorOfSimilarVectors(rand(128, nwf))
    timedata = Fill(0:0.1:12.7,nwf)

    @test @inferred(ArrayOfRDWaveforms((wfdata, timedata))) isa StructArray
    A = ArrayOfRDWaveforms((timedata, wfdata))

    @test A.time[1] == A[1].time == 0:0.1:12.7
    @test A.signal isa ArrayOfSimilarArrays
    @test A.signal[1] == A[1].signal
end # testset

@testset "detector_waveforms with units" begin
    nwf = 50
    wfdata = VectorOfSimilarVectors(rand(128, nwf) * u"eV")
    timedata = Fill((0:0.1:12.7) * u"ns",nwf)

    @test @inferred(ArrayOfRDWaveforms((wfdata, timedata))) isa StructArray
    A = ArrayOfRDWaveforms((timedata, wfdata))

    @test A.time[1] == A[1].time == (0:0.1:12.7) * u"ns"
    @test A.signal isa ArrayOfSimilarArrays
    @test A.signal[1] == A[1].signal
end # testset

@testset "detector_waveform equality" begin
    wfdata = rand(128)
    timedata = 0:0.1:12.7

    wf1 = RDWaveform(wfdata, timedata)
    wf2 = RDWaveform(reverse(wfdata),  timedata)
    wf3 = RDWaveform(deepcopy(wfdata), timedata)

    @test wf1 != wf2
    @test wf1 == wf3

    A = ArrayOfRDWaveforms([wf1, wf2])
    B = ArrayOfRDWaveforms([wf1, wf3])
    C = ArrayOfRDWaveforms([wf3, wf2])

    @test A != B 
    @test A == C
end


@testset "detector_waveform indexing" begin
    # Slicing the signals changes their element type, from views of a reshaped
    # range to views of a matrix. The waveform element type must follow:
    t = (0:3) * u"μs"
    wfs(signals...) = [RDWaveform(t, s) for s in signals]

    A = ArrayOfRDWaveforms((fill(t, 5), VectorOfSimilarVectors(reshape(1:20, 4, 5))))

    @test A[Int32(2)] == RDWaveform(t, 5:8)

    B = A[:]
    @test eltype(B) == typeof(B[1])
    @test collect(B) == wfs(1:4, 5:8, 9:12, 13:16, 17:20)

    B = A[2:3]
    @test eltype(B) == typeof(B[1])
    @test collect(B) == wfs(5:8, 9:12)

    B = A[[true, false, false, true, false]]
    @test eltype(B) == typeof(B[1])
    @test collect(B) == wfs(1:4, 13:16)

    @test_throws BoundsError A[4:6]

    A2 = ArrayOfRDWaveforms((fill(t, 2, 3), ArrayOfSimilarArrays{Int,1}(reshape(1:24, 4, 2, 3))))
    @test A2[CartesianIndex(1, 2)] == RDWaveform(t, 9:12)
    B = A2[1, 2:3]
    @test eltype(B) == typeof(B[1])
    @test collect(B) == wfs(9:12, 17:20)
end
