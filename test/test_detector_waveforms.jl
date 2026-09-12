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


@testset "detector_waveforms broadcast" begin
    nwf = 5
    timedata = Fill((0:0.1:12.7) * u"ns", nwf)
    stats(wf) = (mean = sum(wf.signal) / length(wf.signal), t_last = last(wf.time))

    for signal in (
        VectorOfSimilarVectors(rand(128, nwf)),
        VectorOfVectors([rand(128) for _ in 1:nwf]),
    )
        A = ArrayOfRDWaveforms((timedata, signal))
        S = stats.(A)
        @test S isa StructArray
        @test S.mean == [stats(wf).mean for wf in A]
        @test S.t_last == fill(12.7u"ns", nwf)
        @test (wf -> length(wf.signal)).(A) == fill(128, nwf)
        @test (wf -> 2 .* wf.signal).(A) == [2 .* wf.signal for wf in A]
    end

    # Element type may be less specific than the stored columns:
    signal = VectorOfSimilarVectors(rand(128, nwf))
    B = StructArray{RDWaveform{eltype(eltype(timedata)),Float64,eltype(timedata),Vector{Float64}}}((timedata, signal))
    @test !(typeof(B[1]) <: eltype(B))
    @test stats.(B) isa StructArray
end # testset
