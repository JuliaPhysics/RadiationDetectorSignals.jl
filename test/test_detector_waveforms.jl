# This file is a part of RadiationDetectorSignals.jl, licensed under the MIT License (MIT).

using RadiationDetectorSignals
using Test

using ArraysOfArrays, FillArrays, StatsBase, StructArrays, Unitful


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

@testset "detector_waveform addition and subtraction" begin
    timedata = 0:0.1:12.7
    wfdata_a = rand(128)
    wfdata_b = rand(128)

    wf_a = RDWaveform(timedata, wfdata_a)
    wf_b = RDWaveform(timedata, wfdata_b)

    @test @inferred(wf_a + wf_b) == RDWaveform(timedata, wfdata_a + wfdata_b)
    @test @inferred(wf_a - wf_b) == RDWaveform(timedata, wfdata_a - wfdata_b)
    @test @inferred(-wf_a) == RDWaveform(timedata, -wfdata_a)
    @test wf_a - wf_b == wf_a + (-wf_b)

    wf_other_time = RDWaveform(timedata .+ 1, wfdata_b)
    @test_throws ArgumentError wf_a + wf_other_time
    @test_throws ArgumentError wf_a - wf_other_time
end

@testset "detector_waveform addition and subtraction with units" begin
    timedata = (0:0.1:12.7) * u"ns"
    wfdata_a = rand(128) * u"eV"
    wfdata_b = rand(128) * u"eV"

    wf_a = RDWaveform(timedata, wfdata_a)
    wf_b = RDWaveform(timedata, wfdata_b)

    @test @inferred(wf_a + wf_b) == RDWaveform(timedata, wfdata_a + wfdata_b)
    @test @inferred(wf_a - wf_b) == RDWaveform(timedata, wfdata_a - wfdata_b)
    @test @inferred(-wf_a) == RDWaveform(timedata, -wfdata_a)
end

@testset "detector_waveform scalar multiplication and division" begin
    timedata = 0:0.1:12.7
    wfdata = rand(128)
    wf = RDWaveform(timedata, wfdata)

    @test @inferred(2.0 * wf) == RDWaveform(timedata, 2.0 * wfdata)
    @test @inferred(wf * 2.0) == 2.0 * wf
    @test @inferred(wf / 2.0) == RDWaveform(timedata, wfdata / 2.0)
    @test @inferred(2.0 \ wf) == wf / 2.0
end

@testset "detector_waveform scalar multiplication and division with units" begin
    timedata = (0:0.1:12.7) * u"ns"
    wfdata = rand(128) * u"eV"
    wf = RDWaveform(timedata, wfdata)

    @test @inferred(2.0 * wf) == RDWaveform(timedata, 2.0 * wfdata)
    @test @inferred(wf * 2.0) == 2.0 * wf
    @test @inferred(wf / 2.0) == RDWaveform(timedata, wfdata / 2.0)
    @test @inferred(2.0 \ wf) == wf / 2.0
end

@testset "detector_waveform reductions" begin
    timeaxis = 0:0.5:1.5
    # Sample-wise reductions over these are exact in floating point:
    signals = [[1.0, 2.0, 4.0, 8.0], [3.0, 6.0, 8.0, 16.0], [5.0, 10.0, 12.0, 24.0]]
    nwf = length(signals)

    A = ArrayOfRDWaveforms((Fill(timeaxis, nwf), VectorOfVectors(signals)))

    @test sum(A) == RDWaveform(timeaxis, [9.0, 18.0, 24.0, 48.0])
    @test mean(A) == RDWaveform(timeaxis, [3.0, 6.0, 8.0, 16.0])
    @test var(A) == RDWaveform(timeaxis, [4.0, 16.0, 16.0, 64.0])
    @test std(A) == RDWaveform(timeaxis, [2.0, 4.0, 4.0, 8.0])

    # Time axes stored per waveform rather than as a Fill:
    B = ArrayOfRDWaveforms((fill(timeaxis, nwf), VectorOfVectors(signals)))

    @test sum(B) == sum(A)
    @test mean(B) == mean(A)
    @test var(B) == var(A)
    @test std(B) == std(A)

    C = ArrayOfRDWaveforms(([timeaxis, timeaxis, timeaxis .+ 1], VectorOfVectors(signals)))

    @test_throws ArgumentError sum(C)
    @test_throws ArgumentError mean(C)
    @test_throws ArgumentError var(C)
    @test_throws ArgumentError std(C)
end

@testset "detector_waveform reductions with units" begin
    timeaxis = (0:0.5:1.5) * u"ns"
    signals = [[1.0, 2.0, 4.0, 8.0] * u"eV", [3.0, 6.0, 8.0, 16.0] * u"eV", [5.0, 10.0, 12.0, 24.0] * u"eV"]
    nwf = length(signals)

    A = ArrayOfRDWaveforms((Fill(timeaxis, nwf), VectorOfVectors(signals)))

    @test sum(A) == RDWaveform(timeaxis, [9.0, 18.0, 24.0, 48.0] * u"eV")
    @test mean(A) == RDWaveform(timeaxis, [3.0, 6.0, 8.0, 16.0] * u"eV")
    @test var(A) == RDWaveform(timeaxis, [4.0, 16.0, 16.0, 64.0] * u"eV^2")
    @test std(A) == RDWaveform(timeaxis, [2.0, 4.0, 4.0, 8.0] * u"eV")
end

@testset "detector_waveform arithmetic with mixed sample types" begin
    timedata = 0:0.5:1.5

    wf_int = RDWaveform(timedata, Int32[1, 2, 3, 4])
    wf_float = RDWaveform(timedata, [0.5, 0.5, 0.5, 0.5])

    @test wf_int + wf_float == RDWaveform(timedata, [1.5, 2.5, 3.5, 4.5])
    @test eltype((wf_int + wf_float).signal) == Float64
    @test eltype((2.0 * wf_int).signal) == Float64
    @test eltype((wf_int + wf_int).signal) == Int32

    A = ArrayOfRDWaveforms((Fill(timedata, 2), VectorOfVectors([Int32[1, 2, 3, 4], Int32[3, 4, 5, 6]])))

    @test sum(A) == RDWaveform(timedata, Int32[4, 6, 8, 10])
    @test mean(A) == RDWaveform(timedata, [2.0, 3.0, 4.0, 5.0])
    @test eltype(mean(A).signal) == Float64
end

