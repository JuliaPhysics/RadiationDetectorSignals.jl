# This file is a part of RadiationDetectorSignals.jl, licensed under the MIT License (MIT).

using RadiationDetectorSignals
using Test

using ArraysOfArrays, FillArrays, StatsBase, StructArrays, Unitful


# A vector whose indices do not start at one, used to check that sample-wise
# reductions follow the indices of their inputs instead of assuming 1-based signals.
module OffsetSignals
    export ShiftedVector

    struct ShiftedVector{T,P<:AbstractVector{T}} <: AbstractVector{T}
        parent::P
        offset::Int
    end

    Base.size(v::ShiftedVector) = size(v.parent)
    Base.axes(v::ShiftedVector) =
        (Base.IdentityUnitRange(firstindex(v.parent) + v.offset : lastindex(v.parent) + v.offset),)
    Base.IndexStyle(::Type{<:ShiftedVector}) = IndexLinear()
    Base.@propagate_inbounds Base.getindex(v::ShiftedVector, i::Int) = v.parent[i - v.offset]
    Base.@propagate_inbounds Base.setindex!(v::ShiftedVector, x, i::Int) = (v.parent[i - v.offset] = x; v)
    Base.similar(v::ShiftedVector, ::Type{T}) where {T} = ShiftedVector(similar(v.parent, T), v.offset)

    # Keep broadcast results in the wrapper so the offset axes survive.
    struct ShiftedStyle <: Broadcast.AbstractArrayStyle{1} end
    Base.BroadcastStyle(::Type{<:ShiftedVector}) = ShiftedStyle()
    ShiftedStyle(::Val{0}) = ShiftedStyle()
    ShiftedStyle(::Val{1}) = ShiftedStyle()
    function Base.similar(bc::Broadcast.Broadcasted{ShiftedStyle}, ::Type{T}) where {T}
        ax = only(axes(bc))
        ShiftedVector(similar(Array{T}, length(ax)), first(ax) - 1)
    end
end

using .OffsetSignals: ShiftedVector


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

@testset "detector_waveform scalar shifts" begin
    timedata = 0:0.1:12.7
    wfdata = rand(128)
    wf = RDWaveform(timedata, wfdata)

    @test @inferred(wf + 2.5) == RDWaveform(timedata, wfdata .+ 2.5)
    @test @inferred(2.5 + wf) == wf + 2.5
    @test @inferred(wf - 2.5) == RDWaveform(timedata, wfdata .- 2.5)
    @test @inferred(2.5 - wf) == RDWaveform(timedata, 2.5 .- wfdata)
    @test (wf + 2.5).time == wf.time
    @test wf + 2.5 - 2.5 ≈ wf
end

@testset "detector_waveform scalar shifts with units" begin
    timedata = (0:0.1:12.7) * u"ns"
    wfdata = rand(128) * u"eV"
    wf = RDWaveform(timedata, wfdata)

    @test @inferred(wf + 2.5u"eV") == RDWaveform(timedata, wfdata .+ 2.5u"eV")
    @test @inferred(2.5u"eV" + wf) == wf + 2.5u"eV"
    @test @inferred(wf - 2.5u"eV") == RDWaveform(timedata, wfdata .- 2.5u"eV")

    # A plain number shifts the samples in the unit they already carry.
    @test @inferred(wf + 2.5) == wf + 2.5u"eV"
    @test @inferred(2.5 + wf) == wf + 2.5u"eV"
    @test @inferred(wf - 2.5) == wf - 2.5u"eV"
    @test @inferred(2.5 - wf) == 2.5u"eV" - wf
    @test wf + 2 == wf + 2u"eV"
end

@testset "detector_waveform scalar shifts mixing units" begin
    timedata = 0:0.1:12.7
    wfdata = rand(128)
    wf = RDWaveform(timedata, wfdata)

    # Samples without a unit take the unit of the shift.
    @test (wf + 2.5u"eV").signal == wfdata * u"eV" .+ 2.5u"eV"
    @test (2.5u"eV" + wf).signal == wfdata * u"eV" .+ 2.5u"eV"
    @test (wf - 2.5u"eV").signal == wfdata * u"eV" .- 2.5u"eV"
    @test (2.5u"eV" - wf).signal == 2.5u"eV" .- wfdata * u"eV"
    @test (wf + 2.5u"eV").time == wf.time

    # Both sides unitless stays unitless.
    @test (wf + 2.5).signal == wfdata .+ 2.5
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

@testset "detector_waveform reductions are storage independent" begin
    timeaxis = 0:0.5:1.5
    signals = [[1.0, 2.0, 4.0, 8.0], [3.0, 6.0, 8.0, 16.0], [5.0, 10.0, 12.0, 24.0]]
    nwf = length(signals)

    contiguous = ArrayOfRDWaveforms((Fill(timeaxis, nwf), nestedview(reduce(hcat, signals))))
    ragged = ArrayOfRDWaveforms((Fill(timeaxis, nwf), VectorOfVectors(signals)))

    @test contiguous.signal isa ArrayOfSimilarArrays
    @test sum(contiguous) == sum(ragged)
    @test mean(contiguous) == mean(ragged)
    @test var(contiguous) == var(ragged)
    @test std(contiguous) == std(ragged)
end

@testset "detector_waveform reductions against Statistics" begin
    nwf, nsamples = 25, 64
    timeaxis = range(0.0, step = 0.5, length = nsamples)
    signals = [rand(nsamples) for _ in 1:nwf]
    A = ArrayOfRDWaveforms((Fill(timeaxis, nwf), VectorOfVectors(signals)))

    reference = reduce(hcat, signals)
    @test sum(A).signal ≈ vec(sum(reference, dims = 2))
    @test mean(A).signal ≈ vec(mean(reference, dims = 2))
    @test var(A).signal ≈ vec(var(reference, dims = 2))
    @test std(A).signal ≈ vec(std(reference, dims = 2))
end

@testset "detector_waveform sum widens narrow integers" begin
    timeaxis = 0:0.5:1.5
    big = typemax(Int32) ÷ 2
    signals = [Int32[big, 1, 2, 3], Int32[big, 1, 2, 3], Int32[big, 1, 2, 3]]
    A = ArrayOfRDWaveforms((Fill(timeaxis, 3), VectorOfVectors(signals)))

    @test eltype(sum(A).signal) === Int
    @test sum(A).signal[1] == 3 * Int(big)
    @test eltype(mean(A).signal) === Float64
end

@testset "detector_waveform reductions honor signal axes" begin
    timeaxis = 0:0.5:1.5
    signals = [ShiftedVector([1.0, 2.0, 4.0, 8.0], 2),
               ShiftedVector([3.0, 6.0, 8.0, 16.0], 2),
               ShiftedVector([5.0, 10.0, 12.0, 24.0], 2)]
    A = ArrayOfRDWaveforms((Fill(timeaxis, length(signals)), signals))

    expected_axes = axes(first(signals))
    @test axes(sum(A).signal) == expected_axes
    @test axes(mean(A).signal) == expected_axes
    @test axes(var(A).signal) == expected_axes
    @test axes(std(A).signal) == expected_axes

    @test collect(sum(A).signal) == [9.0, 18.0, 24.0, 48.0]
    @test collect(mean(A).signal) == [3.0, 6.0, 8.0, 16.0]
    @test collect(var(A).signal) == [4.0, 16.0, 16.0, 64.0]

    mismatched = ArrayOfRDWaveforms((Fill(timeaxis, 2),
        [ShiftedVector([1.0, 2.0, 4.0, 8.0], 2), ShiftedVector([1.0, 2.0, 4.0, 8.0], 0)]))
    @test_throws DimensionMismatch sum(mismatched)
    @test_throws DimensionMismatch mean(mismatched)
    @test_throws DimensionMismatch var(mismatched)
end

@testset "detector_waveform broadcasting" begin
    timeaxis = 0:0.5:1.5
    signals = [[1.0, 2.0, 4.0, 8.0], [3.0, 6.0, 8.0, 16.0], [5.0, 10.0, 12.0, 24.0]]
    nwf = length(signals)

    contiguous = ArrayOfRDWaveforms((Fill(timeaxis, nwf), nestedview(reduce(hcat, signals))))
    ragged = ArrayOfRDWaveforms((Fill(timeaxis, nwf), VectorOfVectors(signals)))

    for wfs in (contiguous, ragged)
        @test all(i -> (2.0 .* wfs)[i] == 2.0 * wfs[i], eachindex(wfs))
        @test all(i -> (wfs .* 2.0)[i] == 2.0 * wfs[i], eachindex(wfs))
        @test all(i -> (wfs ./ 2.0)[i] == wfs[i] / 2.0, eachindex(wfs))
        @test all(i -> (2.0 .\ wfs)[i] == wfs[i] / 2.0, eachindex(wfs))
        @test all(i -> (.-wfs)[i] == -wfs[i], eachindex(wfs))
        @test all(i -> (wfs .+ wfs)[i] == wfs[i] + wfs[i], eachindex(wfs))
        @test all(i -> (wfs .- wfs)[i] == wfs[i] - wfs[i], eachindex(wfs))
    end

    # Contiguous sample storage must survive broadcasting, so that reductions and
    # further operations keep operating on a single block of memory.
    @test (2.0 .* contiguous).signal isa ArrayOfSimilarArrays
    @test (contiguous .* 2.0).signal isa ArrayOfSimilarArrays
    @test (contiguous ./ 2.0).signal isa ArrayOfSimilarArrays
    @test (2.0 .\ contiguous).signal isa ArrayOfSimilarArrays
    @test (.-contiguous).signal isa ArrayOfSimilarArrays
    @test (contiguous .+ contiguous).signal isa ArrayOfSimilarArrays
    @test (contiguous .- contiguous).signal isa ArrayOfSimilarArrays

    @test (2.0 .* contiguous).time == contiguous.time

    other_time = ArrayOfRDWaveforms((Fill(timeaxis .+ 1, nwf), nestedview(reduce(hcat, signals))))
    @test_throws ArgumentError contiguous .+ other_time
    @test_throws ArgumentError contiguous .- other_time
end

@testset "detector_waveform broadcast shifts" begin
    timeaxis = 0:0.5:1.5
    signals = [[1.0, 2.0, 4.0, 8.0], [3.0, 6.0, 8.0, 16.0], [5.0, 10.0, 12.0, 24.0]]
    nwf = length(signals)

    contiguous = ArrayOfRDWaveforms((Fill(timeaxis, nwf), nestedview(reduce(hcat, signals))))
    ragged = ArrayOfRDWaveforms((Fill(timeaxis, nwf), VectorOfVectors(signals)))

    # One shift applied to every waveform.
    for wfs in (contiguous, ragged)
        @test all(i -> (wfs .+ 10.0)[i] == wfs[i] + 10.0, eachindex(wfs))
        @test all(i -> (10.0 .+ wfs)[i] == wfs[i] + 10.0, eachindex(wfs))
        @test all(i -> (wfs .- 10.0)[i] == wfs[i] - 10.0, eachindex(wfs))
        @test all(i -> (10.0 .- wfs)[i] == 10.0 - wfs[i], eachindex(wfs))
    end

    # One shift per waveform, as when subtracting per-waveform baselines.
    shifts = [100.0, 200.0, 300.0]
    for wfs in (contiguous, ragged)
        @test all(i -> (wfs .+ shifts)[i] == wfs[i] + shifts[i], eachindex(wfs))
        @test all(i -> (shifts .+ wfs)[i] == wfs[i] + shifts[i], eachindex(wfs))
        @test all(i -> (wfs .- shifts)[i] == wfs[i] - shifts[i], eachindex(wfs))
        @test all(i -> (shifts .- wfs)[i] == shifts[i] - wfs[i], eachindex(wfs))
    end

    @test (contiguous .+ 10.0).signal isa ArrayOfSimilarArrays
    @test (10.0 .- contiguous).signal isa ArrayOfSimilarArrays
    @test (contiguous .+ shifts).signal isa ArrayOfSimilarArrays
    @test (contiguous .- shifts).signal isa ArrayOfSimilarArrays
    @test (contiguous .+ shifts).time == contiguous.time

    @test_throws DimensionMismatch contiguous .+ [1.0, 2.0]
    @test_throws DimensionMismatch contiguous .- [1.0, 2.0]

    # Unitless samples take the unit of a unitful shift, scalar or per waveform.
    for wfs in (contiguous, ragged)
        @test all(i -> (wfs .+ 10.0u"eV")[i] == wfs[i] + 10.0u"eV", eachindex(wfs))
        @test all(i -> (10.0u"eV" .- wfs)[i] == 10.0u"eV" - wfs[i], eachindex(wfs))
        @test all(i -> (wfs .- shifts * u"eV")[i] == wfs[i] - shifts[i] * u"eV", eachindex(wfs))
    end
    @test (contiguous .+ 10.0u"eV").signal isa ArrayOfSimilarArrays
    @test (contiguous .- shifts * u"eV").signal isa ArrayOfSimilarArrays
end

@testset "detector_waveform broadcast shifts with units" begin
    timeaxis = (0:0.5:1.5) * u"ns"
    signals = [[1.0, 2.0, 4.0, 8.0] * u"eV", [3.0, 6.0, 8.0, 16.0] * u"eV"]
    wfs = ArrayOfRDWaveforms((Fill(timeaxis, length(signals)), nestedview(reduce(hcat, signals))))
    shifts = [10.0, 20.0] * u"eV"

    @test all(i -> (wfs .+ 10.0u"eV")[i] == wfs[i] + 10.0u"eV", eachindex(wfs))
    @test all(i -> (wfs .- shifts)[i] == wfs[i] - shifts[i], eachindex(wfs))
    @test (wfs .+ shifts).signal isa ArrayOfSimilarArrays

    # Plain numbers, scalar or one per waveform, shift in the samples' unit.
    @test all(i -> (wfs .+ 10.0)[i] == wfs[i] + 10.0u"eV", eachindex(wfs))
    @test all(i -> (10.0 .- wfs)[i] == 10.0u"eV" - wfs[i], eachindex(wfs))
    @test all(i -> (wfs .- [10.0, 20.0])[i] == wfs[i] - shifts[i], eachindex(wfs))
    @test (wfs .- [10.0, 20.0]).signal isa ArrayOfSimilarArrays
end

@testset "detector_waveform broadcasting with units" begin
    timeaxis = (0:0.5:1.5) * u"ns"
    signals = [[1.0, 2.0, 4.0, 8.0] * u"eV", [3.0, 6.0, 8.0, 16.0] * u"eV"]
    wfs = ArrayOfRDWaveforms((Fill(timeaxis, length(signals)), nestedview(reduce(hcat, signals))))

    @test all(i -> (2.0 .* wfs)[i] == 2.0 * wfs[i], eachindex(wfs))
    @test all(i -> (wfs ./ 2.0)[i] == wfs[i] / 2.0, eachindex(wfs))
    @test all(i -> (wfs .+ wfs)[i] == wfs[i] + wfs[i], eachindex(wfs))
    @test (2.0 .* wfs).signal isa ArrayOfSimilarArrays
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

