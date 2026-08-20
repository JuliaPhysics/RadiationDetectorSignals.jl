# This file is a part of RadiationDetectorSignals.jl, licensed under the MIT License (MIT).

using RadiationDetectorSignals
using Test

using ArraysOfArrays, FillArrays, JLArrays, OffsetArrays, Statistics, StructArrays, Unitful


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

# Sample-wise reductions over these signals are exact in floating point.
const REF_TIME = 0:0.5:1.5
const REF_SIGNALS = [[1.0, 2.0, 4.0, 8.0], [3.0, 6.0, 8.0, 16.0], [5.0, 10.0, 12.0, 24.0]]

contiguous_wfs(signals = REF_SIGNALS, time = REF_TIME) =
    ArrayOfRDWaveforms((Fill(time, length(signals)), nestedview(reduce(hcat, signals))))
ragged_wfs(signals = REF_SIGNALS, time = REF_TIME) =
    ArrayOfRDWaveforms((Fill(time, length(signals)), VectorOfVectors(signals)))


@testset "detector_waveform arithmetic" begin
    t = 0:0.1:12.7
    a, b = rand(128), rand(128)
    wf_a, wf_b = RDWaveform(t, a), RDWaveform(t, b)

    @test @inferred(wf_a + wf_b) == RDWaveform(t, a + b)
    @test @inferred(wf_a - wf_b) == RDWaveform(t, a - b)
    @test @inferred(-wf_a) == RDWaveform(t, -a)
    @test @inferred(2.0 * wf_a) == RDWaveform(t, 2.0 * a)
    @test @inferred(wf_a / 2.0) == RDWaveform(t, a / 2.0)
    @test wf_a * 2.0 == 2.0 * wf_a
    @test 2.0 \ wf_a == wf_a / 2.0
    @test wf_a - wf_b == wf_a + (-wf_b)
    @test all(w -> w.time == t, (wf_a + wf_b, -wf_a, 2.0 * wf_a, wf_a / 2.0))

    # Units ride along on the samples.
    wf_u = RDWaveform(t, a * u"eV")
    @test @inferred(wf_u + wf_u) == RDWaveform(t, 2 * a * u"eV")
    @test @inferred(2.0 * wf_u) == RDWaveform(t, 2.0 * a * u"eV")

    other_time = RDWaveform(t .+ 1, b)
    @test_throws ArgumentError wf_a + other_time
    @test_throws ArgumentError wf_a - other_time
end

@testset "detector_waveform scalar shifts" begin
    t = 0:0.1:12.7
    s = rand(128)
    wf = RDWaveform(t, s)
    wf_u = RDWaveform(t, s * u"eV")

    @test @inferred(wf + 2.5) == RDWaveform(t, s .+ 2.5)
    @test @inferred(2.5 + wf) == wf + 2.5
    @test @inferred(wf - 2.5) == RDWaveform(t, s .- 2.5)
    @test @inferred(2.5 - wf) == RDWaveform(t, 2.5 .- s)
    @test (wf + 2.5).time == t
    @test wf + 2.5 - 2.5 ≈ wf

    # A plain shift is interpreted in unitful samples' own unit.
    @test (wf_u + 2.5u"eV").signal == s * u"eV" .+ 2.5u"eV"
    @test (wf_u + 2.5).signal == s * u"eV" .+ 2.5u"eV"
    @test wf_u + 2 == wf_u + 2u"eV"

    # Plain samples do not adopt a unitful shift's unit: no reverse inference.
    @test_throws Unitful.DimensionError wf + 2.5u"eV"
    @test_throws Unitful.DimensionError 2.5u"eV" - wf
end

@testset "detector_waveform reductions" begin
    # Identical results whether the samples are one block or separately allocated.
    for wfs in (contiguous_wfs(), ragged_wfs())
        @test sum(wfs) == RDWaveform(REF_TIME, [9.0, 18.0, 24.0, 48.0])
        @test mean(wfs) == RDWaveform(REF_TIME, [3.0, 6.0, 8.0, 16.0])
        @test var(wfs) == RDWaveform(REF_TIME, [4.0, 16.0, 16.0, 64.0])
        @test std(wfs) == RDWaveform(REF_TIME, [2.0, 4.0, 4.0, 8.0])
    end
    @test contiguous_wfs().signal isa ArrayOfSimilarArrays

    # Time axes held per waveform rather than shared as a Fill.
    per_row = ArrayOfRDWaveforms((fill(REF_TIME, 3), VectorOfVectors(REF_SIGNALS)))
    @test mean(per_row) == mean(ragged_wfs())

    mismatched = ArrayOfRDWaveforms(([REF_TIME, REF_TIME, REF_TIME .+ 1], VectorOfVectors(REF_SIGNALS)))
    @test_throws ArgumentError sum(mismatched)
    @test_throws ArgumentError std(mismatched)

    # Units carry through, squared for the variance.
    u_wfs = ragged_wfs([s * u"eV" for s in REF_SIGNALS], REF_TIME * u"ns")
    @test mean(u_wfs) == RDWaveform(REF_TIME * u"ns", [3.0, 6.0, 8.0, 16.0] * u"eV")
    @test var(u_wfs) == RDWaveform(REF_TIME * u"ns", [4.0, 16.0, 16.0, 64.0] * u"eV^2")
end

@testset "detector_waveform reductions against Statistics" begin
    nwf, nsamples = 25, 64
    signals = [rand(nsamples) for _ in 1:nwf]
    A = ragged_wfs(signals, range(0.0, step = 0.5, length = nsamples))
    reference = reduce(hcat, signals)

    @test sum(A).signal ≈ vec(sum(reference, dims = 2))
    @test mean(A).signal ≈ vec(mean(reference, dims = 2))
    @test var(A).signal ≈ vec(var(reference, dims = 2))
    @test std(A).signal ≈ vec(std(reference, dims = 2))
end

@testset "detector_waveform sum widens narrow integers" begin
    big = typemax(Int32) ÷ 2
    A = ragged_wfs([Int32[big, 1, 2, 3] for _ in 1:3])

    @test eltype(sum(A).signal) === Int
    @test sum(A).signal[1] == 3 * Int(big)
    @test eltype(mean(A).signal) === Float64
end

@testset "detector_waveform reductions honor signal axes" begin
    signals = [OffsetVector(s, 2) for s in REF_SIGNALS]
    A = ArrayOfRDWaveforms((Fill(REF_TIME, length(signals)), signals))
    expected_axes = axes(first(signals))

    @test all(f -> axes(f(A).signal) == expected_axes, (sum, mean, var, std))
    @test collect(sum(A).signal) == [9.0, 18.0, 24.0, 48.0]
    @test collect(var(A).signal) == [4.0, 16.0, 16.0, 64.0]

    mismatched = ArrayOfRDWaveforms((Fill(REF_TIME, 2),
        [OffsetVector(REF_SIGNALS[1], 2), OffsetVector(REF_SIGNALS[1], 0)]))
    @test_throws DimensionMismatch sum(mismatched)
    @test_throws DimensionMismatch var(mismatched)
end

# Each broadcast form, paired with the same operation applied waveform by waveform.
const BROADCAST_FORMS = (
    (wfs -> 2.0 .* wfs,  wf -> 2.0 * wf),
    (wfs -> wfs .* 2.0,  wf -> 2.0 * wf),
    (wfs -> wfs ./ 2.0,  wf -> wf / 2.0),
    (wfs -> 2.0 .\ wfs,  wf -> wf / 2.0),
    (wfs -> .-wfs,       wf -> -wf),
    (wfs -> wfs .+ 10.0, wf -> wf + 10.0),
    (wfs -> 10.0 .+ wfs, wf -> wf + 10.0),
    (wfs -> wfs .- 10.0, wf -> wf - 10.0),
    (wfs -> 10.0 .- wfs, wf -> 10.0 - wf),
    (wfs -> wfs .+ wfs,  wf -> wf + wf),
    (wfs -> wfs .- wfs,  wf -> wf - wf),
)

@testset "detector_waveform broadcasting" begin
    wfs = contiguous_wfs()
    for (broadcasted, elementwise) in BROADCAST_FORMS
        result = broadcasted(wfs)
        @test all(i -> result[i] == elementwise(wfs[i]), eachindex(wfs))
        # Contiguous sample storage must survive, so that reductions and further
        # operations keep working on a single block of memory.
        @test result.signal isa ArrayOfSimilarArrays
    end
    @test (2.0 .* wfs).time == wfs.time

    # Separately allocated signals take a different path through the same operators.
    ragged = ragged_wfs()
    for (broadcasted, elementwise) in BROADCAST_FORMS[[1, 6, 10]]
        @test all(i -> broadcasted(ragged)[i] == elementwise(ragged[i]), eachindex(ragged))
    end

    other_time = contiguous_wfs(REF_SIGNALS, REF_TIME .+ 1)
    @test_throws ArgumentError wfs .+ other_time
    @test_throws ArgumentError wfs .- other_time
end

@testset "detector_waveform broadcast shifts per waveform" begin
    shifts = [100.0, 200.0, 300.0]
    for wfs in (contiguous_wfs(), ragged_wfs())
        @test all(i -> (wfs .+ shifts)[i] == wfs[i] + shifts[i], eachindex(wfs))
        @test all(i -> (shifts .+ wfs)[i] == wfs[i] + shifts[i], eachindex(wfs))
        @test all(i -> (wfs .- shifts)[i] == wfs[i] - shifts[i], eachindex(wfs))
        @test all(i -> (shifts .- wfs)[i] == shifts[i] - wfs[i], eachindex(wfs))
    end

    wfs = contiguous_wfs()
    @test (wfs .- shifts).signal isa ArrayOfSimilarArrays
    @test (wfs .+ shifts).time == wfs.time
    @test_throws DimensionMismatch wfs .+ [1.0, 2.0]
end

@testset "detector_waveform broadcasting with units" begin
    u_wfs = contiguous_wfs([s * u"eV" for s in REF_SIGNALS], REF_TIME * u"ns")
    plain = contiguous_wfs()
    shifts = [10.0, 20.0, 30.0]

    @test all(i -> (2.0 .* u_wfs)[i] == 2.0 * u_wfs[i], eachindex(u_wfs))
    @test all(i -> (u_wfs .+ u_wfs)[i] == u_wfs[i] + u_wfs[i], eachindex(u_wfs))
    @test (2.0 .* u_wfs).signal isa ArrayOfSimilarArrays

    # A plain shift, scalar or per waveform, is interpreted in unitful samples' own unit.
    @test all(i -> (u_wfs .+ 10.0)[i] == u_wfs[i] + 10.0u"eV", eachindex(u_wfs))
    @test all(i -> (u_wfs .- shifts)[i] == u_wfs[i] - shifts[i] * u"eV", eachindex(u_wfs))

    # Plain samples do not adopt a unitful shift's unit: no reverse inference.
    @test_throws Unitful.DimensionError plain .+ 10.0u"eV"
    @test_throws Unitful.DimensionError plain .- shifts * u"eV"
end

@testset "detector_waveform arithmetic with mixed sample types" begin
    t = REF_TIME
    wf_int = RDWaveform(t, Int32[1, 2, 3, 4])
    wf_float = RDWaveform(t, [0.5, 0.5, 0.5, 0.5])

    @test wf_int + wf_float == RDWaveform(t, [1.5, 2.5, 3.5, 4.5])
    @test eltype((wf_int + wf_float).signal) == Float64
    @test eltype((2.0 * wf_int).signal) == Float64
    @test eltype((wf_int + wf_int).signal) == Int32

    A = ragged_wfs([Int32[1, 2, 3, 4], Int32[3, 4, 5, 6]])
    @test sum(A) == RDWaveform(t, Int32[4, 6, 8, 10])
    @test mean(A) == RDWaveform(t, [2.0, 3.0, 4.0, 5.0])
    @test eltype(mean(A).signal) == Float64
end

@testset "detector_waveform ragged reductions with heterogeneous sample types" begin
    # The accumulator's element type must come from every signal, not just the first:
    # an Int32-typed accumulator sized to the first signal would throw an InexactError
    # (or worse, silently round) on a later, non-integral Float64 signal.
    A = ragged_wfs([Int32[1, 2, 3, 4], [1.5, 2.5, 3.5, 4.5]])

    @test sum(A) == RDWaveform(REF_TIME, [2.5, 4.5, 6.5, 8.5])
    @test eltype(sum(A).signal) == Float64
    @test mean(A) == RDWaveform(REF_TIME, [1.25, 2.25, 3.25, 4.25])
    @test var(A).signal ≈ fill(0.125, 4)
    @test std(A).signal ≈ fill(sqrt(0.125), 4)

    # Order shouldn't matter: the narrower type first is the case that used to break.
    B = ragged_wfs([[1.5, 2.5, 3.5, 4.5], Int32[1, 2, 3, 4]])
    @test sum(A) == sum(B)
end

# JLArrays.jl provides an AbstractArray backend that is not `Array`, with no special
# casing for it anywhere in Base or this package's code — the same property a real GPU
# array type (CuArray, ROCArray, ...) has. It stands in for one here so the
# ArrayOfSimilarVectors/flatview-specialized paths (reductions and broadcasting) are
# checked against a non-Array backend without requiring GPU hardware.
jl_wfs(signals = REF_SIGNALS, time = REF_TIME) =
    ArrayOfRDWaveforms((Fill(time, length(signals)), nestedview(JLArray(reduce(hcat, signals)))))

@testset "detector_waveform generic array backend (JLArrays)" begin
    wfs = jl_wfs()
    ref = contiguous_wfs()  # identical data, Array-backed

    @test wfs.signal isa ArrayOfSimilarVectors
    @test flatview(wfs.signal) isa JLArray

    @testset "reductions" begin
        for f in (sum, mean, var, std)
            @test Array(f(wfs).signal) == f(ref).signal
        end
    end

    @testset "broadcasting preserves JLArray storage" begin
        for (broadcasted, _) in BROADCAST_FORMS
            result = broadcasted(wfs)
            @test flatview(result.signal) isa JLArray
            @test Array(flatview(result.signal)) == flatview(broadcasted(ref).signal)
        end
    end

    @testset "per-waveform shifts preserve JLArray storage" begin
        shifts = [100.0, 200.0, 300.0]
        result = wfs .+ JLArray(shifts)
        @test flatview(result.signal) isa JLArray
        @test Array(flatview(result.signal)) == flatview((ref .+ shifts).signal)
    end
end
