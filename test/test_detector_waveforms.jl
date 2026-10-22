# This file is a part of RadiationDetectorSignals.jl, licensed under the MIT License (MIT).

using RadiationDetectorSignals
using Test

using Adapt, ArraysOfArrays, FillArrays, JLArrays, Statistics, StructArrays, Unitful


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

# Sample-wise reductions over these signals are exact in floating point.
const REF_TIME = 0:0.5:1.5
const REF_SIGNALS = [[1.0, 2.0, 4.0, 8.0], [3.0, 6.0, 8.0, 16.0], [5.0, 10.0, 12.0, 24.0]]

# The same waveforms with the signals in one block, in one flat buffer, or
# allocated separately:
contiguous_wfs(signals = REF_SIGNALS, time = REF_TIME) =
    ArrayOfRDWaveforms((Fill(time, length(signals)), sliced(reduce(hcat, signals))))
vov_wfs(signals = REF_SIGNALS, time = REF_TIME) =
    ArrayOfRDWaveforms((Fill(time, length(signals)), VectorOfVectors(signals)))
nested_wfs(signals = REF_SIGNALS, time = REF_TIME) =
    ArrayOfRDWaveforms((Fill(time, length(signals)), collect(signals)))


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
    @test_throws MethodError wf_a + 2.5

    # Division acts on the samples themselves, following their promotion rules
    # and avoiding the overflow of a reciprocal:
    @test (RDWaveform(t, Float32.(a)) / 2).signal isa Vector{Float32}
    @test (RDWaveform(0:0, [1e-320]) / 1e-320).signal == [1.0]

    # Units ride along on the samples.
    wf_u = RDWaveform(t, a * u"eV")
    @test @inferred(wf_u + wf_u) == RDWaveform(t, 2 * a * u"eV")
    @test @inferred(2.0 * wf_u) == RDWaveform(t, 2.0 * a * u"eV")

    @test_throws ArgumentError wf_a + RDWaveform(t .+ 1, b)
    @test_throws ArgumentError wf_a - RDWaveform(t .+ 1, b)
    @test_throws ArgumentError wf_a + RDWaveform(0:0.1:12.6, b[1:end-1])
end

@testset "detector_waveform time axis matching" begin
    t = 0:0.1:12.7
    s = rand(128)
    wf = RDWaveform(t, s)

    # Time axes that differ only by rounding match, and the left operand's axis is kept:
    rounded = RDWaveform(collect(t) .+ 1e-14, s)
    @test rounded.time != t
    @test (wf + rounded).time == t
    @test (rounded - wf).time == rounded.time

    t_ns = (0:16:160)u"ns"
    c, d = rand(11), rand(11)
    @test (RDWaveform(t_ns, c) + RDWaveform(collect(t_ns), d)).signal == c + d
    @test_throws ArgumentError RDWaveform(t_ns, c) + RDWaveform(t_ns .+ 1u"ns", d)

    # The tolerance is a small fraction of the sampling step, independent of the
    # length and origin of the axis and of the operand order:
    t32 = range(0f0, step = 16f0, length = 64)
    wf32 = RDWaveform(t32, zeros(Float32, 64))
    wf64 = RDWaveform(range(0.0, step = 16.0, length = 64) .+ 1e-3, zeros(64))
    @test (wf32 + wf64).time == t32
    @test (wf64 + wf32).time == wf64.time
    @test_throws ArgumentError wf32 + RDWaveform(collect(t32) .+ 0.1f0, zeros(Float32, 64))

    long32 = range(0f0, step = 1f0, length = 10_000)
    @test_throws ArgumentError RDWaveform(long32, zeros(Float32, 10_000)) + RDWaveform(long32 .+ 1f0, zeros(Float32, 10_000))
    far = 1e9 .+ (0.0:1.0:3.0)
    @test_throws ArgumentError RDWaveform(far, zeros(4)) + RDWaveform(far .+ 1e-3, zeros(4))

    # Integer axes and single samples match exactly only:
    ticks = 0:100_000_000:300_000_000
    @test_throws ArgumentError RDWaveform(ticks, zeros(4)) + RDWaveform(ticks .+ 1, zeros(4))
    @test_throws ArgumentError RDWaveform([0.0], [1.0]) + RDWaveform([1e-300], [1.0])
end

@testset "RDBaseline" begin
    t = 0:0.1:12.7
    s = rand(128)
    wf = RDWaveform(t, s)
    bl = RDBaseline(2.5)

    @test @inferred(wf + bl) == RDWaveform(t, s .+ 2.5)
    @test @inferred(bl + wf) == wf + bl
    @test @inferred(wf - bl) == RDWaveform(t, s .- 2.5)
    @test @inferred(bl - wf) == RDWaveform(t, 2.5 .- s)
    @test wf + bl - bl ≈ wf

    # Baselines form a vector space as well, and absorb scalars:
    @test -bl == RDBaseline(-2.5)
    @test bl + bl == RDBaseline(5.0)
    @test bl - RDBaseline(1.0) == RDBaseline(1.5)
    @test 2 * bl == bl * 2 == RDBaseline(5.0)
    @test bl / 2 == 2 \ bl == RDBaseline(1.25)
    @test (RDBaseline(1f0) / 2).level isa Float32
    @test bl + 1.0 == 1.0 + bl == RDBaseline(3.5)
    @test bl - 1.0 == RDBaseline(1.5)
    @test 1.0 - bl == RDBaseline(-1.5)

    wf_u = RDWaveform(t, s * u"eV")
    @test (wf_u + RDBaseline(2.5u"eV")).signal == s * u"eV" .+ 2.5u"eV"
    @test @inferred(RDBaseline(2.5u"eV") - wf_u) == RDWaveform(t, 2.5u"eV" .- s * u"eV")

    # The level must be dimensionally compatible with the samples, in either direction.
    @test_throws Unitful.DimensionError wf_u + bl
    @test_throws Unitful.DimensionError bl - wf_u
    @test_throws Unitful.DimensionError wf + RDBaseline(2.5u"eV")
    @test_throws Unitful.DimensionError RDBaseline(2.5u"eV") - wf
end

@testset "detector_waveform broadcasting" begin
    forms = (
        (wfs -> 2.0 .* wfs,              wf -> 2.0 * wf),
        (wfs -> wfs ./ 2.0,              wf -> wf / 2.0),
        (wfs -> .-wfs,                   wf -> -wf),
        (wfs -> wfs .+ RDBaseline(10.0), wf -> wf + RDBaseline(10.0)),
        (wfs -> RDBaseline(10.0) .- wfs, wf -> RDBaseline(10.0) - wf),
        (wfs -> wfs .+ wfs,              wf -> wf + wf),
        (wfs -> @.(2.0 * wfs - wfs / 2.0 + RDBaseline(10.0)), wf -> 2.0 * wf - wf / 2.0 + RDBaseline(10.0)),
    )
    for wfs in (contiguous_wfs(), vov_wfs(), nested_wfs()), (broadcasted, elementwise) in forms
        result = broadcasted(wfs)
        @test result isa ArrayOfRDWaveforms
        @test all(i -> result[i] == elementwise(wfs[i]), eachindex(wfs))
    end

    wfs = contiguous_wfs()
    baselines = RDBaseline.([100.0, 200.0, 300.0])
    shifted = wfs .- baselines
    @test all(i -> shifted[i] == wfs[i] - baselines[i], eachindex(wfs))
    @test sum(2.0 .* wfs) == 2.0 * sum(wfs)
end

@testset "detector_waveform reductions" begin
    # Identical results for all signal storage kinds.
    for wfs in (contiguous_wfs(), vov_wfs(), nested_wfs())
        @test @inferred(sum(wfs)) == RDWaveform(REF_TIME, [9.0, 18.0, 24.0, 48.0])
        @test @inferred(mean(wfs)) == RDWaveform(REF_TIME, [3.0, 6.0, 8.0, 16.0])
        @test @inferred(var(wfs)) == RDWaveform(REF_TIME, [4.0, 16.0, 16.0, 64.0])
        @test @inferred(std(wfs)) == RDWaveform(REF_TIME, [2.0, 4.0, 4.0, 8.0])
    end
    @test var(vov_wfs(), corrected = false) == RDWaveform(REF_TIME, [4.0, 16.0, 16.0, 64.0] * (2 / 3))
    @test std(vov_wfs(), corrected = false) ≈ RDWaveform(REF_TIME, [2.0, 4.0, 4.0, 8.0] * sqrt(2 / 3))
    @test contiguous_wfs().signal isa ArrayOfSimilarArrays
    @test vov_wfs().signal isa VectorOfVectors

    # Against Statistics on the stacked samples:
    signals = [rand(64) for _ in 1:25]
    A = vov_wfs(signals, range(0.0, step = 0.5, length = 64))
    reference = reduce(hcat, signals)
    @test sum(A).signal ≈ vec(sum(reference, dims = 2))
    @test var(A).signal ≈ vec(var(reference, dims = 2))
    @test std(A).signal ≈ vec(std(reference, dims = 2))

    # Time axes held per waveform, equal or differing by rounding only:
    per_row = ArrayOfRDWaveforms((fill(REF_TIME, 3), VectorOfVectors(REF_SIGNALS)))
    @test mean(per_row) == mean(vov_wfs())
    rounded = ArrayOfRDWaveforms(([REF_TIME, collect(REF_TIME) .+ 1e-14, REF_TIME], VectorOfVectors(REF_SIGNALS)))
    @test sum(rounded).signal == sum(vov_wfs()).signal

    mismatched = ArrayOfRDWaveforms(([REF_TIME, REF_TIME, REF_TIME .+ 1], VectorOfVectors(REF_SIGNALS)))
    @test_throws ArgumentError sum(mismatched)
    @test_throws ArgumentError std(mismatched)
    @test_throws ArgumentError sum(contiguous_wfs()[1:0])

    t_ns = REF_TIME * u"ns"
    @test sum(ArrayOfRDWaveforms(([t_ns, collect(t_ns), t_ns], VectorOfVectors(REF_SIGNALS)))) ==
        RDWaveform(t_ns, [9.0, 18.0, 24.0, 48.0])
    @test_throws ArgumentError sum(ArrayOfRDWaveforms(([t_ns, t_ns, t_ns .+ 1u"ns"], VectorOfVectors(REF_SIGNALS))))

    # Units carry through, squared for the variance.
    u_wfs = vov_wfs([s * u"eV" for s in REF_SIGNALS], REF_TIME * u"ns")
    @test mean(u_wfs) == RDWaveform(REF_TIME * u"ns", [3.0, 6.0, 8.0, 16.0] * u"eV")
    @test var(u_wfs) == RDWaveform(REF_TIME * u"ns", [4.0, 16.0, 16.0, 64.0] * u"eV^2")
end

# Accumulation widens sample types narrower than `Int`, as `sum` does. `Int32` is
# narrower only where `Int` is 64 bits; on a 32-bit platform it is `Int` itself and
# `sum` does not widen it either.
@testset "detector_waveform sum widens narrow integers" begin
    narrow_types = Sys.WORD_SIZE > 32 ? (Int16, Int32) : (Int16,)

    for T in narrow_types
        big = T(typemax(T) ÷ 2)
        signals = [T[big, 1, 2, 3] for _ in 1:3]

        for A in (contiguous_wfs(signals), vov_wfs(signals), nested_wfs(signals))
            total, average = sum(A), mean(A)
            @test eltype(total.signal) === Int
            @test total.signal[1] == 3 * Int(big)
            @test eltype(average.signal) === Float64
            @test average.signal[1] == Float64(big)
        end
    end
end

@testset "detector_waveform reductions with mixed sample types" begin
    A = nested_wfs([Int32[1, 2, 3, 4], [1.5, 2.5, 3.5, 4.5]])
    @test sum(A) == RDWaveform(REF_TIME, [2.5, 4.5, 6.5, 8.5])
    @test mean(A) == RDWaveform(REF_TIME, [1.25, 2.25, 3.25, 4.25])
    @test var(A).signal ≈ fill(0.125, 4)
    @test sum(A) == sum(nested_wfs([[1.5, 2.5, 3.5, 4.5], Int32[1, 2, 3, 4]]))
end

@testset "detector_waveform reductions reject ragged signals" begin
    for signals in (VectorOfVectors([[1.0, 2.0], [1.0, 2.0, 3.0]]), [[1.0, 2.0], [1.0, 2.0, 3.0]])
        A = ArrayOfRDWaveforms((Fill(REF_TIME, 2), signals))
        @test_throws DimensionMismatch sum(A)
        @test_throws DimensionMismatch var(A)
    end
end

# JLArrays stands in for a GPU array type: the reductions must work with scalar
# indexing disabled, also on the shape information of a device-resident
# VectorOfVectors. Scalar indexing stays disabled for the rest of the run.
@testset "detector_waveform reductions on a generic array backend (JLArrays)" begin
    JLArrays.allowscalar(false)
    device_wfs = (
        ArrayOfRDWaveforms((Fill(REF_TIME, 3), sliced(JLArray(reduce(hcat, REF_SIGNALS))))),
        ArrayOfRDWaveforms((Fill(REF_TIME, 3), adapt(JLArray, VectorOfVectors(REF_SIGNALS)))),
    )
    for wfs in device_wfs, f in (sum, mean, var, std)
        result = f(wfs).signal
        @test result isa JLArray
        @test Array(result) == f(contiguous_wfs()).signal
    end
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

    V = view(A, 2:3)
    @test eltype(V) == typeof(V[1])
    @test collect(V) == wfs(5:8, 9:12)
    @test_throws BoundsError view(A, 4:6)

    A2 = ArrayOfRDWaveforms((fill(t, 2, 3), ArrayOfSimilarArrays{Int,1}(reshape(1:24, 4, 2, 3))))
    @test A2[CartesianIndex(1, 2)] == RDWaveform(t, 9:12)
    B = A2[1, 2:3]
    @test eltype(B) == typeof(B[1])
    @test collect(B) == wfs(9:12, 17:20)

    # Views alias the parent, A[:] is a writable copy:
    Am = A[:]
    V = view(Am, 2:3)
    Am.signal[2][1] = 0
    @test V[1].signal[1] == 0
end
