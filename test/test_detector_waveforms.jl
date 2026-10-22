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
