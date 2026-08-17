# This file is a part of RadiationDetectorSignals.jl, licensed under the MIT License (MIT).

# TODO: Replace by true custom types to avoid type piracy (e.g. with plotting
# recipes) on NamedTuple and TypedTables.Table (even though the types aliased
# here are quite specific, so the risk of a conflict is low).


const WaveformSamples{T<:RealQuantity} = AbstractVector{T}
const TimeAxis{T<:RealQuantity} = AbstractVector{T}


"""
    RDWaveform

Represents a radiation detector signal waveform.

Fields:

* `time`: time axis, typically a range
* `signal`: detector signal values

Use [`ArrayOfRDWaveforms`](@ref) for arrays of `RDWaveform` that have a
compact memory layout.
"""
struct RDWaveform{
    T<:RealQuantity,U<:RealQuantity,
    TV<:TimeAxis{T},UV<:WaveformSamples{U},
}
    time::TV
    signal::UV
end

export RDWaveform


RDWaveform{T,U,TV,UV}(wf::RDWaveform) where {T,U,TV,UV} = RDWaveform{T,U,TV,UV}(wf.time, wf.signal)
Base.convert(::Type{RDWaveform{T,U,TV,UV}}, wf::RDWaveform) where {T,U,TV,UV} = RDWaveform{T,U,TV,UV}(wf)

Base.:(==)(a::RDWaveform, b::RDWaveform) = a.time == b.time && a.signal == b.signal
Base.isapprox(a::RDWaveform, b::RDWaveform; kwargs...) = isapprox(a.time, b.time; kwargs...) && isapprox(a.signal, b.signal; kwargs...)

Base.float(wf::RDWaveform) = RDWaveform(float(wf.time), float(wf.signal))

"""
    +(a::RDWaveform, b::RDWaveform)

Sample-wise sum of two waveforms that share the same time axis.

Throws an `ArgumentError` if `a` and `b` have different time axes.
"""
function Base.:(+)(a::RDWaveform, b::RDWaveform)
    a.time == b.time || throw(ArgumentError("Can't add RDWaveform with different time axes"))
    RDWaveform(a.time, a.signal + b.signal)
end

"""
    -(a::RDWaveform, b::RDWaveform)

Sample-wise difference of two waveforms that share the same time axis.

Throws an `ArgumentError` if `a` and `b` have different time axes.
"""
function Base.:(-)(a::RDWaveform, b::RDWaveform)
    a.time == b.time || throw(ArgumentError("Can't subtract RDWaveform with different time axes"))
    RDWaveform(a.time, a.signal - b.signal)
end

"""
    -(a::RDWaveform)

Negate a waveform's samples, keeping its time axis.
"""
Base.:(-)(a::RDWaveform) = RDWaveform(a.time, -a.signal)

# Shifting mixes samples and a shift amount, either of which may carry a unit.
# Whichever side has none is taken to be expressed in the unit of the other, so
# unitful samples accept a plain shift and plain samples accept a unitful one.
_matching_shift(a, ::Type) = a
_matching_shift(a::Real, ::Type{T}) where {T<:Quantity} = a * unit(T)
_matching_shift(a::AbstractArray{<:Real}, ::Type{T}) where {T<:Quantity} = a * unit(T)

_matching_samples(x, a) = x
_matching_samples(x::AbstractArray{<:Real}, a::Quantity) = x * unit(a)
_matching_samples(x::AbstractArray{<:Real}, a::AbstractArray{<:Quantity}) = x * unit(eltype(a))

_shift_op(f, x, a) = f(_matching_samples(x, a), _matching_shift(a, eltype(x)))

"""
    +(wf::RDWaveform, a::RealQuantity)
    +(a::RealQuantity, wf::RDWaveform)

Shift every sample of a waveform by `a`, keeping its time axis.

Whichever of `wf`'s samples and `a` carries no unit takes on the other's: a
plain number shifts unitful samples in their own unit, and a unitful `a`
gives plain samples that unit. If neither carries a unit, the shift is plain.
"""
Base.:(+)(wf::RDWaveform, a::RealQuantity) =
    RDWaveform(wf.time, _shift_op((x, s) -> x .+ s, wf.signal, a))
Base.:(+)(a::RealQuantity, wf::RDWaveform) = wf + a

"""
    -(wf::RDWaveform, a::RealQuantity)

Shift every sample of a waveform by `-a`, keeping its time axis.

Whichever of `wf`'s samples and `a` carries no unit takes on the other's: a
plain number shifts unitful samples in their own unit, and a unitful `a`
gives plain samples that unit. If neither carries a unit, the shift is plain.
"""
Base.:(-)(wf::RDWaveform, a::RealQuantity) =
    RDWaveform(wf.time, _shift_op((x, s) -> x .- s, wf.signal, a))

"""
    -(a::RealQuantity, wf::RDWaveform)

Subtract every sample of a waveform from `a`, keeping its time axis.

Whichever of `wf`'s samples and `a` carries no unit takes on the other's: a
plain `a` is taken in the samples' own unit, and a unitful `a` gives plain
samples that unit. If neither carries a unit, the result is plain.
"""
Base.:(-)(a::RealQuantity, wf::RDWaveform) =
    RDWaveform(wf.time, _shift_op((x, s) -> s .- x, wf.signal, a))

"""
    *(a::Real, b::RDWaveform)
    *(a::RDWaveform, b::Real)

Scale a waveform's samples by a scalar, keeping its time axis.
"""
Base.:(*)(a::Real, b::RDWaveform) = RDWaveform(b.time, a * b.signal)
Base.:(*)(a::RDWaveform, b::Real) = b * a

"""
    /(a::RDWaveform, b::Real)

Divide a waveform's samples by a scalar, keeping its time axis.
"""
Base.:(/)(a::RDWaveform, b::Real) = a * inv(b)

"""
    \\(a::Real, b::RDWaveform)

Divide a waveform's samples by a scalar, keeping its time axis.

Equivalent to `b / a`.
"""
Base.:(\)(a::Real, b::RDWaveform) = b / a

# ToDo: function for waveform duration. Use IntervalSets.duration?


"""
    ArrayOfRDWaveforms = StructArray{<:RDWaveform, ...)

A `StructsArrays.StructArray` of [`RDWaveform`](@ref).

By default, uses `ArraysOfArrays.VectorOfVectors` for contiguous memory
layout.
"""
const ArrayOfRDWaveforms{
    T<:RealQuantity,U<:RealQuantity,N,
    VVT <: AbstractVector{<:AbstractVector{T}}, VVU <: AbstractVector{<:AbstractVector{U}}
} = StructArray{
    <:RDWaveform{T,U},
    N,
    NamedTuple{(:time, :signal), Tuple{VVT,VVU}}
}

export ArrayOfRDWaveforms


function StructArray{RDWaveform}(
    contents::Tuple{
        AbstractArray{<:AbstractVector{<:RealQuantity},N},
        AbstractArray{<:AbstractVector{<:RealQuantity},N}
    }
) where {N}
    time, signal = contents
    VT = eltype(time)
    VU = eltype(signal)
    T = eltype(VT)
    U = eltype(VU)
    StructArray{RDWaveform{T,U,VT,VU}}((time, signal))
end


StructArray{RDWaveform}(waveforms::AbstractVector{<:RDWaveform}) =
    StructArray{RDWaveform}((map(w -> w.time, waveforms), VectorOfVectors(map(w -> w.signal, waveforms))))

Base.convert(::Type{ArrayOfRDWaveforms}, waveforms::AbstractVector{<:RDWaveform}) = StructArray{RDWaveform}(waveforms)
Base.convert(::Type{ArrayOfRDWaveforms}, waveforms::StructArray{<:RDWaveform}) = waveforms

Base.convert(::Type{StructArray{RDWaveform}}, waveforms::AbstractVector{<:RDWaveform}) = StructArray{RDWaveform}(waveforms)
Base.convert(::Type{StructArray{RDWaveform}}, waveforms::StructArray{<:RDWaveform}) = waveforms

# Workaround for Uniful.jl issue 562:
function _array_isapprox(x, y; kwargs...)
    all(ab -> isapprox(ab[1], ab[2]; kwargs...), zip(x, y))
end

function Base.isapprox(a::ArrayOfRDWaveforms, b::ArrayOfRDWaveforms; kwargs...)
    _array_isapprox(a.time, b.time; kwargs...) && _array_isapprox(a.signal, b.signal; kwargs...)
end


# Specialize getindex to properly support ArraysOfArrays, preventing
# conversion to exact element type:
@inline Base.getindex(A::StructArray{<:RDWaveform}, I::Int...) =
    RDWaveform(A.time[I...], A.signal[I...])


@inline ArrayOfRDWaveforms(contents) = StructArray{RDWaveform}(contents)


# Reduce the time axes of an ArrayOfRDWaveforms to the single axis they all share:
_common_time_axis(X::Fill) = first(X)

function _common_time_axis(X::AbstractArray)
    x = first(X)
    all(isequal(x), X) || throw(ArgumentError("Waveform time axes must all be equal"))
    return x
end


# Accumulator element type for sample-wise sums. Small integers widen to Int, as
# Base.sum does for plain arrays: detector samples are commonly Int32, and summing
# thousands of them in Int32 overflows silently.
_sample_sum_eltype(::Type{T}) where {T} = typeof(zero(T) + zero(T))
_sample_sum_eltype(::Type{T}) where {T<:Union{Int8,Int16,Int32}} = Int
_sample_sum_eltype(::Type{T}) where {T<:Union{UInt8,UInt16,UInt32}} = UInt

# Sample-wise reductions accumulate into a single preallocated buffer rather than
# combining whole signal vectors pairwise: one allocation instead of one per
# waveform, and independent of how the signals are stored.
#
# The accumulator's element type is promoted across every signal, not just the
# first: a ragged collection need not have a single concrete sample type, and an
# accumulator sized to only the first signal would fail on or truncate a later
# signal of a wider type.
function _sample_sum(signals::AbstractVector{<:AbstractVector})
    T = mapreduce(eltype, promote_type, signals)
    out = similar(first(signals), _sample_sum_eltype(T))
    fill!(out, zero(eltype(out)))
    for signal in signals
        axes(signal) == axes(out) || throw(DimensionMismatch("Waveform signals must all have the same axes: $(axes(signal)) vs $(axes(out))"))
        out .+= signal
    end
    return out
end

_sample_mean(signals::AbstractVector{<:AbstractVector}) = _sample_sum(signals) ./ length(signals)

# Two-pass variance: numerically better behaved than accumulating raw squares, and
# the fused broadcast below allocates no temporary per waveform.
function _sample_var(signals::AbstractVector{<:AbstractVector})
    m = _sample_mean(signals)
    acc = similar(m, typeof(abs2(zero(eltype(m)))))
    fill!(acc, zero(eltype(acc)))
    for signal in signals
        axes(signal) == axes(acc) || throw(DimensionMismatch("Waveform signals must all have the same axes: $(axes(signal)) vs $(axes(acc))"))
        acc .+= abs2.(signal .- m)
    end
    return acc ./ (length(signals) - 1)
end

_sample_std(signals::AbstractVector{<:AbstractVector}) = sqrt.(_sample_var(signals))

# Signals held in one block reduce across that block in a single pass instead of one
# pass per waveform: a modest win on a CPU (single-digit-to-low-double-digit µs on
# 2000x1024 waveforms), but the deciding factor is a device that dispatches each pass
# separately, which runs the loop above tens of times slower than this.
_sample_sum(signals::ArrayOfSimilarVectors) = dropdims(sum(flatview(signals), dims = 2), dims = 2)
_sample_mean(signals::ArrayOfSimilarVectors) = dropdims(Statistics.mean(flatview(signals), dims = 2), dims = 2)
_sample_var(signals::ArrayOfSimilarVectors) = dropdims(Statistics.var(flatview(signals), dims = 2), dims = 2)
_sample_std(signals::ArrayOfSimilarVectors) = dropdims(Statistics.std(flatview(signals), dims = 2), dims = 2)


"""
    sum(wfs::ArrayOfRDWaveforms)

Sample-wise sum over all waveforms in `wfs`, as a single [`RDWaveform`](@ref).

All waveforms must share the same time axis, which becomes the time axis of the
result; throws an `ArgumentError` otherwise. Integer samples narrower than `Int`
accumulate in `Int` to avoid overflow.
"""
Base.sum(wfs::ArrayOfRDWaveforms) = RDWaveform(_common_time_axis(wfs.time), _sample_sum(wfs.signal))

"""
    mean(wfs::ArrayOfRDWaveforms)

Sample-wise mean over all waveforms in `wfs`, as a single [`RDWaveform`](@ref).

All waveforms must share the same time axis, which becomes the time axis of the
result; throws an `ArgumentError` otherwise.
"""
Statistics.mean(wfs::ArrayOfRDWaveforms) = RDWaveform(_common_time_axis(wfs.time), _sample_mean(wfs.signal))

"""
    var(wfs::ArrayOfRDWaveforms)

Sample-wise variance over all waveforms in `wfs`, as a single [`RDWaveform`](@ref).

Uses the Bessel-corrected denominator `length(wfs) - 1`. All waveforms must share
the same time axis, which becomes the time axis of the result; throws an
`ArgumentError` otherwise.
"""
Statistics.var(wfs::ArrayOfRDWaveforms) = RDWaveform(_common_time_axis(wfs.time), _sample_var(wfs.signal))

"""
    std(wfs::ArrayOfRDWaveforms)

Sample-wise standard deviation over all waveforms in `wfs`, as a single
[`RDWaveform`](@ref).

Uses the Bessel-corrected denominator `length(wfs) - 1`. All waveforms must share
the same time axis, which becomes the time axis of the result; throws an
`ArgumentError` otherwise.
"""
Statistics.std(wfs::ArrayOfRDWaveforms) = RDWaveform(_common_time_axis(wfs.time), _sample_std(wfs.signal))


# Broadcasting an operator over an ArrayOfRDWaveforms is evaluated eagerly, one
# operation over the whole underlying sample storage, so that signals stored
# contiguously stay contiguous instead of being rebuilt as a vector of separately
# allocated vectors. Broadcast fusion is given up in exchange: an expression like
# `2 .* wfs .+ wfs` evaluates in two steps rather than one.

_broadcast_signals(f, signals::ArrayOfSimilarVectors) = nestedview(f(flatview(signals)))
_broadcast_signals(f, signals::AbstractVector{<:AbstractVector}) = map(f, signals)

_broadcast_signals(f, a::ArrayOfSimilarVectors, b::ArrayOfSimilarVectors) =
    nestedview(f(flatview(a), flatview(b)))
_broadcast_signals(f, a::AbstractVector{<:AbstractVector}, b::AbstractVector{<:AbstractVector}) =
    map(f, a, b)

_scaled_waveforms(wfs::ArrayOfRDWaveforms, f) =
    ArrayOfRDWaveforms((wfs.time, _broadcast_signals(f, wfs.signal)))

function _combined_waveforms(a::ArrayOfRDWaveforms, b::ArrayOfRDWaveforms, f)
    a.time == b.time || throw(ArgumentError("Can't combine ArrayOfRDWaveforms with different time axes"))
    ArrayOfRDWaveforms((a.time, _broadcast_signals(f, a.signal, b.signal)))
end

Base.Broadcast.broadcasted(::typeof(*), a::Real, wfs::ArrayOfRDWaveforms) =
    _scaled_waveforms(wfs, x -> a .* x)
Base.Broadcast.broadcasted(::typeof(*), wfs::ArrayOfRDWaveforms, a::Real) =
    _scaled_waveforms(wfs, x -> x .* a)
Base.Broadcast.broadcasted(::typeof(/), wfs::ArrayOfRDWaveforms, a::Real) =
    _scaled_waveforms(wfs, x -> x ./ a)
Base.Broadcast.broadcasted(::typeof(\), a::Real, wfs::ArrayOfRDWaveforms) =
    _scaled_waveforms(wfs, x -> a .\ x)

Base.Broadcast.broadcasted(::typeof(-), wfs::ArrayOfRDWaveforms) =
    _scaled_waveforms(wfs, x -> .-x)

Base.Broadcast.broadcasted(::typeof(+), a::ArrayOfRDWaveforms, b::ArrayOfRDWaveforms) =
    _combined_waveforms(a, b, (x, y) -> x .+ y)
Base.Broadcast.broadcasted(::typeof(-), a::ArrayOfRDWaveforms, b::ArrayOfRDWaveforms) =
    _combined_waveforms(a, b, (x, y) -> x .- y)

Base.Broadcast.broadcasted(::typeof(+), wfs::ArrayOfRDWaveforms, a::RealQuantity) =
    _scaled_waveforms(wfs, x -> _shift_op((y, s) -> y .+ s, x, a))
Base.Broadcast.broadcasted(::typeof(+), a::RealQuantity, wfs::ArrayOfRDWaveforms) =
    _scaled_waveforms(wfs, x -> _shift_op((y, s) -> s .+ y, x, a))
Base.Broadcast.broadcasted(::typeof(-), wfs::ArrayOfRDWaveforms, a::RealQuantity) =
    _scaled_waveforms(wfs, x -> _shift_op((y, s) -> y .- s, x, a))
Base.Broadcast.broadcasted(::typeof(-), a::RealQuantity, wfs::ArrayOfRDWaveforms) =
    _scaled_waveforms(wfs, x -> _shift_op((y, s) -> s .- y, x, a))


# One shift per waveform: the shifts broadcast along the sample axis, so
# contiguously stored signals are shifted in a single operation.
_shift_signals(f, signals::ArrayOfSimilarVectors, a::AbstractVector) =
    nestedview(_shift_op((x, s) -> f(x, transpose(s)), flatview(signals), a))
_shift_signals(f, signals::AbstractVector{<:AbstractVector}, a::AbstractVector) =
    map((signal, x) -> _shift_op(f, signal, x), signals, a)

function _shifted_waveforms(wfs::ArrayOfRDWaveforms, a::AbstractVector{<:RealQuantity}, f)
    axes(a) == axes(wfs) || throw(DimensionMismatch("Need one shift per waveform: $(axes(a)) vs $(axes(wfs))"))
    ArrayOfRDWaveforms((wfs.time, _shift_signals(f, wfs.signal, a)))
end

Base.Broadcast.broadcasted(::typeof(+), wfs::ArrayOfRDWaveforms, a::AbstractVector{<:RealQuantity}) =
    _shifted_waveforms(wfs, a, (x, y) -> x .+ y)
Base.Broadcast.broadcasted(::typeof(+), a::AbstractVector{<:RealQuantity}, wfs::ArrayOfRDWaveforms) =
    _shifted_waveforms(wfs, a, (x, y) -> y .+ x)
Base.Broadcast.broadcasted(::typeof(-), wfs::ArrayOfRDWaveforms, a::AbstractVector{<:RealQuantity}) =
    _shifted_waveforms(wfs, a, (x, y) -> x .- y)
Base.Broadcast.broadcasted(::typeof(-), a::AbstractVector{<:RealQuantity}, wfs::ArrayOfRDWaveforms) =
    _shifted_waveforms(wfs, a, (x, y) -> y .- x)


# ToDo:
# Base.broadcast.broadcasted(::typeof(Base.float), a::ArrayOfRDWaveforms)



#=
@generated function Base.getindex(x::StructArray{T, N, NamedTuple{names, types}}, I::Int...) where {T, N, names, types}
    args = [:(getfield(cols, $i)[I...]) for i in 1:length(names)]
    return quote
        cols = fieldarrays(x)
        @boundscheck checkbounds(x, I...)
        @inbounds $(Expr(:call, :createinstance, :T, args...))
    end
end
=#




#=
function StatsBase.Histogram(waveforms::AbstractVector{<:RDWaveforms})
    samples = ...

    xedge = axes(samples, 1)
    ph = Histogram((xedge, yedge), Float64, :left)
    @inbounds for evtno in axes(samples, 2)
        for i in axes(samples, 1)
            push!(ph, (i, samples[i, evtno]))
        end
    end
    ph.weights .= log10orNaN.(ph.weights)
    ph
end
=#
