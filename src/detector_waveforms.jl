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

Waveforms support arithmetic, always keeping the time axis of the operands:
`+`, `-` and unary `-` between two waveforms that share a time axis, scalar
`*`, `/` and `\\`, and `+`/`-` with a scalar to shift every sample. Combining
waveforms with different time axes throws an `ArgumentError`.

A shift amount must be dimensionally compatible with the samples: shifting
unitful samples by a plain number, or plain samples by a unitful amount,
throws a `Unitful.DimensionError`.

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

function Base.:(+)(a::RDWaveform, b::RDWaveform)
    a.time == b.time || throw(ArgumentError("Can't add RDWaveform with different time axes"))
    RDWaveform(a.time, a.signal + b.signal)
end

function Base.:(-)(a::RDWaveform, b::RDWaveform)
    a.time == b.time || throw(ArgumentError("Can't subtract RDWaveform with different time axes"))
    RDWaveform(a.time, a.signal - b.signal)
end

Base.:(-)(a::RDWaveform) = RDWaveform(a.time, -a.signal)

Base.:(+)(wf::RDWaveform, a::RealQuantity) = RDWaveform(wf.time, wf.signal .+ a)
Base.:(+)(a::RealQuantity, wf::RDWaveform) = wf + a

Base.:(-)(wf::RDWaveform, a::RealQuantity) = RDWaveform(wf.time, wf.signal .- a)

Base.:(-)(a::RealQuantity, wf::RDWaveform) = RDWaveform(wf.time, a .- wf.signal)

Base.:(*)(a::Real, b::RDWaveform) = RDWaveform(b.time, a * b.signal)
Base.:(*)(a::RDWaveform, b::Real) = b * a

Base.:(/)(a::RDWaveform, b::Real) = a * inv(b)

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


# Accumulator element type for sample-wise sums: whatever `sum` returns for a vector
# of such samples. `Base.add_sum` is the reduction operator `sum` uses, and widens
# narrow integers because detector samples are commonly Int32 and summing thousands
# of them in Int32 overflows silently. Deferring to it keeps every other sample type
# consistent with `sum` as well, instead of enumerating types here.
_sample_sum_eltype(::Type{T}) where {T} = Base.promote_op(Base.add_sum, T, T)

# A matrix view of the samples, one column per waveform, sharing the signals' own
# storage; `nothing` when the signals are not one contiguous block of equal-length
# vectors. Operating on the whole block in a single pass keeps the work on whatever
# device holds the samples, instead of dispatching one pass per waveform.
_sample_matrix(signals::ArrayOfSimilarVectors) = flatview(signals)
_sample_matrix(signals::AbstractVector{<:AbstractVector}) = nothing

# A VectorOfVectors keeps its elements in one flat buffer, so equal-length elements
# reshape into that matrix without copying. Ragged ones have no matrix form.
function _sample_matrix(signals::VectorOfVectors)
    isempty(signals) && return nothing
    n = length(first(signals))
    all(signal -> length(signal) == n, signals) || return nothing
    return reshape(flatview(signals), n, length(signals))
end

# Signals with no contiguous block behind them accumulate into a single preallocated
# buffer: one allocation instead of one per waveform.
#
# The accumulator's element type follows `sum`, so narrow integer samples cannot
# overflow. Adding whole sample vectors would not widen them: `Base.add_sum` widens
# narrow integers, but `Vector{Int32} + Vector{Int32}` is a `Vector{Int32}`.
function _nested_sample_sum(signals::AbstractVector{<:AbstractVector})
    out = similar(first(signals), _sample_sum_eltype(eltype(eltype(signals))))
    fill!(out, zero(eltype(out)))
    for signal in signals
        axes(signal) == axes(out) || throw(DimensionMismatch("Waveform signals must all have the same axes: $(axes(signal)) vs $(axes(out))"))
        out .+= signal
    end
    return out
end

function _sample_sum(signals::AbstractVector{<:AbstractVector})
    M = _sample_matrix(signals)
    isnothing(M) || return dropdims(sum(M, dims = 2), dims = 2)
    return _nested_sample_sum(signals)
end

function _sample_mean(signals::AbstractVector{<:AbstractVector})
    M = _sample_matrix(signals)
    isnothing(M) || return dropdims(Statistics.mean(M, dims = 2), dims = 2)
    return _sample_sum(signals) ./ length(signals)
end

function _sample_var(signals::AbstractVector{<:AbstractVector})
    M = _sample_matrix(signals)
    isnothing(M) || return dropdims(Statistics.var(M, dims = 2), dims = 2)
    # Statistics reduces vector elements sample-wise; supplying the mean keeps the
    # accumulator that narrow integer samples need.
    return Statistics.varm(signals, _sample_mean(signals))
end

_sample_std(signals::AbstractVector{<:AbstractVector}) = sqrt.(_sample_var(signals))


"""
    sum(wfs::ArrayOfRDWaveforms)

Sample-wise sum over all waveforms in `wfs`, as a single [`RDWaveform`](@ref).

All waveforms must share the same time axis, which becomes the time axis of the
result; throws an `ArgumentError` otherwise. Signals themselves must share the
same axes; throws a `DimensionMismatch` otherwise. Integer samples narrower
than `Int` accumulate in `Int` to avoid overflow.
"""
Base.sum(wfs::ArrayOfRDWaveforms) = RDWaveform(_common_time_axis(wfs.time), _sample_sum(wfs.signal))

"""
    mean(wfs::ArrayOfRDWaveforms)

Sample-wise mean over all waveforms in `wfs`, as a single [`RDWaveform`](@ref).

All waveforms must share the same time axis, which becomes the time axis of the
result; throws an `ArgumentError` otherwise. Signals themselves must share the
same axes; throws a `DimensionMismatch` otherwise.
"""
Statistics.mean(wfs::ArrayOfRDWaveforms) = RDWaveform(_common_time_axis(wfs.time), _sample_mean(wfs.signal))

"""
    var(wfs::ArrayOfRDWaveforms)

Sample-wise variance over all waveforms in `wfs`, as a single [`RDWaveform`](@ref).

Uses the Bessel-corrected denominator `length(wfs) - 1`. All waveforms must share
the same time axis, which becomes the time axis of the result; throws an
`ArgumentError` otherwise. Signals themselves must share the same axes; throws
a `DimensionMismatch` otherwise.
"""
Statistics.var(wfs::ArrayOfRDWaveforms) = RDWaveform(_common_time_axis(wfs.time), _sample_var(wfs.signal))

"""
    std(wfs::ArrayOfRDWaveforms)

Sample-wise standard deviation over all waveforms in `wfs`, as a single
[`RDWaveform`](@ref).

Uses the Bessel-corrected denominator `length(wfs) - 1`. All waveforms must share
the same time axis, which becomes the time axis of the result; throws an
`ArgumentError` otherwise. Signals themselves must share the same axes; throws
a `DimensionMismatch` otherwise.
"""
Statistics.std(wfs::ArrayOfRDWaveforms) = RDWaveform(_common_time_axis(wfs.time), _sample_std(wfs.signal))


# Broadcasting an operator over an ArrayOfRDWaveforms is evaluated eagerly, one
# operation over the whole underlying sample storage, so that signals stored
# contiguously stay contiguous instead of being rebuilt as a vector of separately
# allocated vectors. Broadcast fusion is given up in exchange: an expression like
# `2 .* wfs .+ wfs` evaluates in two steps rather than one.
#
# Signals with no contiguous block behind them are mapped over waveform by waveform.

function _broadcast_signals(f, signals::AbstractVector{<:AbstractVector})
    M = _sample_matrix(signals)
    isnothing(M) && return map(f, signals)
    return nestedview(f(M))
end

function _broadcast_signals(f, a::AbstractVector{<:AbstractVector}, b::AbstractVector{<:AbstractVector})
    Ma, Mb = _sample_matrix(a), _sample_matrix(b)
    (isnothing(Ma) || isnothing(Mb)) && return map(f, a, b)
    return nestedview(f(Ma, Mb))
end

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
    _scaled_waveforms(wfs, x -> x .+ a)
Base.Broadcast.broadcasted(::typeof(+), a::RealQuantity, wfs::ArrayOfRDWaveforms) =
    _scaled_waveforms(wfs, x -> a .+ x)
Base.Broadcast.broadcasted(::typeof(-), wfs::ArrayOfRDWaveforms, a::RealQuantity) =
    _scaled_waveforms(wfs, x -> x .- a)
Base.Broadcast.broadcasted(::typeof(-), a::RealQuantity, wfs::ArrayOfRDWaveforms) =
    _scaled_waveforms(wfs, x -> a .- x)


# One shift per waveform: the shifts broadcast along the sample axis, so
# contiguously stored signals are shifted in a single operation.
function _shift_signals(f, signals::AbstractVector{<:AbstractVector}, a::AbstractVector)
    M = _sample_matrix(signals)
    isnothing(M) && return map(f, signals, a)
    return nestedview(f(M, transpose(a)))
end

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
