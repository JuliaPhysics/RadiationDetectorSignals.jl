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

Waveforms with equal time axes form a vector space: `+`, `-` and unary `-`
on waveforms and `*`, `/` and `\\` with a real scalar keep the time axis.
Floating-point time axes that differ by no more than a small fraction of the
sampling step, as from rounding, count as equal, and the result keeps the
time axis of the left operand. Combining waveforms with other time axes
throws an `ArgumentError`.

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

# Time axes match if they are equal or, for floating-point element types,
# differ pointwise by no more than a small fraction of the sampling step, as
# from rounding. Equal same-type ranges compare in O(1):
function _time_axes_match(a::AbstractVector, b::AbstractVector)
    a === b && return true
    axes(a) == axes(b) || return false
    a == b && return true
    rtol = max(_rounding_rtol(eltype(a)), _rounding_rtol(eltype(b)))
    rtol > zero(rtol) || return false
    tol = rtol * min(_sampling_step(a), _sampling_step(b))
    return all(((x, y),) -> abs(x - y) <= tol, zip(a, b))
end

_rounding_rtol(::Type{T}) where {T<:AbstractFloat} = sqrt(eps(T))
_rounding_rtol(::Type{T}) where {T<:Real} = zero(T)
_rounding_rtol(::Type{<:Unitful.AbstractQuantity{T}}) where {T} = _rounding_rtol(T)

# The first sampling interval, assuming uniform sampling:
_sampling_step(t::AbstractVector) = length(t) > 1 ? abs(t[begin + 1] - t[begin]) : zero(eltype(t))

function Base.:(+)(a::RDWaveform, b::RDWaveform)
    _time_axes_match(a.time, b.time) || throw(ArgumentError("Can't add RDWaveform with different time axes"))
    RDWaveform(a.time, a.signal + b.signal)
end

function Base.:(-)(a::RDWaveform, b::RDWaveform)
    _time_axes_match(a.time, b.time) || throw(ArgumentError("Can't subtract RDWaveform with different time axes"))
    RDWaveform(a.time, a.signal - b.signal)
end

Base.:(-)(a::RDWaveform) = RDWaveform(a.time, -a.signal)

Base.:(*)(a::Real, b::RDWaveform) = RDWaveform(b.time, a * b.signal)
Base.:(*)(a::RDWaveform, b::Real) = b * a

Base.:(/)(a::RDWaveform, b::Real) = RDWaveform(a.time, a.signal / b)

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


# Unlike StructArrays, don't convert elements to the element type of `A` or keep
# it for non-scalar results: the element types of the components can change
# under indexing, e.g. when slicing reads lazily loaded data into memory.
Base.@propagate_inbounds Base.getindex(A::StructArray{<:RDWaveform}, I...) =
    _wfs_getindex(A, to_indices(A, I)...)

Base.@propagate_inbounds function _wfs_getindex(A::StructArray{<:RDWaveform}, I::Int...)
    @boundscheck checkbounds(A, I...)
    RDWaveform(@inbounds(A.time[I...]), @inbounds(A.signal[I...]))
end

Base.@propagate_inbounds function _wfs_getindex(A::StructArray{<:RDWaveform}, I...)
    @boundscheck checkbounds(A, I...)
    ArrayOfRDWaveforms((@inbounds(A.time[I...]), @inbounds(A.signal[I...])))
end

@inline function Base.view(A::StructArray{<:RDWaveform}, I...)
    @boundscheck checkbounds(A, I...)
    ArrayOfRDWaveforms((@inbounds(view(A.time, I...)), @inbounds(view(A.signal, I...))))
end


@inline ArrayOfRDWaveforms(contents) = StructArray{RDWaveform}(contents)


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
