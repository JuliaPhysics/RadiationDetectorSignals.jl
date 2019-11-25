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
* `value`: sample values

Use [`ArrayOfRDWaveforms`](@ref) for arrays of `RDWaveform` that have a
compact memory layout.
"""
struct RDWaveform{
    T<:RealQuantity,U<:RealQuantity,
    TV<:TimeAxis{T},UV<:WaveformSamples{U},
}
    time::TV
    value::UV
end

export RDWaveform


RDWaveform{T,U,TV,UV}(wf::RDWaveform) where {T,U,TV,UV} = RDWaveform{T,U,TV,UV}(wf.time, wf.value)
Base.convert(::Type{RDWaveform{T,U,TV,UV}}, wf::RDWaveform) where {T,U,TV,UV} = RDWaveform{T,U,TV,UV}(wf)


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
    NamedTuple{(:time, :value), Tuple{VVT,VVU}}
}

export ArrayOfRDWaveforms


function StructArray{RDWaveform}(
    contents::Tuple{
        AbstractArray{<:AbstractVector{<:RealQuantity},N},
        AbstractArray{<:AbstractVector{<:RealQuantity},N}
    }
) where {N}
    time, value = contents
    VT = eltype(time)
    VU = eltype(value)
    T = eltype(VT)
    U = eltype(VU)
    StructArray{RDWaveform{T,U,VT,VU}}((time, value))
end


StructArray{RDWaveform}(waveforms::AbstractVector{<:RDWaveform}) =
    StructArray{RDWaveform}((map(w -> w.time, waveforms), VectorOfVectors(map(w -> w.value, waveforms))))

Base.convert(::Type{ArrayOfRDWaveforms}, waveforms::AbstractVector{<:RDWaveform}) = StructArray{RDWaveform}(waveforms)
Base.convert(::Type{ArrayOfRDWaveforms}, waveforms::StructArray{<:RDWaveform}) = waveforms

Base.convert(::Type{StructArray{RDWaveform}}, waveforms::AbstractVector{<:RDWaveform}) = StructArray{RDWaveform}(waveforms)
Base.convert(::Type{StructArray{RDWaveform}}, waveforms::StructArray{<:RDWaveform}) = waveforms


# Specialize getindex to properly support ArraysOfArrays, preventing
# conversion to exact element type:
@inline Base.getindex(A::StructArray{<:RDWaveform}, I::Int...) =
    RDWaveform(A.time[I...], A.value[I...])


@inline ArrayOfRDWaveforms(contents) = StructArray{RDWaveform}(contents)





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

function Base.float(wf::RDWaveform)
    #...
end



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
