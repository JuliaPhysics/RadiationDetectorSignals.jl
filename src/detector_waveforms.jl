# This file is a part of LegendDataTypes.jl, licensed under the MIT License (MIT).

# TODO: Replace by true custom types to avoid type piracy (e.g. with plotting
# recipes) on NamedTuple and TypedTables.Table (even though the types aliased
# here are quite specific, so the risk of a conflict is low).


const WaveformSamples = AbstractVector{<:RealQuantity}
const TimeAxis = AbstractVector{<:RealQuantity}


"""
    RDWaveform

Represents a radiation detector signal waveform.

Fields:

* `v`: sample values
* `t`: time axis, typically a range
"""
struct RDWaveform{SV<:WaveformSamples,TV<:TimeAxis}
    v::SV
    t::TV
end

export RDWaveform





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
