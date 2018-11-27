# This file is a part of LegendDataTypes.jl, licensed under the MIT License (MIT).

# TODO: Replace by true custom types to avoid type piracy (e.g. with plotting
# recipes) on NamedTuple and TypedTables.Table (even though the types aliased
# here are quite specific, so the risk of a conflict is low).


const WaveformSamples = AbstractVector{<:RealQuantity}


"""
    ChannelWaveform

Fields:

* `evtno::Integer`: channel-event number
* `chno::Integer`: channel number
* `samplerate::RealQuantity`: sample rate
* ``
"""
const ChannelWaveform = NamedTuple{
    (:evtno, :chno, :delta_t, :offset, :samples),
    <:Tuple{Integer, Integer, RealQuantity, RealQuantity, WaveformSamples}
}
export ChannelWaveform


const ChannelWaveforms = TypedTables.Table{<:ChannelWaveform}
export ChannelWaveforms


timeaxis(waveform::ChannelWaveform) = waveform.delta_t .* (waveform.offset - 1 .+ axes(waveform.samples, 1))
export timeaxis


#=
function StatsBase.Histogram(waveforms::ChannelWaveforms)
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
