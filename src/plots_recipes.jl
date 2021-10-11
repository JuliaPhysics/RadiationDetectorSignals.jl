# This file is a part of RadiationDetectorSignals.jl, licensed under the MIT License (MIT).


function _unitful_axis_label(axis_label::AbstractString, units::Unitful.Unitlike)
    u_str = units == NoUnits ? "" : "$units"
    u_str2 = replace(u_str, "μ" => "u")
    isempty(u_str2) ? axis_label : "$axis_label [$u_str2]"
end


function prep_for_plotting(X::AbstractVector{<:RealQuantity}, axis_label::AbstractString)
    X_units = unit(eltype(X))
    X_unitless = X_units == NoUnits ? X : ustrip.(X)
    Array(X_unitless), _unitful_axis_label(axis_label, X_units), X_units
end


@recipe function f(events::DetectorHitEvents)

    edep_flat = collect(flatview(events.edep))
    edep_unit = unit(eltype(edep_flat))
    edep_unitless = Array(ustrip(edep_flat))

    layout := (2,2)
    unitformat --> :square

    pos_flat = flatview(collect(flatview(events.pos)))
    pos_x = pos_flat[1,:]
    pos_y = pos_flat[2,:]
    pos_z = pos_flat[3,:]

    @series begin
        subplot := 1
        seriestype := :histogram2d
        weights := edep_unitless
        aspect_ratio := 1.0

        bins --> 500

        #title := "Detector Hits, XY"
        xguide := "x"
        yguide := "y"
        

        (pos_x, pos_y)
    end

    @series begin
        subplot := 2
        seriestype := :histogram2d
        weights := edep_unitless
        aspect_ratio := 1.0

        bins --> 500

        #title := "Detector Hits, ZY"
        xguide := "z"
        yguide := "y"

        (pos_z, pos_y)
    end

    @series begin
        subplot := 3
        seriestype := :histogram2d
        weights := edep_unitless
        aspect_ratio := 1.0

        bins --> 500

        # title := "Detector Hits, XZ"
        xguide := "x"
        yguide := "z"

        (pos_x, pos_z)
    end

    @series begin
        subplot := 4
        seriestype := :stephist

        bins --> 1000
        yscale := :log10

        normalize := :density
        legend := false

        # title:="Energy Spectrum"
        xguide := "E"
        yguide := edep_unit == NoUnits ? "cts / E" : "cts / $edep_unit"

        sum.(events.edep)
    end
end


@recipe function f(wf::RDWaveform)
    X, X_label = prep_for_plotting(wf.time, "t")
    Y, Y_label = prep_for_plotting(wf.value, "sample value")

    seriestype = get(plotattributes, :seriestype, :line)

    if seriestype in (:line, :scatter)
        @series begin
            seriestype := seriestype
            xguide --> X_label
            yguide --> Y_label
            (X, Y)
        end

    #=
    elseif seriestype == :histogram2d
        n = length(linearindices(waveforms))
        h = pulse_hist(waveforms, ybinning)
        @series begin
            seriestype --> :bins2d
            x := h.edges[1]
            y := h.edges[2]
            z := Surface(h.weights)
            title --> "($n waveforms)"
            xlabel --> X_label
            ylabel --> Y_label
            ()
        end
    end
    =#
    else
        error("seriestype $seriestype not supported")
    end
end

@recipe function f(wfs::ArrayOfRDWaveforms)
    for wf in wfs
        @series begin
            wf
        end
    end
end
@recipe function f(wfs::Array{<:RDWaveform})
    @series begin
        ArrayOfRDWaveforms(wfs)
    end
end
