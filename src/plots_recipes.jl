# This file is a part of LegendDataTypes.jl, licensed under the MIT License (MIT).


_unitful_axis_label(axis_label::AbstractString, units::Unitful.Unitlike) =
    units == NoUnits ? axis_label : "$axis_label [$units]"


function prep_for_plotting(X::AbstractVector{<:RealQuantity}, axis_label::AbstractString)
    X_units = unit(eltype(X))
    X_unitless = X_units == NoUnits ? X : ustrip.(X)
    Array(X_unitless), _unitful_axis_label(axis_label, X_units), X_units
end


@recipe function f(events::DetectorHitEvents)
    ustring(u) = u == NoUnits ? "" : " [$u]"

    edep_flat = collect(flatview(events.edep))
    edep_unit = unit(eltype(edep_flat))
    edep_unitless = Array(ustrip(edep_flat))

    layout := (2,2)

    pos_flat = flatview(collect(flatview(events.pos)))
    length_unit = unit(eltype(pos_flat))
    length_ustring = ustring(length_unit)
    pos_unitless = ustrip.(pos_flat)
    pos_x = pos_unitless[1,:]
    pos_y = pos_unitless[2,:]
    pos_z = pos_unitless[3,:]

    @series begin
        subplot := 1
        seriestype := :histogram2d
        weights := edep_unitless
        ratio := 1.0

        nbins --> 500

        #title := "Detector Hits, XY"
        xlabel := "x$length_ustring"
        ylabel := "y$length_ustring"

        (pos_x, pos_y)
    end

    @series begin
        subplot := 2
        seriestype := :histogram2d
        weights := edep_unitless
        ratio := 1.0

        nbins --> 500

        #title := "Detector Hits, ZY"
        xlabel := "z$length_ustring"
        ylabel := "y$length_ustring"

        (pos_z, pos_y)
    end

    @series begin
        subplot := 3
        seriestype := :histogram2d
        weights := edep_unitless
        ratio := 1.0

        nbins --> 500

        # title := "Detector Hits, XZ"
        xlabel := "x$length_ustring"
        ylabel := "z$length_ustring"

        (pos_x, pos_z)
    end

    @series begin
        subplot := 4
        seriestype := :stephist

        nbins --> 1000
        yscale := :log10

        normalized := :density
        legend := false

        # title:="Energy Spectrum"
        xlabel := "E $(ustring(edep_unit))"
        ylabel := edep_unit == NoUnits ? "cts / E" : "cts / $edep_unit"

        ustrip.(sum.(events.edep))
    end
end


@recipe function f(chwf::ChannelWaveform)
    X, X_label = prep_for_plotting(timeaxis(chwf), "t")
    Y, Y_label = prep_for_plotting(chwf.samples, "Amplitude")

    seriestype = get(plotattributes, :seriestype, :line)

    if seriestype in (:line, :scatter)
        @series begin
            seriestype := seriestype
            color --> chwf.chno
            xlabel --> X_label
            ylabel --> Y_label
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
