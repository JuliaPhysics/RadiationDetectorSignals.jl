# This file is a part of LegendDataTypes.jl, licensed under the MIT License (MIT).

# TODO: Replace by true custom types to avoid type piracy (e.g. with plotting
# recipes) on NamedTuple and TypedTables.Table (even though the types aliased
# here are quite specific, so the risk of a conflict is low).


const DetectorHit = NamedTuple{
    (:evtno, :detno, :thit, :edep, :pos),
    <:Tuple{Integer, Integer, RealQuantity, RealQuantity, AbstractVector{<:RealQuantity}}
}
export DetectorHit


const DetectorHits = TypedTables.Table{<:DetectorHit,1}
export DetectorHits
DetectorHits(x::NamedTuple{(:evtno, :detno, :thit, :edep, :pos)}) = TypedTables.Table(x)::DetectorHits
DetectorHits(x::Any) = TypedTables.Table(detno = x.detno, thit = x.thit, edep = x.edep, pos = x.pos)::DetectorHits


const DetectorHitEvents = TypedTables.Table{
    <:NamedTuple{
        (:evtno, :detno, :thit, :edep, :pos),
        <:Tuple{
            Union{Integer, AbstractVector{<:Integer}},
            Union{Integer, AbstractVector{<:Integer}},
            AbstractVector{<:RealQuantity},
            AbstractVector{<:RealQuantity},
            AbstractVector{<:AbstractVector{<:RealQuantity}}
        }
    }
}

export DetectorHitEvents

DetectorHitEvents(x::NamedTuple{(:evtno, :detno, :thit, :edep, :pos)}) = TypedTables.Table(x)::DetectorHitEvents
DetectorHitEvents(x::Any) = TypedTables.Table(evtno = x.evtno, detno = x.detno, thit = x.thit, edep = x.edep, pos = x.pos)::DetectorHitEvents


function _to_scalar(A::AbstractArray)
    x = first(A)
    !all(isequal(x), A) && throw(ArgumentError("Array values are not all equal, can't convert to scalar"))
    x
end


function group_by_evtno(hits::DetectorHits)
    issorted(hits.evtno) || throw(ArgumentError("Hits are not sorted by evtno"))
    grouped_cols = consgroupedview(hits.evtno, Tables.columns(hits))
    Table(merge(grouped_cols, (evtno = _to_scalar.(grouped_cols.evtno),)))
end


function ungroup_by_evtno(events::DetectorHitEvents)
    expanded_evtno = ((evtno, edep) -> Fill(evtno, size(edep))).(events.evtno, events.edep)
    Table(merge(
        map(flatview, Tables.columns(events)),
        (evtno = collect(flatview(expanded_evtno)),)
    ))
end


function group_by_evtno_and_detno(hits::DetectorHits)
    k = ((a,b) -> (b,a)).(hits.evtno, hits.detno)
    perm = sortperm(k)
    k_sorted = k[perm]
    sorted_hits = hits[perm]
    grouped_cols = consgroupedview(k_sorted, Tables.columns(sorted_hits))
    Table(merge(
        grouped_cols,
        (
            evtno = _to_scalar.(grouped_cols.evtno),
            detno = _to_scalar.(grouped_cols.detno),
        )
    ))
end
