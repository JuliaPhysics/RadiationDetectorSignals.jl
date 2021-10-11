# This file is a part of RadiationDetectorSignals.jl, licensed under the MIT License (MIT).

using RadiationDetectorSignals
using Test

using ArraysOfArrays, StaticArrays, TypedTables, Unitful


@testset "detector_hits" begin
    evtno_data = [[1, 1, 1], [2], [3, 4]]
    detno_data = [[2, 4, 4], [3], [2, 5]]
    thit_data = [[0.1, 0.2, 0.4], [0.3], [0.1, 0.7]]
    edep_data = [[45.2, 683.1, 137.4], [64.3], [84.2, 1043.0]]
    pos_data = [
        [SVector(0.4,-5.2,3.3), SVector(2.1,4.7,-5.5), SVector(-7.2,1.1,4.2)],
        [SVector(8.1,-0.9,-4.9)],
        [SVector(2.0,8.7,0.3), SVector(-2.3,9.8,7.4)]
    ]

    @test @inferred(DetectorHits((evtno = evtno_data[1], detno = detno_data[1], thit = thit_data[1], edep = edep_data[1], pos = pos_data[1]))) isa DetectorHits
    x = DetectorHits((evtno = evtno_data[1], detno = detno_data[1], thit = thit_data[1], edep = edep_data[1], pos = pos_data[1]))

    @test @inferred(DetectorHitEvents((evtno = evtno_data, detno = detno_data, thit = thit_data, edep = edep_data, pos = pos_data))) isa DetectorHitEvents
    X = DetectorHitEvents((evtno = evtno_data, detno = detno_data, thit = thit_data, edep = edep_data, pos = pos_data))

    @test @inferred(Table(X[1])) == x
end # testset

@testset "detector_hits with units" begin
    evtno_data = [[1, 1, 1], [2], [3, 4]]
    detno_data = [[2, 4, 4], [3], [2, 5]]
    thit_data = [[0.1, 0.2, 0.4]u"ns", [0.3]u"ns", [0.1, 0.7]u"ns"]
    edep_data = [[45.2, 683.1, 137.4]u"keV", [64.3]u"keV", [84.2, 1043.0]u"keV"]
    pos_data = [
        [SVector(0.4,-5.2,3.3)u"mm", SVector(2.1,4.7,-5.5)u"mm", SVector(-7.2,1.1,4.2)u"mm"],
        [SVector(8.1,-0.9,-4.9)u"mm"],
        [SVector(2.0,8.7,0.3)u"mm", SVector(-2.3,9.8,7.4)u"mm"]
    ]

    @test @inferred(DetectorHits((evtno = evtno_data[1], detno = detno_data[1], thit = thit_data[1], edep = edep_data[1], pos = pos_data[1]))) isa DetectorHits
    x = DetectorHits((evtno = evtno_data[1], detno = detno_data[1], thit = thit_data[1], edep = edep_data[1], pos = pos_data[1]))

    @test @inferred(DetectorHitEvents((evtno = evtno_data, detno = detno_data, thit = thit_data, edep = edep_data, pos = pos_data))) isa DetectorHitEvents
    X = DetectorHitEvents((evtno = evtno_data, detno = detno_data, thit = thit_data, edep = edep_data, pos = pos_data))

    @test @inferred(Table(X[1])) == x
end # testset
