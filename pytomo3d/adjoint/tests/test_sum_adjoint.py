import os
import inspect
import pytest
from copy import deepcopy
import numpy as np
import numpy.testing as npt
from collections import namedtuple
from obspy import UTCDateTime, read_events
from pyadjoint import AdjointSource
import pytomo3d.adjoint.sum_adjoint as sa


def _upper_level(path, nlevel=4):
    """
    Go the nlevel dir up
    """
    for i in range(nlevel):
        path = os.path.dirname(path)
    return path


# Most generic way to get the data folder path.
TESTBASE_DIR = _upper_level(os.path.abspath(
    inspect.getfile(inspect.currentframe())), 4)
DATA_DIR = os.path.join(TESTBASE_DIR, "tests", "data")

obsfile = os.path.join(DATA_DIR, "proc", "IU.KBL.obs.proc.mseed")
synfile = os.path.join(DATA_DIR, "proc", "IU.KBL.syn.proc.mseed")
EVENTFILE = os.path.join(DATA_DIR, "quakeml", "C201009031635A.xml")


def test_check_adj_consistency():
    array = np.zeros(10)
    adj1 = AdjointSource(
        "cc_traveltime_misfit", 0, 1.0, 17, 40, "BHZ", adjoint_source=array,
        network="II", station="AAK", location="",
        starttime=UTCDateTime(1990, 1, 1))
    adj2 = deepcopy(adj1)
    sa.check_adj_consistency(adj1, adj2)

    adj2 = deepcopy(adj1)
    adj2.component = "HHZ"
    with pytest.raises(ValueError) as errmsg:
        sa.check_adj_consistency(adj1, adj2)

    adj2 = deepcopy(adj1)
    adj2.dt *= 2
    with pytest.raises(ValueError) as errmsg:
        sa.check_adj_consistency(adj1, adj2)
    print(str(errmsg))
    assert "DeltaT of current adjoint source" in str(errmsg.value)

    adj2 = deepcopy(adj1)
    adj2.starttime += 1
    with pytest.raises(ValueError) as errmsg:
        sa.check_adj_consistency(adj1, adj2)
    assert "Start time of current adjoint source" in str(errmsg.value)


def test_check_events_consistent():
    cat1 = read_events(EVENTFILE)[0]
    cat2 = read_events(EVENTFILE)[0]

    events = {"file1": cat1, "file2": cat2}
    sa.check_events_consistent(events)

    cat2.event_descriptions = []
    with pytest.raises(ValueError):
        sa.check_events_consistent(events)


def adjoint_equal(adj1, adj2):
    assert adj1.adj_src_type == adj2.adj_src_type
    npt.assert_almost_equal(adj1.misfit, adj2.misfit)
    npt.assert_almost_equal(adj1.dt, adj2.dt)
    npt.assert_almost_equal(adj1.min_period, adj2.min_period)
    npt.assert_almost_equal(adj1.max_period, adj2.max_period)
    assert adj1.id == adj2.id
    assert adj1.measurement == adj2.measurement
    npt.assert_array_almost_equal(
        adj1.adjoint_source, adj2.adjoint_source)
    assert adj1.starttime == adj2.starttime


def test_dump_adjsrc():
    array = np.array([1., 2., 3., 4., 5.])
    adj = AdjointSource(
        "cc_traveltime_misfit", 2.0, 1.0, 17, 40, "BHZ",
        adjoint_source=array, network="II", station="AAK",
        location="", starttime=UTCDateTime(1990, 1, 1))
    station_info = {"latitude": 1.0, "longitude": 2.0, "depth_in_m": 3.0,
                    "elevation_in_m": 4.0}
    adj_array, adj_path, parameters = sa.dump_adjsrc(adj, station_info)

    npt.assert_array_almost_equal(adj_array, array)
    for key in station_info:
        npt.assert_almost_equal(station_info[key], parameters[key])
    assert adj_path == "II_AAK_BHZ"
    npt.assert_almost_equal(parameters["misfit"], 2.0)
    npt.assert_almost_equal(parameters["dt"], 1.0)
    npt.assert_almost_equal(parameters["min_period"], 17.0)
    npt.assert_almost_equal(parameters["max_period"], 40.0)
    assert parameters["adjoint_source_type"] == "cc_traveltime_misfit"
    assert parameters["station_id"] == "II.AAK"
    assert parameters["component"], "BHZ"
    assert UTCDateTime(parameters["starttime"]) == UTCDateTime(1990, 1, 1)
    assert parameters["units"] == "m"


def test_load_to_adjsrc():
    array = np.array([1., 2., 3., 4., 5.])
    adj = AdjointSource(
        "cc_traveltime_misfit", 2.0, 1.0, 17, 40, "BHZ",
        adjoint_source=array, network="II", station="AAK",
        location="", starttime=UTCDateTime(1990, 1, 1))
    station_info = {"latitude": 1.0, "longitude": 2.0, "depth_in_m": 3.0,
                    "elevation_in_m": 4.0}
    adj_array, adj_path, parameters = sa.dump_adjsrc(adj, station_info)

    # ensemble a faked adjoint source from hdf5
    hdf5_adj = namedtuple("HDF5Adj", ['data', 'parameters'])
    hdf5_adj.data = array
    hdf5_adj.parameters = parameters

    # load and check
    loaded_adj, loaded_station_info = sa.load_to_adjsrc(hdf5_adj)

    adjoint_equal(loaded_adj, adj)

    for k in station_info:
        npt.assert_almost_equal(station_info[k], loaded_station_info[k])
    assert loaded_station_info["station"] == "AAK"
    assert loaded_station_info["network"] == "II"
    assert loaded_station_info["location"] == ""


def test_created_weighted_adj():
    array = np.array([1., 2., 3., 4., 5.])
    adj = AdjointSource(
        "cc_traveltime_misfit", 2.0, 1.0, 17, 40, "BHZ",
        adjoint_source=array, network="II", station="AAK",
        location="", starttime=UTCDateTime(1990, 1, 1))

    new_adj = sa.create_weighted_adj(adj, 1.0)
    adjoint_equal(new_adj, adj)

    new_adj = sa.create_weighted_adj(adj, 0.5)
    npt.assert_array_almost_equal(new_adj.adjoint_source,
                                  adj.adjoint_source * 0.5)
    npt.assert_almost_equal(new_adj.misfit, adj.misfit * 0.5)
    assert new_adj.location == ""


def test_sum_adj_to_base():
    array = np.array([1., 2., 3., 4., 5.])
    adjbase = AdjointSource(
        "cc_traveltime_misfit", 2.0, 1.0, 17, 40, "BHZ",
        adjoint_source=array, network="II", station="AAK",
        location="", starttime=UTCDateTime(1990, 1, 1))

    adj1 = deepcopy(adjbase)
    adj2 = deepcopy(adjbase)
    sa.sum_adj_to_base(adj1, adj2, 0.0)
    adjoint_equal(adj1, adjbase)

    adj2 = AdjointSource(
        "cc_traveltime_misfit", 2.0, 1.0, 40, 100, "BHZ",
        adjoint_source=array, network="II", station="AAK",
        location="00", starttime=UTCDateTime(1990, 1, 1))
    sa.sum_adj_to_base(adj1, adj2, 1.0)
    adj2.component = "BHR"
    with pytest.raises(ValueError):
        sa.sum_adj_to_base(adj1, adj2, 1.0)

    adj1 = deepcopy(adjbase)
    adj2 = AdjointSource(
        "cc_traveltime_misfit", 10.0, 1.0, 40, 100, "BHZ",
        adjoint_source=array, network="II", station="AAK",
        location="", starttime=UTCDateTime(1990, 1, 1))

    sa.sum_adj_to_base(adj1, adj2, 0.1)
    npt.assert_almost_equal(adj1.adjoint_source,
                            adjbase.adjoint_source * 1.1)
    npt.assert_almost_equal(adj1.misfit, 3.0)
    npt.assert_almost_equal(adj1.min_period, 17)
    npt.assert_almost_equal(adj1.max_period, 100)


def test_check_station_consistent():
    sinfo1 = {"latitude": 1.0, "longitude": 2.0, "depth_in_m": 3.0,
              "elevation_in_m": 4.0, "network": "II", "station": "AAK"}
    sinfo2 = deepcopy(sinfo1)
    assert sa.check_station_consistent(sinfo1, sinfo2)

    sinfo2 = deepcopy(sinfo1)
    sinfo2["latitude"] += 1.0
    assert not sa.check_station_consistent(sinfo1, sinfo2)


def construct_3_component_adjsrc(network="II", station="AAK", location=""):
    array = np.array([1., 2., 3., 4., 5.])
    adjz = AdjointSource(
        "cc_traveltime_misfit", 2.0, 1.0, 40, 100, "MXZ",
        adjoint_source=array, network=network, station=station,
        location=location, starttime=UTCDateTime(1990, 1, 1))

    adjr = deepcopy(adjz)
    adjr.adjoint_source = 2 * array
    adjr.component = "MXR"

    adjt = deepcopy(adjz)
    adjt.adjoint_source = 3 * array
    adjt.component = "MXT"

    return [adjr, adjt, adjz]


def test_get_station_adjsrcs():
    adjbase = construct_3_component_adjsrc("II", "AAK", "")

    adjsrcs = {"II_AAK_MXR": adjbase[0]}
    adjlist = sa.get_station_adjsrcs(adjsrcs, "II_AAK")
    assert len(adjlist) == 1
    adjoint_equal(adjlist[0], adjbase[0])

    adjsrcs = {"II_AAK_MXR": adjbase[0], "II_AAK_MXT": adjbase[1]}
    adjlist = sa.get_station_adjsrcs(adjsrcs, "II_AAK")
    assert len(adjlist) == 2
    adjoint_equal(adjlist[0], adjbase[0])
    adjoint_equal(adjlist[1], adjbase[1])

    adjsrcs = {"II_AAK_MXR": adjbase[0], "II_AAK_MXT": adjbase[1],
               "II_AAK_MXZ": adjbase[2]}
    adjlist = sa.get_station_adjsrcs(adjsrcs, "II_AAK")
    assert len(adjlist) == 3
    adjoint_equal(adjlist[0], adjbase[0])
    adjoint_equal(adjlist[1], adjbase[1])
    adjoint_equal(adjlist[2], adjbase[2])


def test_rotate_one_station_adjsrcs():
    sta_adjs = construct_3_component_adjsrc("II", "AAK", "")

    # case 1
    rotated = sa.rotate_one_station_adjsrcs(sta_adjs, 0, 10, 0, 0)
    adjoint_equal(rotated["II_AAK_MXZ"], sta_adjs[2])

    _true_east = deepcopy(sta_adjs[0])
    _true_east.component = "MXE"
    _true_east.misfit = 0.0
    adjoint_equal(rotated["II_AAK_MXE"], _true_east)

    _true_north = deepcopy(sta_adjs[1])
    _true_north.component = "MXN"
    _true_north.misfit = 0.0
    _true_north.adjoint_source *= -1
    adjoint_equal(rotated["II_AAK_MXN"], _true_north)

    # case 2
    rotated = sa.rotate_one_station_adjsrcs(sta_adjs, 0, -10, 0, 0)
    adjoint_equal(rotated["II_AAK_MXZ"], sta_adjs[2])

    _true_east = deepcopy(sta_adjs[0])
    _true_east.component = "MXE"
    _true_east.misfit = 0.0
    _true_east.adjoint_source *= -1
    adjoint_equal(rotated["II_AAK_MXE"], _true_east)

    _true_north = deepcopy(sta_adjs[1])
    _true_north.component = "MXN"
    _true_north.misfit = 0.0
    adjoint_equal(rotated["II_AAK_MXN"], _true_north)


def test_rotate_one_station_adjsrcs_2():
    sta_adjs = construct_3_component_adjsrc("II", "AAK", "")

    # case 1
    rotated = sa.rotate_one_station_adjsrcs(sta_adjs, 10, 0, 0, 0)
    adjoint_equal(rotated["II_AAK_MXZ"], sta_adjs[2])

    _true_east = deepcopy(sta_adjs[1])
    _true_east.component = "MXE"
    _true_east.misfit = 0.0
    adjoint_equal(rotated["II_AAK_MXE"], _true_east)

    _true_north = deepcopy(sta_adjs[0])
    _true_north.component = "MXN"
    _true_north.misfit = 0.0
    adjoint_equal(rotated["II_AAK_MXN"], _true_north)

    # case 2
    rotated = sa.rotate_one_station_adjsrcs(sta_adjs, -10, 0, 0, 0)
    adjoint_equal(rotated["II_AAK_MXZ"], sta_adjs[2])

    _true_east = deepcopy(sta_adjs[1])
    _true_east.component = "MXE"
    _true_east.misfit = 0.0
    _true_east.adjoint_source *= -1
    adjoint_equal(rotated["II_AAK_MXE"], _true_east)

    _true_north = deepcopy(sta_adjs[0])
    _true_north.component = "MXN"
    _true_north.misfit = 0.0
    _true_north.adjoint_source *= -1
    adjoint_equal(rotated["II_AAK_MXN"], _true_north)


def test_rotate_adjoint_sources():
    adjlist1 = construct_3_component_adjsrc("II", "AAK", "")
    adjlist2 = construct_3_component_adjsrc("II", "BBK", "")
    adjlist3 = construct_3_component_adjsrc("IU", "CCK", "")
    adjlist4 = construct_3_component_adjsrc("IU", "DDK", "")

    adjs = {"II_AAK_MXR": adjlist1[0], "II_AAK_MXT": adjlist1[1],
            "II_AAK_MXZ": adjlist1[2],
            "II_BBK_MXR": adjlist2[0], "II_BBK_MXT": adjlist2[1],
            "IU_CCK_MXZ": adjlist3[2],
            "IU_DDK_MXT": adjlist4[1]}

    stations = {"II_AAK": {"latitude": 0.0, "longitude": 10.0},
                "II_BBK": {"latitude": 0.0, "longitude": -10.0},
                "IU_CCK": {"latitude": 10.0, "longitude": 0.0},
                "IU_DDK": {"latitude": -10.0, "longitude": 0.0}}

    results = sa.rotate_adjoint_sources(adjs, stations, 0.0, 0.0)

    # validate II_AAK
    adjoint_equal(results["II_AAK_MXZ"], adjlist1[2])

    _true_east = deepcopy(adjlist1[0])
    _true_east.component = "MXE"
    _true_east.misfit = 0.0
    adjoint_equal(results["II_AAK_MXE"], _true_east)

    _true_north = deepcopy(adjlist1[1])
    _true_north.component = "MXN"
    _true_north.misfit = 0.0
    _true_north.adjoint_source *= -1
    adjoint_equal(results["II_AAK_MXN"], _true_north)

    # validate II_BBK
    _true_z = deepcopy(adjlist2[2])
    _true_z.adjoint_source.fill(0.0)
    _true_z.misfit = 0.0
    adjoint_equal(results["II_BBK_MXZ"], _true_z)

    _true_e = deepcopy(adjlist2[0])
    _true_e.component = "MXE"
    _true_e.misfit = 0.0
    _true_e.adjoint_source *= -1
    adjoint_equal(results["II_BBK_MXE"], _true_e)

    _true_n = deepcopy(adjlist2[1])
    _true_n.component = "MXN"
    _true_n.misfit = 0.0
    adjoint_equal(results["II_BBK_MXN"], _true_n)

    # validate IU_CCK
    adjoint_equal(results["IU_CCK_MXZ"], adjlist3[2])

    _true_n = deepcopy(adjlist3[0])
    _true_n.component = "MXN"
    _true_n.misfit = 0.0
    _true_n.adjoint_source.fill(0.0)
    adjoint_equal(results["IU_CCK_MXN"], _true_n)

    _true_e = deepcopy(_true_n)
    _true_e.component = "MXE"
    _true_e.misfit = 0.0
    adjoint_equal(results["IU_CCK_MXE"], _true_e)

    # validate IU_DDK
    _true_e = deepcopy(adjlist4[1])
    _true_e.component = "MXE"
    _true_e.misfit = 0.0
    _true_e.adjoint_source *= -1
    adjoint_equal(results["IU_DDK_MXE"], _true_e)

    _true_z = deepcopy(adjlist4[2])
    _true_z.misfit = 0.0
    _true_z.adjoint_source.fill(0.0)
    adjoint_equal(results["IU_DDK_MXZ"], _true_z)

    _true_n = deepcopy(_true_e)
    _true_n.component = "MXN"
    _true_n.adjoint_source.fill(0.0)
    adjoint_equal(results["IU_DDK_MXN"], _true_n)
