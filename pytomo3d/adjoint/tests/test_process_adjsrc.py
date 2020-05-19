import os
import inspect
from copy import deepcopy
import numpy as np
import numpy.testing as npt
import obspy
from obspy import UTCDateTime, read
from pyadjoint import AdjointSource
import pytomo3d.adjoint.process_adjsrc as pa
from pytomo3d.signal.rotate import rotate_stream

# use the obspy default stream in read function
SAMPLE_STREAM = read()


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


def test_calculate_baz():
    npt.assert_almost_equal(pa.calculate_baz(0, 0, 10, 0), 180.0)
    npt.assert_almost_equal(pa.calculate_baz(0, 0, 0, 10), 270.0)
    npt.assert_almost_equal(pa.calculate_baz(0, 0, 0, -10), 90.0)
    npt.assert_almost_equal(pa.calculate_baz(0, 0, -10, 0), 360.0)


def test_change_channel_name():
    st = SAMPLE_STREAM.copy()
    for tr in st:
        assert tr.stats.channel[0:2] == "EH"
    pa.change_channel_name(st, "MX")
    for tr in st:
        assert tr.stats.channel[0:2] == "MX"


def check_array_reverse(arr1, arr2):
    assert len(arr1) == len(arr2)
    for i in range(len(arr1)):
        npt.assert_almost_equal(arr1[i], arr2[-(i+1)])


def test_time_reverse_array():
    st = SAMPLE_STREAM.copy()
    _st = st.copy()
    pa.time_reverse_array(_st)
    for tr, _tr in zip(st, _st):
        check_array_reverse(tr.data, _tr.data)


def test_convert_adjs_to_trace():
    array = np.array([1., 2., 3., 4., 5.])
    starttime = UTCDateTime(1990, 1, 1)
    adj = AdjointSource(
        "cc_traveltime_misfit", 0, 1.0, 17, 40, "BHZ", adjoint_source=array,
        network="II", station="AAK", location="",
        starttime=starttime)

    tr, meta = pa.convert_adj_to_trace(adj)
    npt.assert_allclose(tr.data, array)
    assert tr.stats.starttime == starttime
    npt.assert_almost_equal(tr.stats.delta, 1.0)
    assert tr.id == "II.AAK..BHZ"

    assert meta["adj_src_type"] == "cc_traveltime_misfit"
    npt.assert_almost_equal(meta["misfit"], 0.0)
    npt.assert_almost_equal(meta["min_period"], 17.0)
    npt.assert_almost_equal(meta["max_period"], 40.0)


def get_sample_adjsrcs(array, starttime):
    adjz = AdjointSource(
        "cc_traveltime_misfit", 0, 1.0, 17, 40, "BHZ", adjoint_source=array,
        network="II", station="AAK", location="",
        starttime=starttime)
    adjr = deepcopy(adjz)
    adjr.component = "BHR"
    adjt = deepcopy(adjz)
    adjt.component = "BHT"
    return [adjz, adjr, adjt]


def test_convert_adjs_to_stream():

    array = np.array([1., 2., 3., 4., 5.])
    starttime = UTCDateTime(1990, 1, 1)
    adjsrcs = get_sample_adjsrcs(array, starttime)

    true_keys = ["II.AAK..BHZ", "II.AAK..BHR", "II.AAK..BHT"]
    st, meta = pa.convert_adjs_to_stream(adjsrcs)
    assert len(meta) == 3
    keys = meta.keys()
    assert set(keys) == set(true_keys)
    for m in meta.values():
        assert m["adj_src_type"] == "cc_traveltime_misfit"
        npt.assert_almost_equal(m["misfit"], 0.0)
        npt.assert_almost_equal(m["min_period"], 17.0)
        npt.assert_almost_equal(m["max_period"], 40.0)

    for tr, trid in zip(st, true_keys):
        assert tr.id == trid
        npt.assert_allclose(tr.data, array)
        npt.assert_almost_equal(tr.stats.delta, 1.0)
        assert tr.stats.starttime == starttime


def test_convert_trace_to_adj():
    tr = SAMPLE_STREAM.copy()[0]
    meta = {"adj_src_type": "cc_traveltime_misfit", "misfit": 1.0,
            "min_period": 17.0, "max_period": 40.0}
    adj = pa.convert_trace_to_adj(tr, meta)
    npt.assert_allclose(adj.adjoint_source, tr.data)
    npt.assert_almost_equal(adj.dt, tr.stats.delta)
    assert adj.id == tr.id
    assert adj.starttime == tr.stats.starttime
    assert adj.adj_src_type == "cc_traveltime_misfit"
    npt.assert_almost_equal(adj.misfit, 1.0)
    npt.assert_almost_equal(adj.min_period, 17.0)
    npt.assert_almost_equal(adj.max_period, 40.0)


def assert_adj_same(adj1, adj2):
    assert adj1.id == adj2.id
    npt.assert_almost_equal(adj1.misfit, adj2.misfit)
    npt.assert_almost_equal(adj1.dt, adj2.dt)
    npt.assert_almost_equal(adj1.min_period, adj2.min_period)
    npt.assert_almost_equal(adj1.max_period, adj2.max_period)
    npt.assert_allclose(adj1.adjoint_source, adj2.adjoint_source)
    assert adj1.starttime == adj2.starttime


def test_convert_trace_to_adj_2():
    array = np.array([1., 2., 3., 4., 5.])
    starttime = UTCDateTime(1990, 1, 1)
    adj = AdjointSource(
        "cc_traveltime_misfit", 0, 1.0, 17, 40, "BHZ", adjoint_source=array,
        network="II", station="AAK", location="",
        starttime=starttime)

    tr, meta = pa.convert_adj_to_trace(adj)
    adj_new = pa.convert_trace_to_adj(tr, meta)
    assert_adj_same(adj, adj_new)


def test_convert_stream_to_adjs():
    array = np.array([1., 2., 3., 4., 5.])
    starttime = UTCDateTime(1990, 1, 1)
    adjsrcs = get_sample_adjsrcs(array, starttime)

    st, meta = pa.convert_adjs_to_stream(adjsrcs)
    adjsrcs_new = pa.convert_stream_to_adjs(st, meta)

    for adj, adj_new in zip(adjsrcs, adjsrcs_new):
        assert_adj_same(adj, adj_new)


def test_zero_padding_stream():
    tr = obspy.Trace()
    array = np.array([1., 2., 3.])
    tr.data = np.array(array)
    st = obspy.Stream([tr])

    starttime = tr.stats.starttime - 10 * tr.stats.delta
    endtime = tr.stats.endtime + 5 * tr.stats.delta
    st_new = deepcopy(st)
    pa.zero_padding_stream(st_new, starttime, endtime)

    assert len(st_new) == 1
    tr_new = st_new[0]
    assert tr_new.stats.starttime == (starttime - 1.0)
    assert tr_new.stats.endtime == (endtime + 1.0)
    assert len(tr_new) == 20
    npt.assert_allclose(tr_new.data[0:11], np.zeros(11))
    npt.assert_allclose(tr_new.data[11:14], array)
    npt.assert_allclose(tr_new.data[14:20], np.zeros(6))


def assert_trace_equal(tr1, tr2, rtol=1e-07, atol=0):
    assert tr1.id == tr2.id
    assert tr1.stats.starttime == tr2.stats.starttime
    assert tr1.stats.endtime == tr2.stats.endtime
    assert tr1.stats.npts == tr2.stats.npts
    npt.assert_almost_equal(tr1.stats.delta, tr2.stats.delta)
    npt.assert_allclose(tr1.data, tr2.data, rtol=rtol, atol=atol)


def test_sum_adjoint_no_weighting():
    st = SAMPLE_STREAM.copy()
    trz = deepcopy(st.select(component="Z")[0])
    trz.stats.location = "00"
    st.append(trz)
    meta_info = {"BW.RJOB..EHZ": {"misfit": 1.0, "type": "test1"},
                 "BW.RJOB..EHN": {"misfit": 2.0, "type": "test2"},
                 "BW.RJOB..EHE": {"misfit": 3.0, "type": "test3"},
                 "BW.RJOB.00.EHZ": {"misfit": 4.0, "type": "test4"}}
    new_st, new_meta = pa.sum_adjoint_no_weighting(st, meta_info)

    assert len(new_st) == 3
    assert len(new_meta) == 3

    _true_z = st.select(component="Z")[0]
    _true_z.data *= 2
    _true_z.stats.channel = "MXZ"
    assert_trace_equal(_true_z, new_st.select(component="Z")[0])
    assert new_meta["BW.RJOB..MXZ"] == {"misfit": 5.0, "type": "test1"}

    _true_n = st.select(component="N")[0]
    _true_n.stats.channel = "MXN"
    assert_trace_equal(_true_n, new_st.select(component="N")[0])
    assert new_meta["BW.RJOB..MXN"] == {"misfit": 2.0, "type": "test2"}

    _true_e = st.select(component="E")[0]
    _true_e.stats.channel = "MXE"
    assert_trace_equal(_true_e, new_st.select(component="E")[0])
    assert new_meta["BW.RJOB..MXE"] == {"misfit": 3.0, "type": "test3"}


def test_sum_adjoint_with_weighting():
    st = SAMPLE_STREAM.copy()
    trz = deepcopy(st.select(component="Z")[0])
    trz.stats.location = "00"
    st.append(trz)
    meta_info = {"BW.RJOB..EHZ": {"misfit": 1.0, "type": "test1"},
                 "BW.RJOB..EHN": {"misfit": 2.0, "type": "test2"},
                 "BW.RJOB..EHE": {"misfit": 3.0, "type": "test3"},
                 "BW.RJOB.00.EHZ": {"misfit": 4.0, "type": "test4"}}
    weight_dict = {"MXZ": {"BW.RJOB..EHZ": 2.0, "BW.RJOB.00.EHZ": 1.0},
                   "MXN": {"BW.RJOB..EHN": 2.0},
                   "MXE": {"BW.RJOB..EHE": 3.0}}
    new_st, new_meta = pa.sum_adjoint_with_weighting(
        st, meta_info, weight_dict)

    assert len(new_st) == 3
    assert len(new_meta) == 3

    _true_z = st.select(component="Z")[0]
    _true_z.data *= 3
    _true_z.stats.channel = "MXZ"
    assert_trace_equal(_true_z, new_st.select(component="Z")[0])
    assert new_meta["BW.RJOB..MXZ"] == {"misfit": 6.0, "type": "test1"}

    _true_n = st.select(component="N")[0]
    _true_n.stats.channel = "MXN"
    _true_n.data *= 2
    assert_trace_equal(_true_n, new_st.select(component="N")[0])
    assert new_meta["BW.RJOB..MXN"] == {"misfit": 4.0, "type": "test2"}

    _true_e = st.select(component="E")[0]
    _true_e.stats.channel = "MXE"
    _true_e.data *= 3
    assert_trace_equal(_true_e, new_st.select(component="E")[0])
    assert new_meta["BW.RJOB..MXE"] == {"misfit": 9.0, "type": "test3"}


def test_add_missing_components():
    array = np.array([1., 2., 3., 4., 5.])
    starttime = UTCDateTime(1990, 1, 1)
    adjsrcs = get_sample_adjsrcs(array, starttime)
    st, _ = pa.convert_adjs_to_stream(adjsrcs)

    nadds = pa.add_missing_components(st)
    assert nadds == 0

    st.remove(st.select(component="Z")[0])
    nadds = pa.add_missing_components(st)
    assert nadds == 1
    trz = st.select(component="Z")[0]
    assert trz.id == "II.AAK..BHZ"
    npt.assert_allclose(trz.data, np.zeros(5))
    assert trz.stats.starttime == starttime

    st.remove(st.select(component="R")[0])
    nadds = pa.add_missing_components(st)
    assert nadds == 1
    trr = st.select(component="R")[0]
    assert trr.id == "II.AAK..BHR"
    npt.assert_allclose(trr.data, np.zeros(5))
    assert trr.stats.starttime == starttime


def test_rotate_adj_stream():
    # rotate from NEZ to RTZ and rotate back
    st = SAMPLE_STREAM.copy()
    inv = obspy.read_inventory()
    st_rtz = st.copy()
    rotate_stream(st_rtz, 0, 0, inv, mode="NE->RT", sanity_check=True)
    st_nez = st_rtz.copy()
    pa.rotate_adj_stream(st_nez, 0, 0, inv)
    for tr1, tr2 in zip(st, st_nez):
        assert_trace_equal(tr1, tr2)


def test_interp_adj_stream():
    st = SAMPLE_STREAM.copy()
    _st = st.copy()
    starttime = st[0].stats.starttime
    delta = st[0].stats.delta
    npts = st[0].stats.npts

    dnpts = 20
    dt = dnpts * delta
    pa.interp_adj_stream(
        st, interp_starttime=starttime-dt, interp_delta=delta,
        interp_npts=npts + 2 * dnpts)

    st.interpolate(sampling_rate=1/delta, starttime=starttime, npts=npts)
    for tr, _tr in zip(st, _st):
        print(tr.stats.npts, _tr)
        assert_trace_equal(tr, _tr, rtol=5e-3, atol=1e-2)


def test_process_adjoint():
    array = np.array([1, 2, 3, 4, 5])
    starttime = UTCDateTime(1990, 1, 1)
    adjsrcs = get_sample_adjsrcs(array, starttime)
    final_adjsrcs = pa.process_adjoint(
        adjsrcs, interp_flag=False, sum_over_comp_flag=False,
        filter_flag=False, add_missing_comp_flag=False,
        rotate_flag=False)

    assert len(final_adjsrcs) == 3
    for adj1, adj2 in zip(adjsrcs, final_adjsrcs):
        npt.assert_allclose(adj1.adjoint_source, adj2.adjoint_source[::-1])
        assert adj1.id == adj2.id

    adjsrcs = [adjsrcs[0], adjsrcs[1]]
    final_adjsrcs = pa.process_adjoint(
        adjsrcs, interp_flag=False, sum_over_comp_flag=False,
        filter_flag=False, add_missing_comp_flag=True,
        rotate_flag=False)
    assert len(final_adjsrcs) == 3
    assert adjsrcs[0].id == final_adjsrcs[0].id
    npt.assert_allclose(adjsrcs[0].adjoint_source,
                        final_adjsrcs[0].adjoint_source[::-1])
    assert final_adjsrcs[1].id == "II.AAK..BHR"
    npt.assert_allclose(final_adjsrcs[1].adjoint_source,
                        adjsrcs[1].adjoint_source[::-1])
    assert final_adjsrcs[2].id == "II.AAK..BHT"
    npt.assert_allclose(final_adjsrcs[2].adjoint_source,
                        np.zeros(len(array)))


def prepare_real_adj_data():
    st = SAMPLE_STREAM.copy()
    inv = obspy.read_inventory()
    event = obspy.read_events()[0]
    elat = event.preferred_origin().latitude
    elon = event.preferred_origin().longitude

    rotate_stream(st, elat, elon, inv)
    new_z = st.select(component="Z")[0].copy()
    new_z.stats.location = "00"
    st.append(new_z)

    meta = {
        "BW.RJOB..EHZ": {
            "adj_src_type": "waveform_misfit", "misfit": 1.0,
            "min_period": 17.0, "max_period": 40.0},
        "BW.RJOB..EHR": {
            "adj_src_type": "waveform_misfit", "misfit": 2.0,
            "min_period": 17.0, "max_period": 40.0},
        "BW.RJOB..EHT": {
            "adj_src_type": "waveform_misfit", "misfit": 3.0,
            "min_period": 17.0, "max_period": 40.0},
        "BW.RJOB.00.EHZ": {
            "adj_src_type": "waveform_misfit", "misfit": 4.0,
            "min_period": 17.0, "max_period": 40.0}}
    return st, meta


def test_process_adjoint_2():
    st, meta = prepare_real_adj_data()
    inv = obspy.read_inventory()
    event = obspy.read_events()[0]

    adjs = pa.convert_stream_to_adjs(st, meta)
    starttime = adjs[0].starttime - 20
    delta = adjs[0].dt / 2.0
    npts = 40 + 2 * len(adjs[0].adjoint_source)
    new_adj = pa.process_adjoint(
        adjs, interp_flag=True, interp_starttime=starttime,
        interp_delta=delta, interp_npts=npts, sum_over_comp_flag=True,
        weight_flag=False, filter_flag=True,
        pre_filt=[0.02, 0.025, 0.059, 0.073],
        taper_percentage=0.05, taper_type="hann",
        add_missing_comp_flag=True, rotate_flag=True,
        inventory=inv, event=event
    )
    assert len(new_adj) == 3
