import os
import inspect
import numpy as np
import pytest
import obspy
import pytomo3d.signal.process as proc
from copy import deepcopy


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

staxmlfile = os.path.join(DATA_DIR, "stationxml", "IU.KBL.xml")
teststaxml = obspy.read_inventory(staxmlfile)
testquakeml = os.path.join(DATA_DIR, "quakeml", "C201009031635A.xml")
obsfile = os.path.join(DATA_DIR, "raw", "IU.KBL.obs.mseed")
testobs = obspy.read(obsfile)
synfile = os.path.join(DATA_DIR, "raw", "IU.KBL.syn.mseed")
testsyn = obspy.read(synfile)
small_mseed = os.path.join(DATA_DIR, "raw", "BW.RJOB.obs.mseed")


def test_check_array():
    array = [1, 2, 3, 4]
    assert proc.check_array_order(array, order='ascending')
    array = [-1.0, -2.0, -3, -4]
    assert proc.check_array_order(array, order='descending')
    array = [2.0, 1.0, 3.0, 4.0]
    assert (not proc.check_array_order(array))


def test_flex_cut_trace():

    st = obspy.read(small_mseed)
    tr_old = st[0]
    tstart = tr_old.stats.starttime
    tend = tr_old.stats.endtime
    npts = tr_old.stats.npts
    dt = tr_old.stats.delta

    tr = tr_old.copy()
    t1 = tstart + int(npts / 4) * dt
    t2 = tend - int(npts / 4) * dt
    proc.flex_cut_trace(tr, t1, t2)
    assert tr.stats.starttime == t1
    assert tr.stats.endtime == t2

    tr = tr_old.copy()
    t1 = tstart + 20 * dt
    t2 = tend - 20 * dt
    proc.flex_cut_trace(tr, t1, t2, dynamic_npts=10)
    assert tr.stats.starttime == (t1 - 10 * dt)
    assert tr.stats.endtime == (t2 + 10 * dt)

    tr = tr_old.copy()
    t1 = tstart - int(npts / 4) * dt
    t2 = tend + int(npts / 4) * dt
    proc.flex_cut_trace(tr, t1, t2)
    assert tr.stats.starttime == tstart
    assert tr.stats.endtime == tend

    tr = tr_old.copy()
    t1 = tstart + int(npts * 0.8) * dt
    t2 = tend - int(npts * 0.8) * dt
    with pytest.raises(ValueError):
        proc.flex_cut_trace(tr, t1, t2)


def test_flex_cut_stream():
    st = obspy.read(small_mseed)
    tstart = st[0].stats.starttime
    tend = st[0].stats.endtime
    dt = st[0].stats.delta
    t1 = tstart + 100 * dt
    t2 = tend - 100 * dt
    dynamic_npts = 5
    st = proc.flex_cut_stream(st, t1, t2, dynamic_npts=dynamic_npts)
    for tr in st:
        assert tr.stats.starttime == t1 - dynamic_npts * dt
        assert tr.stats.endtime == t2 + dynamic_npts * dt


def test_filter_trace():
    st = testsyn.copy()
    pre_filt = [1/90., 1/60., 1/27.0, 1/22.5]

    # check length doesn't change after filtering
    tr = st[0].copy()
    proc.filter_trace(tr, pre_filt)
    assert len(tr.data) == len(st[0].data)


def compare_stream_kernel(st1, st2):
    if len(st1) != len(st2):
        return False
    for tr1 in st1:
        tr2 = st2.select(id=tr1.id)[0]
        if not compare_trace_kernel(tr1, tr2):
            return False
    return True


def compare_trace_kernel(tr1, tr2):
    if tr1.stats.starttime != tr2.stats.starttime:
        return False
    if tr1.stats.endtime != tr2.stats.endtime:
        return False
    if tr1.stats.sampling_rate != tr2.stats.sampling_rate:
        return False
    if tr1.stats.npts != tr2.stats.npts:
        return False
    if not np.allclose(tr1.data, tr2.data):
        return False
    return True


def test_process_obsd():

    st = testobs.copy()
    inv = deepcopy(teststaxml)
    event = obspy.read_events(testquakeml)[0]
    origin = event.preferred_origin() or event.origins[0]
    event_lat = origin.latitude
    event_lon = origin.longitude
    event_time = origin.time

    pre_filt = [1/90., 1/60., 1/27.0, 1/22.5]
    t1 = event_time
    t2 = event_time + 6000.0
    st_new = proc.process_stream(
        st, remove_response_flag=True, water_level=60, inventory=inv,
        filter_flag=True, pre_filt=pre_filt,
        starttime=t1, endtime=t2, resample_flag=True,
        sampling_rate=2.0, taper_type="hann",
        taper_percentage=0.05, rotate_flag=True,
        event_latitude=event_lat,
        event_longitude=event_lon)
    bmfile = os.path.join(DATA_DIR, "proc", "IU.KBL.obs.proc.mseed")
    st_compare = obspy.read(bmfile)
    assert compare_stream_kernel(st_new, st_compare)


def test_process_obsd_2():
    st = testobs.copy()
    inv = deepcopy(teststaxml)
    event = obspy.read_events(testquakeml)[0]
    origin = event.preferred_origin() or event.origins[0]
    event_lat = origin.latitude
    event_lon = origin.longitude
    event_time = origin.time

    pre_filt = [1/90., 1/60., 1/27.0, 1/22.5]
    t1 = event_time
    t2 = event_time + 6000.0
    st_new = proc.process_stream(
        st, remove_response_flag=True, water_level=60, inventory=inv,
        filter_flag=True, pre_filt=pre_filt,
        starttime=t1, endtime=t2, resample_flag=True,
        sampling_rate=2.0, taper_type="hann",
        taper_percentage=0.05, rotate_flag=True,
        event_latitude=event_lat,
        event_longitude=event_lon,
        sanity_check=True)
    bmfile = os.path.join(DATA_DIR, "proc", "IU.KBL.obs.proc.mseed")
    st_compare = obspy.read(bmfile)
    assert len(st_new) == 3
    assert compare_trace_kernel(st_new.select(channel="BHZ")[0],
                                st_compare.select(channel="BHZ")[0])


def test_process_synt():
    staxmlfile = os.path.join(DATA_DIR, "stationxml", "IU.KBL.syn.xml")
    inv = obspy.read_inventory(staxmlfile)

    st = testsyn.copy()
    event = obspy.read_events(testquakeml)[0]
    origin = event.preferred_origin() or event.origins[0]
    event_lat = origin.latitude
    event_lon = origin.longitude
    event_time = origin.time

    pre_filt = [1/90., 1/60., 1/27.0, 1/22.5]
    t1 = event_time
    t2 = event_time + 6000.0
    st_new = proc.process_stream(
        st, remove_response_flag=False, inventory=inv,
        filter_flag=True, pre_filt=pre_filt,
        starttime=t1, endtime=t2, resample_flag=True,
        sampling_rate=2.0, taper_type="hann",
        taper_percentage=0.05, rotate_flag=True,
        event_latitude=event_lat,
        event_longitude=event_lon)
    bmfile = os.path.join(DATA_DIR, "proc", "IU.KBL.syn.proc.mseed")
    st_compare = obspy.read(bmfile)
    assert compare_stream_kernel(st_new, st_compare)
