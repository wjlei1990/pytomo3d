import os
import inspect
import pytest
import numpy as np
import numpy.testing as npt
import pytomo3d.signal.rotate as rotate
from obspy import read, read_inventory, Stream
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
teststaxml = read_inventory(staxmlfile)
testquakeml = os.path.join(DATA_DIR, "quakeml", "C201009031635A.xml")

obsfile = os.path.join(DATA_DIR, "raw", "IU.KBL.obs.mseed")
testobs = read(obsfile)
synfile = os.path.join(DATA_DIR, "raw", "IU.KBL.syn.mseed")
testsyn = read(synfile)
small_mseed = os.path.join(DATA_DIR, "raw", "BW.RJOB.obs.mseed")


def test_calculate_baz():
    elat = 0.0
    elon = 0.0
    slat = 10.0
    # put a very small value here
    slon = 0.000000001
    npt.assert_allclose(rotate.calculate_baz(elat, elon, slat, slon),
                        180.0)
    assert np.isclose(rotate.calculate_baz(slat, slon, elat, elon), 0.0)

    elat = 0.0
    elon = 0.0
    slat = 0.0
    slon = 10.0
    npt.assert_allclose(rotate.calculate_baz(elat, elon, slat, slon),
                        270.0)
    npt.assert_allclose(rotate.calculate_baz(slat, slon, elat, elon),
                        90.0)


def test_ensemble_synthetic_channel_orientation():
    dip, azi = rotate.ensemble_synthetic_channel_orientation("MXZ")
    assert np.isclose(dip, 90.0)
    assert np.isclose(azi, 0.0)

    dip, azi = rotate.ensemble_synthetic_channel_orientation("MXN")
    assert np.isclose(dip, 0.0)
    assert np.isclose(azi, 0.0)

    dip, azi = rotate.ensemble_synthetic_channel_orientation("MXE")
    assert np.isclose(dip, 0.0)
    assert np.isclose(azi, 90.0)

    with pytest.raises(Exception) as err:
        rotate.ensemble_synthetic_channel_orientation("MXR")
    assert ": MXR" in str(err.value)


def test_extract_channel_orientation():
    st = testobs.copy()
    inv = deepcopy(teststaxml)

    tr_z = st.select(channel="*Z")[0]
    dip, azi = rotate.extract_channel_orientation(tr_z, inv)
    assert dip == -90.0
    assert azi == 0.0

    tr_e = st.select(channel="*E")[0]
    dip, azi = rotate.extract_channel_orientation(tr_e, inv)
    assert dip == 0.0
    assert azi == 90.0

    tr_n = st.select(channel="*N")[0]
    dip, azi = rotate.extract_channel_orientation(tr_n, inv)
    assert dip == 0.0
    assert azi == 0.0

    tr_fake = deepcopy(tr_n)
    tr_fake.stats.station = "FAKE_SOMETHING"
    dip, azi = rotate.extract_channel_orientation(tr_fake, inv)
    assert dip is None
    assert azi is None


def test_check_vertical_inventory_sanity():
    tr_z = testobs.copy().select(channel="BHZ")[0]

    inv = deepcopy(teststaxml)
    assert rotate.check_vertical_inventory_sanity(tr_z, inv)

    inv = deepcopy(teststaxml)
    inv.select(channel="BHZ")[0][0][0].dip = 1.0
    assert not rotate.check_vertical_inventory_sanity(tr_z, inv)

    inv = deepcopy(teststaxml)
    inv.select(channel="BHZ")[0][0][0].azimuth = 1.0
    assert not rotate.check_vertical_inventory_sanity(tr_z, inv)


def test_check_horizontal_inventory_sanity():
    tr_n = testobs.copy().select(channel="BHN")[0]
    tr_e = testobs.copy().select(channel="BHE")[0]
    inv = deepcopy(teststaxml)
    assert rotate.check_horizontal_inventory_sanity(tr_n, tr_e, inv)

    inv = deepcopy(teststaxml)
    inv.select(channel="BHE")[0][0][0].dip = 1.0
    assert not rotate.check_horizontal_inventory_sanity(tr_n, tr_e, inv)

    inv = deepcopy(teststaxml)
    inv.select(channel="BHN")[0][0][0].dip = 1.0
    assert not rotate.check_horizontal_inventory_sanity(tr_n, tr_e, inv)

    inv = deepcopy(teststaxml)
    inv.select(channel="BHE")[0][0][0].azimuth += 1.0
    assert not rotate.check_horizontal_inventory_sanity(tr_n, tr_e, inv)

    inv = deepcopy(teststaxml)
    inv.select(channel="BHN")[0][0][0].azimuth += 1.0
    assert not rotate.check_horizontal_inventory_sanity(tr_n, tr_e, inv)


def test_check_information_before_rotate():
    tr_n = testobs.copy().select(channel="BHN")[0]
    tr_e = testobs.copy().select(channel="BHE")[0]
    inv = deepcopy(teststaxml)

    rotate.check_information_before_rotation(
        tr_n, tr_e, inv, True)

    tr_n_2 = tr_n.copy()
    tr_n_2.stats.delta *= 2
    with pytest.raises(ValueError) as msg:
        rotate.check_information_before_rotation(
            tr_n_2, tr_e, inv, True)
    assert "All components need to have" in str(msg.value)

    inv = deepcopy(teststaxml)
    inv.select(channel="BHN")[0][0][0].azimuth += 1.0
    with pytest.raises(ValueError) as msg:
        rotate.check_information_before_rotation(
            tr_n, tr_e, inv, True)
    assert "Horizontal component" in str(msg.value)


def test_rotate_12_rt_func():
    st = testobs.copy()
    inv = deepcopy(teststaxml)
    baz = 180
    rotate.rotate_12_rt_func(st, inv, baz, method="NE->RT",
                             sanity_check=True)
    assert len(st) == 3

    inv = deepcopy(teststaxml)
    inv.select(channel="BHE")[0][0][0].dip = 1.0
    rotate.rotate_12_rt_func(st, inv, baz, method="NE->RT",
                             sanity_check=False)
    assert len(st) == 3


def test_rotate_12_rt_func_2():
    st = testobs.copy()
    inv = deepcopy(teststaxml)
    baz = 180

    inv.select(channel="BHE")[0][0][0].dip = 1.0
    rotate.rotate_12_rt_func(st, inv, baz, method="NE->RT",
                             sanity_check=True)
    assert len(st) == 3
    assert len(st.select(component="E")) == 1
    assert len(st.select(component="N")) == 1
    assert len(st.select(component="Z")) == 1


def test_rotate_12_rt_func_3():
    st = testobs.copy()
    inv = deepcopy(teststaxml)
    baz = 180

    inv.select(channel="BHE")[0][0][0].dip = 1.0
    rotate.rotate_12_rt_func(st, inv, baz, method="12->RT",
                             sanity_check=True)
    assert len(st) == 3


def test_sort_stream_by_station():

    st = read(small_mseed)
    st += testobs.copy()
    st += testsyn.copy()

    sorted = rotate.sort_stream_by_station(st)
    assert len(sorted) == 3

    st = read(small_mseed)
    st2 = st.copy()
    tr1 = st2.select(component="N")[0]
    tr1.stats.channel = "EH1"
    tr2 = st2.select(component="E")[0]
    tr2.stats.channel = "EH2"
    st += Stream([tr1, tr2])
    sorted = rotate.sort_stream_by_station(st)
    assert len(sorted) == 1


def test_rotate_one_station_stream_obsd():
    return
    obs = testobs.copy()
    inv = deepcopy(teststaxml)

    obs1 = obs.copy()
    obs_r1 = rotate.rotate_one_station_stream(
        obs1, 0.0, 0.0, inventory=inv, mode="NE->RT",
        sanity_check=False)

    obs2 = obs.copy()
    obs_r2 = rotate.rotate_one_station_stream(
        obs2, 0.0, 0.0, station_latitude=34.5408,
        station_longitude=69.0432, mode="NE->RT",
        sanity_check=False)
    assert obs_r1 == obs_r2

    obs3 = obs.copy()
    obs_r3 = rotate.rotate_one_station_stream(
        obs3, 0.0, 0.0, inventory=inv, mode="NE->RT",
        sanity_check=True)
    assert len(obs_r3) == 1
    assert obs_r3[0].stats.channel == "BHZ"
    assert obs_r3[0] == obs_r1.select(channel="BHZ")[0]


def test_rotate_one_station_stream_synt():
    syn = testsyn.copy()
    inv = deepcopy(teststaxml)

    syn1 = syn.copy()
    syn_r1 = rotate.rotate_one_station_stream(
        syn1, 0.0, 0.0, inventory=inv, mode="NE->RT")
    syn2 = syn.copy()
    syn_r2 = rotate.rotate_one_station_stream(
        syn2, 0.0, 0.0, station_latitude=34.5408,
        station_longitude=69.0432, mode="NE->RT")
    assert syn_r1 == syn_r2

    syn3 = syn.copy()
    syn_r3 = rotate.rotate_one_station_stream(
        syn3, 0.0, 0.0, inventory=inv, mode="NE->RT")
    assert syn_r3 == syn_r1


def test_rotate_stream():
    return
    obs = testobs.copy()
    syn = testsyn.copy()
    inv = deepcopy(teststaxml)
    obs_r = rotate.rotate_stream(obs, 0.0, 0.0, inv, mode="NE->RT")
    syn_r = rotate.rotate_stream(syn, 0.0, 0.0, inv, mode="NE->RT")

    st = testobs.copy() + testsyn.copy()
    st_r = rotate.rotate_stream(st, 0.0, 0.0, inv, mode="NE->RT")
    assert st_r.select(location="") == obs_r
    assert st_r.select(location="S3") == syn_r

    st = testobs.copy() + testsyn.copy()
    st_r = rotate.rotate_stream(st, 0.0, 0.0, inv, mode="NE->RT",
                                sanity_check=True)
    assert len(st_r.select(location="")) == 1
    assert st_r.select(location="")[0].stats.channel == "BHZ"
    assert len(st_r.select(location="S3")) == 0


if __name__ == "__main__":
    test_rotate_stream()
