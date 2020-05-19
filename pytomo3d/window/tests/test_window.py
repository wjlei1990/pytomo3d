import os
import inspect
import pytest
import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from obspy import read, read_inventory, read_events
from pyflex import WindowSelector
from pyflex.window import Window
import pytomo3d.window.window as win
import pytomo3d.window.io as wio


def _upper_level(path, nlevel=4):
    """
    Go the nlevel dir up
    """
    for i in range(nlevel):
        path = os.path.dirname(path)
    return path


def reset_matplotlib():
    """
    Reset matplotlib to a common default.
    """
    # Set all default values.
    mpl.rcdefaults()
    # Force agg backend.
    plt.switch_backend('agg')
    # These settings must be hardcoded for running the comparision tests and
    # are not necessarily the default values.
    mpl.rcParams['font.family'] = 'Bitstream Vera Sans'
    mpl.rcParams['text.hinting'] = False
    # Not available for all matplotlib versions.
    try:
        mpl.rcParams['text.hinting_factor'] = 8
    except KeyError:
        pass
    import locale
    locale.setlocale(locale.LC_ALL, str('en_US.UTF-8'))


# Most generic way to get the data folder path.
TESTBASE_DIR = _upper_level(
    os.path.abspath(inspect.getfile(inspect.currentframe())), 4)
DATA_DIR = os.path.join(TESTBASE_DIR, "tests", "data")

obsfile = os.path.join(DATA_DIR, "proc", "IU.KBL.obs.proc.mseed")
synfile = os.path.join(DATA_DIR, "proc", "IU.KBL.syn.proc.mseed")
staxml = os.path.join(DATA_DIR, "stationxml", "IU.KBL.xml")
quakeml = os.path.join(DATA_DIR, "quakeml", "C201009031635A.xml")


def test_update_user_levels():
    obs_tr = read(obsfile)[0]
    syn_tr = read(synfile)[0]

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = read_events(quakeml)
    inv = read_inventory(staxml)

    user_module = "pytomo3d.window.tests.user_module_example"
    config = win.update_user_levels(user_module, config, inv, cat,
                                    obs_tr, syn_tr)

    npts = obs_tr.stats.npts
    assert isinstance(config.stalta_waterlevel, np.ndarray)
    assert len(config.stalta_waterlevel) == npts
    assert isinstance(config.tshift_acceptance_level, np.ndarray)
    assert len(config.tshift_acceptance_level) == npts
    assert isinstance(config.dlna_acceptance_level, np.ndarray)
    assert len(config.dlna_acceptance_level) == npts
    assert isinstance(config.cc_acceptance_level, np.ndarray)
    assert len(config.cc_acceptance_level) == npts
    assert isinstance(config.s2n_limit, np.ndarray)
    assert len(config.s2n_limit) == npts


def test_update_user_levels_raise():
    user_module = "pytomo3d.window.tests.which_does_not_make_sense"
    with pytest.raises(Exception) as errmsg:
        win.update_user_levels(user_module, None, None, None,
                               None, None)

    assert "Could not import the user_function module" in str(errmsg.value)

    user_module = "pytomo3d.window.io"
    with pytest.raises(Exception) as errmsg:
        win.update_user_levels(user_module, None, None, None,
                               None, None)
    assert "Given user module does not have a generate_user_levels method" \
        in str(errmsg.value)


def assert_window_close(win1, win2):
    def _compare_value(value1, value2):
        if isinstance(value1, float) or isinstance(value1, np.float64):
            np.testing.assert_allclose(value1, value2)
        elif isinstance(value1, np.float32):
            np.testing.assert_allclose(value1, value2, rtol=1e-06)
        else:
            if value1 != value2:
                raise AssertionError

    dict1 = win1.__dict__
    dict2 = win2.__dict__
    try:
        for key in dict1.keys():
            _compare_value(dict1[key], dict2[key])
    except Exception as exp:
        print("Exception raised: {}".format(exp))
        return False
    return True


def test_window_on_trace():
    obs_tr = read(obsfile).select(channel="*R")[0]
    syn_tr = read(synfile).select(channel="*R")[0]

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = read_events(quakeml)
    inv = read_inventory(staxml)

    windows = win.window_on_trace(obs_tr, syn_tr, config, station=inv,
                                  event=cat, _verbose=False,
                                  figure_mode=False)

    assert len(windows) == 5

    winfile_bm = os.path.join(DATA_DIR, "window",
                              "IU.KBL..BHR.window.json")
    with open(winfile_bm) as fh:
        windows_json = json.load(fh)

    for _win, _win_json_bm in zip(windows, windows_json):
        _win_bm = Window._load_from_json_content(_win_json_bm)
        assert_window_close(_win, _win_bm)


def test_window_on_trace_user_levels():
    obs_tr = read(obsfile)[0]
    syn_tr = read(synfile)[0]

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = read_events(quakeml)
    inv = read_inventory(staxml)
    user_module = "pytomo3d.window.tests.user_module_example"

    windows = win.window_on_trace(obs_tr, syn_tr, config, station=inv,
                                  event=cat, user_module=user_module,
                                  _verbose=False,
                                  figure_mode=False)
    assert len(windows) == 4


def test_window_on_trace_with_none_user_levels():
    obs_tr = read(obsfile).select(channel="*R")[0]
    syn_tr = read(synfile).select(channel="*R")[0]

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = read_events(quakeml)
    inv = read_inventory(staxml)

    windows = win.window_on_trace(obs_tr, syn_tr, config, station=inv,
                                  event=cat, user_module="None",
                                  _verbose=False, figure_mode=False)

    winfile_bm = os.path.join(DATA_DIR, "window",
                              "IU.KBL..BHR.window.json")
    with open(winfile_bm) as fh:
        windows_json = json.load(fh)
    for _win, _win_json_bm in zip(windows, windows_json):
        _win_bm = Window._load_from_json_content(_win_json_bm)
        assert_window_close(_win, _win_bm)


def test_window_on_stream():
    obs_tr = read(obsfile)
    syn_tr = read(synfile)

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)
    config_dict = {"Z": config, "R": config, "T": config}

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = read_events(quakeml)
    inv = read_inventory(staxml)

    windows = win.window_on_stream(obs_tr, syn_tr, config_dict, station=inv,
                                   event=cat, _verbose=False,
                                   figure_mode=False)

    assert len(windows) == 3
    nwins = dict((_w, len(windows[_w])) for _w in windows)
    assert nwins == {"IU.KBL..BHR": 5, "IU.KBL..BHZ": 2, "IU.KBL..BHT": 4}


def test_window_on_stream_user_levels():
    obs_tr = read(obsfile)
    syn_tr = read(synfile)

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)
    config_dict = {"Z": config, "R": config, "T": config}

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = read_events(quakeml)
    inv = read_inventory(staxml)

    _mod = "pytomo3d.window.tests.user_module_example"
    user_modules = {"BHZ": _mod, "BHR": _mod, "BHT": _mod}

    windows = win.window_on_stream(obs_tr, syn_tr, config_dict, station=inv,
                                   event=cat, user_modules=user_modules,
                                   _verbose=False,
                                   figure_mode=False)

    assert len(windows) == 3
    nwins = dict((_w, len(windows[_w])) for _w in windows)
    assert nwins == {"IU.KBL..BHR": 5, "IU.KBL..BHZ": 2, "IU.KBL..BHT": 4}


def test_plot_window_figure(tmpdir):
    reset_matplotlib()

    obs_tr = read(obsfile).select(channel="*R")[0]
    syn_tr = read(synfile).select(channel="*R")[0]

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = read_events(quakeml)
    inv = read_inventory(staxml)

    ws = WindowSelector(obs_tr, syn_tr, config, event=cat, station=inv)
    windows = ws.select_windows()

    assert len(windows) > 0

    win.plot_window_figure(str(tmpdir), obs_tr.id, ws, True,
                           figure_format="png")
