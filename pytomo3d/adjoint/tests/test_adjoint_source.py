import os
import inspect
import json
from obspy import read, Stream
from pyflex.window import Window
import pytomo3d.adjoint.adjoint_source as adj
import pytomo3d.adjoint.io as adj_io
import pytest
import matplotlib.pyplot as plt
# import pyadjoint.adjoint_source


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
winfile = os.path.join(DATA_DIR, "window", "IU.KBL..BHR.window.json")


def load_config_multitaper():
    config_file = os.path.join(DATA_DIR, "adjoint",
                               "multitaper.adjoint.config.yaml")
    return adj_io.load_adjoint_config_yaml(config_file)


def setup_calculate_adjsrc_on_trace_args():
    obs = read(obsfile).select(channel="*R")[0]
    syn = read(synfile).select(channel="*R")[0]

    with open(winfile) as fh:
        wins_json = json.load(fh)
    windows = []
    for _win in wins_json:
        windows.append(Window._load_from_json_content(_win))

    return obs, syn, windows


def test_calculate_adjsrc_on_trace_raises_if_obs_is_not_trace():
    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
    config = load_config_multitaper()
    obs = []
    with pytest.raises(ValueError):
        adj.calculate_adjsrc_on_trace(obs, syn, win_time, config,
                                      adj_src_type="multitaper_misfit")


def test_calculate_adrjrc_on_trace_raises_if_syn_is_not_trace():
    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
    config = load_config_multitaper()
    syn = []
    with pytest.raises(ValueError):
        adj.calculate_adjsrc_on_trace(obs, syn, win_time, config,
                                      adj_src_type="multitaper_misfit")


def test_calculate_adjsrc_on_trace_raises_if_config_is_not_config():
    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
    config = []
    with pytest.raises(ValueError):
        adj.calculate_adjsrc_on_trace(obs, syn, win_time, config,
                                      adj_src_type="multitaper_misfit")


def test_calculate_adjsrc_on_trace_raises_bad_windows_shape():
    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
    config = load_config_multitaper()
    win_time = []
    with pytest.raises(ValueError):
        adj.calculate_adjsrc_on_trace(obs, syn, win_time, config,
                                      adj_src_type="multitaper_misfit")


def test_calculate_adjsrc_on_trace_figure_mode_none_figure_dir():
    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
    config = load_config_multitaper()
    plt.switch_backend('agg')
    adjsrc = adj.calculate_adjsrc_on_trace(
        obs, syn, win_time, config, adj_src_type="multitaper_misfit",
        figure_mode=True)
    assert adjsrc


# def test_calculate_adjsrc_on_trace_waveform_misfit_produces_adjsrc():
#    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
#    config = load_config_waveform()

#    adjsrc = adj.calculate_adjsrc_on_trace(
#        obs, syn, win_time, config, adj_src_type="waveform_misfit",
#        adjoint_src_flag=True, figure_mode=False)
#    assert adjsrc


def test_calculate_adjsrc_on_trace_multitaper_misfit_produces_adjsrc():
    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
    config = load_config_multitaper()

    adjsrc = adj.calculate_adjsrc_on_trace(
        obs, syn, win_time, config, adj_src_type="multitaper_misfit",
        adjoint_src_flag=True, figure_mode=False)
    assert adjsrc


# def test_calculate_adjsrc_on_trace_traveltime_misfit_produces_adjsrc():
#    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
#    config = load_config_traveltime()
#
#    adjsrc = adj.calculate_adjsrc_on_trace(
#        obs, syn, win_time, config, adj_src_type="cc_traveltime_misfit",
#        adjoint_src_flag=True, figure_mode=False)
#    assert adjsrc


def setup_calculate_adjsrc_on_stream_args():
    obs = Stream(traces=[read(obsfile).select(channel="*R")[0]])
    syn = Stream(traces=[read(synfile).select(channel="*R")[0]])

    with open(winfile) as fh:
        wins_json = json.load(fh)

    return obs, syn, {obs[0].id: wins_json}


def test_calculate_adjsrc_on_stream_raises_if_obs_is_not_stream():
    _, syn, windows = setup_calculate_adjsrc_on_stream_args()
    config = load_config_multitaper()
    obs = []
    with pytest.raises(ValueError):
        adj.calculate_adjsrc_on_stream(obs, syn, windows, config,
                                       adj_src_type="multitaper_misfit")


def test_calculate_adjsrc_on_stream_raises_if_syn_is_not_stream():
    obs, _, windows = setup_calculate_adjsrc_on_stream_args()
    config = load_config_multitaper()
    syn = []
    with pytest.raises(ValueError):
        adj.calculate_adjsrc_on_stream(obs, syn, windows, config,
                                       adj_src_type="multitaper_misfit")


def test_calculate_adjsrc_on_stream_raises_if_config_is_not_config():
    obs, syn, windows = setup_calculate_adjsrc_on_stream_args()
    config = []
    with pytest.raises(ValueError):
        adj.calculate_adjsrc_on_stream(obs, syn, windows, config,
                                       adj_src_type="multitaper_misfit")


def test_calculate_adjsrc_on_stream_raises_if_windows_is_empty():
    obs, syn, _ = setup_calculate_adjsrc_on_stream_args()
    config = load_config_multitaper()
    windows = None
    ret = adj.calculate_adjsrc_on_stream(obs, syn, windows, config,
                                         adj_src_type="multitaper_misfit")
    assert ret is None
    windows = {}
    ret = adj.calculate_adjsrc_on_stream(obs, syn, windows, config,
                                         adj_src_type="multitaper_misfit")
    assert ret is None


# def test_calculate_adjsrc_on_stream_multitaper_misfit_produces_adjsrc():
#    obs, syn, windows = setup_calculate_adjsrc_on_stream_args()
#    config = load_config_traveltime()
#
#    adjsrc = adj.calculate_adjsrc_on_stream(
#        obs, syn, windows, config, adj_src_type="multitaper_misfit",
#        adjoint_src_flag=True, figure_mode=False)
#    assert adjsrc


# def test_calculate_adjsrc_on_stream_waveform_misfit_produces_adjsrc():
#    obs, syn, windows = setup_calculate_adjsrc_on_stream_args()
#    config = load_config_traveltime()
#
#    adjsrc = adj.calculate_adjsrc_on_stream(
#        obs, syn, windows, config, adj_src_type="waveform_misfit",
#        adjoint_src_flag=True, figure_mode=False)
#    assert adjsrc


# def test_calculate_adjsrc_on_stream_traveltime_misfit_produces_adjsrc():
#    obs, syn, windows = setup_calculate_adjsrc_on_stream_args()
#    config = load_config_traveltime()
#
#    adjsrc = adj.calculate_adjsrc_on_stream(
#        obs, syn, windows, config, adj_src_type="cc_traveltime_misfit",
#        adjoint_src_flag=True, figure_mode=False)
#    assert adjsrc


def test_measure_adjoint_on_stream():
    pass
