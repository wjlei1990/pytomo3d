import os
import inspect
from obspy import read, read_inventory, read_events
import pyflex
import pytomo3d.window.window as win
import pytomo3d.window.io as wio


def _upper_level(path, nlevel=4):
    """
    Go the nlevel dir up
    """
    for i in range(nlevel):
        path = os.path.dirname(path)
    return path


# Most generic way to get the data folder path.
TESTBASE_DIR = _upper_level(
    os.path.abspath(inspect.getfile(inspect.currentframe())), 4)
print(TESTBASE_DIR)
DATA_DIR = os.path.join(TESTBASE_DIR, "tests", "data")

obsfile = os.path.join(DATA_DIR, "proc", "IU.KBL.obs.proc.mseed")
synfile = os.path.join(DATA_DIR, "proc", "IU.KBL.syn.proc.mseed")
staxml = os.path.join(DATA_DIR, "stationxml", "IU.KBL.xml")
quakeml = os.path.join(DATA_DIR, "quakeml", "C201009031635A.xml")


def test_load_window_config_yaml():
    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)
    assert isinstance(config, pyflex.Config)
    assert config.max_period == 60.0
    assert config.min_period == 27.0
    assert config.stalta_waterlevel == 0.10


class TestWrite:

    @staticmethod
    def get_windows():
        obs_tr = read(obsfile).select(channel="*R")[0]
        syn_tr = read(synfile).select(channel="*R")[0]

        config_file = os.path.join(
            DATA_DIR, "window", "27_60.BHZ.config.yaml")
        config = wio.load_window_config_yaml(config_file)

        cat = read_events(quakeml)

        inv = read_inventory(staxml)
        windows = win.window_on_trace(obs_tr, syn_tr, config, station=inv,
                                      event=cat, _verbose=False,
                                      figure_mode=False)
        return windows

    def test_write_txtfile(self, tmpdir):
        windows = self.get_windows()
        filename = os.path.join(str(tmpdir), "window.txt")
        wio.write_txtfile(windows, filename)

    def test_write_jsonfile(self, tmpdir):
        windows = self.get_windows()
        filename = os.path.join(str(tmpdir), "window.json")
        wio.write_jsonfile(windows, filename)
