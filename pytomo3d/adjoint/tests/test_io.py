import os
import inspect
import pytomo3d.adjoint.io as adj_io
import pyadjoint


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


# @pytest.fixture
# def load_config_waveform():
#    config_file = os.path.join(DATA_DIR, "adjoint",
#                               "waveform.adjoint.config.yaml")
#    return adj_utils.load_adjoint_config_yaml(config_file)


# @pytest.fixture
# def load_config_traveltime():
#    config_file = os.path.join(DATA_DIR, "adjoint",
#                               "cc_traveltime.adjoint.config.yaml")
#    return adj_utils.load_adjoint_config_yaml(config_file)


def load_config_multitaper():
    config_file = os.path.join(DATA_DIR, "adjoint",
                               "multitaper.adjoint.config.yaml")
    return adj_io.load_adjoint_config_yaml(config_file)


# def test_load_adjoint_config_yaml_for_waveform_misfit():
#    config = load_config_waveform()
#    assert isinstance(config, pyadjoint.Config)
#    assert config.max_period == 60.0
#    assert config.min_period == 27.0
#    assert config.taper_percentage == 0.15
#    assert config.taper_type == 'hann'
#    assert not config.use_cc_error


# def test_load_adjoint_config_yaml_for_traveltime_misfit():
#    config = load_config_traveltime()
#    assert isinstance(config, pyadjoint.Config)
#    assert config.max_period == 60.0
#    assert config.min_period == 27.0
#    assert config.ipower_costaper == 10
#    assert config.taper_percentage == 0.15
#    assert config.taper_type == 'hann'
#    assert config.use_cc_error


def test_multitaper_config_keys():
    default_args = inspect.getargspec(
        pyadjoint.ConfigMultiTaper.__init__).args
    default_args.remove("self")

    args = set([
        "max_period", "min_period", "lnpt", "transfunc_waterlevel",
        "water_threshold",
        "ipower_costaper", "min_cycle_in_window", "taper_type",
        "taper_percentage", "mt_nw", "num_taper", "dt_fac",
        "phase_step", "err_fac", "dt_max_scale", "measure_type",
        "dt_sigma_min", "dlna_sigma_min", "use_cc_error", "use_mt_error"])

    assert set(default_args) == args


def test_load_adjoint_config_yaml_for_multitaper_misfit():
    config = load_config_multitaper()
    assert isinstance(config, pyadjoint.ConfigMultiTaper)
    assert config.max_period == 60.0
    assert config.min_period == 27.0
    assert config.lnpt == 15
    assert config.transfunc_waterlevel == 1.0e-10
    assert config.water_threshold == 0.02
    assert config.ipower_costaper == 10
    assert config.min_cycle_in_window == 3
    assert config.taper_percentage == 0.3
    assert config.mt_nw == 4.0
    assert config.num_taper == 5
    assert config.phase_step == 1.5
    assert config.dt_fac == 2.0
    assert config.err_fac == 2.5
    assert config.dt_max_scale == 3.5
    assert config.measure_type == "dt"
    assert config.taper_type == 'hann'
    assert config.dt_sigma_min == 1.0
    assert config.dlna_sigma_min == 0.5
    assert config.use_cc_error
    assert not config.use_mt_error
