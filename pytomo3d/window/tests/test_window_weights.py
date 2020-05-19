import os
import inspect
import json
import numpy.testing as npt
import pytest

from spaceweight import SpherePoint
import pytomo3d.window.window_weights as ww


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
DATA_DIR = os.path.join(TESTBASE_DIR, "tests", "data")


def load_json(filename):
    with open(filename) as fh:
        return json.load(fh)


window_file = os.path.join(DATA_DIR, "window", "windows.fake.json")
windows = load_json(window_file)
station_file = os.path.join(DATA_DIR, "stations", "stations.fake.json")
stations = load_json(station_file)


def test_calculate_receiver_window_counts():
    rec_counts, cat_wcounts = ww.calculate_receiver_window_counts(windows)

    _true = {"BHZ": {"II.AAK..BHZ": 3, "II.ABKT..BHZ": 2, "IU.BCD..BHZ": 5},
             "BHR": {"II.AAK..BHR": 2, "II.ABKT..BHR": 1, "IU.BCD..BHR": 2},
             "BHT": {"II.AAK..BHT": 1, "IU.BCD..BHT": 2}}
    assert rec_counts == _true

    _true = {"BHZ": 10, "BHR": 5, "BHT": 3}
    assert cat_wcounts == _true


def test_assign_receiver_points():
    rec_counts, _ = ww.calculate_receiver_window_counts(windows)

    def _assert(_points):
        for p in _points:
            if p.tag == "II.AAK..BHZ":
                npt.assert_almost_equal(p.latitude, 0.0)
                npt.assert_almost_equal(p.longitude, 0.0)
            elif p.tag == "II.ABKT..BHZ":
                npt.assert_almost_equal(p.latitude, 0.0)
                npt.assert_almost_equal(p.longitude, 120.0)
            elif p.tag == "IU.BCD..BHZ":
                npt.assert_almost_equal(p.latitude, 0.0)
                npt.assert_almost_equal(p.longitude, -120.0)

    points = ww.assign_receiver_to_points(rec_counts["BHZ"], stations)
    assert len(points) == 3
    _assert(points)

    points = ww.assign_receiver_to_points(rec_counts["BHR"], stations)
    assert len(points) == 3
    _assert(points)

    points = ww.assign_receiver_to_points(rec_counts["BHT"], stations)
    assert len(points) == 2
    _assert(points)


def test_get_receiver_weights():
    center = SpherePoint(0, 0, tag="source")

    rec_counts, _ = ww.calculate_receiver_window_counts(windows)

    points = ww.assign_receiver_to_points(rec_counts["BHZ"], stations)
    ref_distance, cond_number = ww.get_receiver_weights(
        "BHZ", center, points, 0.35, plot=False)
    for p in points:
        npt.assert_almost_equal(p.weight, 1.0)
    npt.assert_almost_equal(cond_number, 1.0)

    points = ww.assign_receiver_to_points(rec_counts["BHT"], stations)
    ref_distance, cond_number = ww.get_receiver_weights(
        "BHZ", center, points, 0.35, plot=False)
    for p in points:
        npt.assert_almost_equal(p.weight, 1.0)
    npt.assert_almost_equal(cond_number, 1.0)


def test_normalize_receiver_weights():
    rec_counts, cat_wcounts = ww.calculate_receiver_window_counts(windows)

    comp = "BHZ"
    channels = list(rec_counts[comp].keys())
    channels.sort()
    points = ww.assign_receiver_to_points(channels, stations)
    weights = ww.normalize_receiver_weights(points, rec_counts[comp])
    assert len(weights) == 3
    for v in weights.values():
        npt.assert_almost_equal(v, 1.0)

    points[0].weight = 0.5
    points[1].weight = 0.75
    points[2].weight = 1.0
    weights = ww.normalize_receiver_weights(points, rec_counts[comp])
    assert len(weights) == 3
    npt.assert_almost_equal(weights["II.AAK..BHZ"], 0.625)
    npt.assert_almost_equal(weights["II.ABKT..BHZ"], 0.9375)
    npt.assert_almost_equal(weights["IU.BCD..BHZ"], 1.25)


def test_determin_receiver_weighting():
    src = {"latitude": 0.0, "longitude": 0.0}
    results = ww.determine_receiver_weighting(
        src, stations, windows, search_ratio=0.35, weight_flag=True,
        plot_flag=False)

    assert len(results) == 5


def test_receiver_validator():
    src = {"latitude": 0.0, "longitude": 0.0}
    results = ww.determine_receiver_weighting(
        src, stations, windows, search_ratio=0.35, weight_flag=True,
        plot_flag=False)

    weights = results["rec_weights"]

    weights["BHZ"]["II.AAK..BHZ"] *= 2
    rec_counts, cat_wcounts = ww.calculate_receiver_window_counts(windows)
    with pytest.raises(ValueError):
        ww._receiver_validator(weights["BHZ"], rec_counts["BHZ"],
                               cat_wcounts["BHZ"])


def test_normalize_category_weights():
    cat_wcounts = {"17_40":  {"BHR": 8, "BHT": 4, "BHZ": 16},
                   "40_100": {"BHR": 4, "BHT": 2, "BHZ": 4},
                   "90_250": {"BHR": 1, "BHT": 1, "BHZ": 2}}

    category_ratio = {"17_40":  {"BHR": 0.125, "BHT": 0.25, "BHZ": 0.125},
                      "40_100": {"BHR": 0.25, "BHT": 0.5, "BHZ": 0.25},
                      "90_250": {"BHR": 1, "BHT": 1, "BHZ": 0.5}}

    weights = ww.normalize_category_weights(category_ratio, cat_wcounts)

    _true = {'17_40': {'BHR': 0.525, 'BHT': 1.05, 'BHZ': 0.525},
             '40_100': {'BHR': 1.05, 'BHT': 2.1, 'BHZ': 1.05},
             '90_250': {'BHR': 4.2, 'BHT': 4.2, 'BHZ': 2.1}}

    assert weights == _true


def test_calculate_receiver_weights_interface():
    pass


def test_check_category_ratio_consistency():
    ratio = {
        "17_40": {"BHR": 1.0, "BHT": 1.0, "BHZ": 1.0},
        "40_100": {"BHR": 1.0, "BHT": 1.0, "BHZ": 1.0},
        "90_250": {"BHR": 1.0, "BHT": 1.0, "BHZ": 1.0}
    }
    wcounts = {
        "17_40": {"BHR": 100, "BHT": 100, "BHZ": 100},
        "40_100": {"BHR": 100, "BHT": 100, "BHZ": 100},
        "90_250": {"BHR": 100, "BHT": 100, "BHZ": 100}
    }
    ww.check_category_ratio_consistency(ratio, wcounts)

    wcounts["90_250"].pop("BHZ", None)
    with pytest.raises(ValueError):
        ww.check_category_ratio_consistency(ratio, wcounts)


def test_determin_category_weighting():
    cat_wcounts = {"17_40":  {"BHR": 8, "BHT": 4, "BHZ": 16},
                   "40_100": {"BHR": 4, "BHT": 2, "BHZ": 4},
                   "90_250": {"BHR": 1, "BHT": 1, "BHZ": 2}}

    category_ratio = {"17_40":  {"BHR": 1, "BHT": 2, "BHZ": 1},
                      "40_100": {"BHR": 2, "BHT": 4, "BHZ": 2},
                      "90_250": {"BHR": 8, "BHT": 8, "BHZ": 4}}
    category_param = {"flag": True, "ratio": category_ratio}

    weights = ww.calculate_category_weights_interface(
        category_param, cat_wcounts)

    _true = {'17_40': {'BHR': 0.525, 'BHT': 1.05, 'BHZ': 0.525},
             '40_100': {'BHR': 1.05, 'BHT': 2.1, 'BHZ': 1.05},
             '90_250': {'BHR': 4.2, 'BHT': 4.2, 'BHZ': 2.1}}

    assert weights == _true


def test_category_validator():

    cat_wcounts = {"17_40":  {"BHR": 8, "BHT": 4, "BHZ": 16},
                   "40_100": {"BHR": 4, "BHT": 2, "BHZ": 4},
                   "90_250": {"BHR": 1, "BHT": 1, "BHZ": 2}}

    category_ratio = {"17_40":  {"BHR": 1, "BHT": 2, "BHZ": 1},
                      "40_100": {"BHR": 2, "BHT": 4, "BHZ": 2},
                      "90_250": {"BHR": 8, "BHT": 8, "BHZ": 4}}
    category_param = {"flag": True, "ratio": category_ratio}

    weights = ww.calculate_category_weights_interface(
        category_param, cat_wcounts)

    weights["17_40"]["BHR"] *= 2
    with pytest.raises(ValueError):
        ww._category_validator(weights, cat_wcounts)


def test_combine_receiver_and_category_weights():
    pass
