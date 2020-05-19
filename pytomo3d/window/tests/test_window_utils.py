import os
import pytomo3d.window.utils as wu
from pytomo3d.utils.io import load_json


def test_sort_windows_on_channel_and_location():
    sta_win = {"II.AAK.10.BHZ": [1, 2, 3], "II.AAK.10.BHR": [1, 2],
               "II.AAK.10.BHT": [1], "II.AAK.00.BHZ": [1, 2, 3, 4],
               "II.AAK..BHZ": [1, 2, 3],
               "II.AAK..EHZ": [1, 2, 3, 4, 5, 6, 7, 8],
               "II.AAK..EHR": [1, 2, 3, 4],
               "II.AAK.10.EHZ": [1, 2],
               "II.AAK..LHZ": [1, 2, 3, 4]}
    results = wu.sort_windows_on_channel_and_location(sta_win)
    _true = {
        'BH': {
            '10': {'nwins': 6, 'traces': ['II.AAK.10.BHR', 'II.AAK.10.BHT',
                                          'II.AAK.10.BHZ']},
            '00': {'nwins': 4, 'traces': ['II.AAK.00.BHZ']},
            '': {'nwins': 3, 'traces': ['II.AAK..BHZ']}},
        'EH': {
            '': {'nwins': 12, 'traces': ['II.AAK..EHR', 'II.AAK..EHZ']},
            '10': {'nwins': 2, 'traces': ['II.AAK.10.EHZ']}},
        'LH': {
            '': {'nwins': 4, 'traces': ["II.AAK..LHZ"]}}}
    assert results == _true


def test_sort_windows_on_channel_and_location_2():
    sta_win = {"II.AAK.10.BHZ": [], "II.AAK.00.BHZ": [1, 2, 3]}
    results = wu.sort_windows_on_channel_and_location(sta_win)
    _true = {'BH': {'10': {'nwins': 0, 'traces': ['II.AAK.10.BHZ']},
                    '00': {'nwins': 3, 'traces': ['II.AAK.00.BHZ']}}}
    assert results == _true

    sta_win = {}
    results = wu.sort_windows_on_channel_and_location(sta_win)
    assert results == {}


def test_pick_location_with_more_windows():
    input = {
        "BH": {
            '': {'nwins': 6, 'traces': ["II.AAK..BHZ", "II.AAK..BHR"]},
            '10': {'nwins': 10, 'traces': ["II.AAK..BHR"]}},
        "LH": {
            '00': {'nwins': 3, 'traces': ["II.AAK..LHZ", "II.AAK..LHR",
                                          "II.AAK.LHT"]},
            '20': {'nwins': 2, 'traces': ["II.AAK.20.LHZ"]}},
        "EH": {
            '': {'nwins': 10, 'traces': ["II.AAK..LHR"]}}
    }
    choosen = wu.pick_location_with_more_windows(input)
    assert choosen == {"BH": '10', "EH": '', "LH": "00"}


def test_pick_location_with_more_windows_2():
    input = {
        "BH": {
            '': {'nwins': 0, 'traces': ["II.AAK..BHZ", "II.AAK..BHR"]}},
        "LH": {
            '00': {'nwins': 3, 'traces': ["II.AAK..LHZ", "II.AAK..LHR",
                                          "II.AAK.LHT"]},
            '20': {'nwins': 2, 'traces': ["II.AAK.20.LHZ"]}}
    }
    choosen = wu.pick_location_with_more_windows(input)
    _true = {'BH': '', 'LH': '00'}
    assert choosen == _true


def test_merge_instruments_windows():
    sta_win = {"II.AAK.10.BHZ": [1, 2, 3], "II.AAK.10.BHR": [1, 2],
               "II.AAK.10.BHT": [1], "II.AAK.00.BHZ": [1, 2, 3, 4],
               "II.AAK..BHZ": [1, 2, 3],
               "II.AAK..EHZ": [1, 2, 3, 4, 5, 6, 7, 8],
               "II.AAK..EHR": [1, 2, 3, 4],
               "II.AAK.10.EHZ": [1, 2],
               "II.AAK..LHZ": [1, 2, 3, 4]}

    choosen_wins = wu.merge_instruments_window(sta_win)
    _true = {
        'II.AAK.10.BHZ': [1, 2, 3], 'II.AAK.10.BHT': [1],
        'II.AAK.10.BHR': [1, 2],
        'II.AAK..EHR': [1, 2, 3, 4], 'II.AAK..EHZ': [1, 2, 3, 4, 5, 6, 7, 8],
        'II.AAK..LHZ': [1, 2, 3, 4]}
    assert choosen_wins == _true


def test_sort_windows_on_channel():
    sta_win = {"II.AAK.10.BHZ": [1, 2, 3], "II.AAK.10.BHR": [1, 2],
               "II.AAK.10.BHT": [1], "II.AAK.00.BHZ": [1, 2, 3, 4],
               "II.AAK..BHZ": [1, 2, 3],
               "II.AAK..EHZ": [1, 2, 3, 4, 5, 6, 7, 8],
               "II.AAK..EHR": [1, 2, 3, 4],
               "II.AAK.10.EHZ": [1, 2],
               "II.AAK..LHZ": [1, 2, 3, 4]}
    sort_dict = wu.sort_windows_on_channel(sta_win)
    _true = {
        'BH': {'nwins': 13, 'traces': ['II.AAK.10.BHR', 'II.AAK..BHZ',
                                       'II.AAK.10.BHT', 'II.AAK.10.BHZ',
                                       'II.AAK.00.BHZ']},
        'EH': {'nwins': 14, 'traces': ['II.AAK..EHR', 'II.AAK.10.EHZ',
                                       'II.AAK..EHZ']},
        'LH': {'nwins': 4, 'traces': ['II.AAK..LHZ']}}
    for comp in sort_dict:
        d1 = sort_dict[comp]
        d2 = _true[comp]
        assert d1["nwins"] == d2["nwins"]
        assert set(d1["traces"]) == set(d2["traces"])


def test_sort_windows_on_channel_2():
    sta_win = {"II.AAK.10.BHZ": [], "II.AAK.10.BHR": [1, 2, 3, 4]}
    sort_dict = wu.sort_windows_on_channel(sta_win)
    _true = {
        "BH": {"nwins": 4, "traces": ["II.AAK.10.BHZ", "II.AAK.10.BHR"]}
    }
    assert sort_dict == _true


def test_pick_channel_with_more_windows():
    input = {
        'BH': {'nwins': 13, 'traces': ['II.AAK.10.BHR', 'II.AAK..BHZ',
                                       'II.AAK.10.BHT', 'II.AAK.10.BHZ',
                                       'II.AAK.00.BHZ']},
        'EH': {'nwins': 10, 'traces': ['II.AAK..EHR', 'II.AAK.10.EHZ',
                                       'II.AAK..EHZ']},
        'LH': {'nwins': 4, 'traces': ['II.AAK..LHZ']}}

    results = wu.pick_channel_with_more_windows(input)
    assert results == "BH"

    input = {'BH': {'nwins': 0, 'traces': ['II.AAK.00.BHR']}}
    results = wu.pick_channel_with_more_windows(input)
    assert results == "BH"

    input = {}
    results = wu.pick_channel_with_more_windows(input)
    assert results is None


def test_merge_channels_window():
    sta_win = {"II.AAK.10.BHZ": [1, 2, 3], "II.AAK.10.BHR": [1, 2],
               "II.AAK.10.BHT": [1], "II.AAK.00.BHZ": [1, 2, 3, 4],
               "II.AAK..BHZ": [1, 2, 3],
               "II.AAK..EHZ": [1, 2, 3, 4, 5, 6, 7, 8],
               "II.AAK..EHR": [1, 2, 3, 4],
               "II.AAK.10.EHZ": [1, 2],
               "II.AAK..LHZ": [1, 2, 3, 4]}

    results = wu.merge_channels_window(sta_win)
    _true = {'II.AAK..EHZ': [1, 2, 3, 4, 5, 6, 7, 8],
             'II.AAK..EHR': [1, 2, 3, 4], 'II.AAK.10.EHZ': [1, 2]}
    assert results == _true


def test_merge_station_windows():
    sta_win = {"II.AAK.10.BHZ": [1, 2, 3], "II.AAK.10.BHR": [1, 2],
               "II.AAK.10.BHT": [1], "II.AAK.00.BHZ": [1, 2, 3, 4],
               "II.AAK..BHZ": [1, 2, 3],
               "II.AAK..EHZ": [1, 2, 3, 4, 5, 6, 7, 8],
               "II.AAK..EHR": [1, 2, 3, 4],
               "II.AAK.10.EHZ": [1, 2],
               "II.AAK..LHZ": [1, 2, 3, 4]}

    results = wu.merge_station_windows(sta_win)
    _true = {"II.AAK..EHZ": [1, 2, 3, 4, 5, 6, 7, 8],
             "II.AAK..EHR": [1, 2, 3, 4]}
    assert results == _true

    sta_win = {"II.AAK.10.BHZ": [1, 2, 3], "II.AAK.10.BHR": [1, 2],
               "II.AAK.10.BHT": [1], "II.AAK.00.BHZ": [1, 2, 3, 4],
               "II.AAK..BHZ": [1, 2, 3],
               "II.AAK..EHZ": [1, 2],
               "II.AAK..EHR": [1, 2, 3, 4],
               "II.AAK.10.EHZ": [1, 2],
               "II.AAK..LHZ": [1, 2, 3, 4]}

    results = wu.merge_station_windows(sta_win)
    _true = {"II.AAK.10.BHZ": [1, 2, 3], "II.AAK.10.BHR": [1, 2],
             "II.AAK.10.BHT": [1]}
    assert results == _true


def get_sample_windows():
    AAK_win = {"II.AAK.10.BHZ": [1, 2, 3], "II.AAK.10.BHR": [1, 2],
               "II.AAK.10.BHT": [1], "II.AAK.00.BHZ": [1, 2, 3, 4],
               "II.AAK..BHZ": [1, 2, 3],
               "II.AAK..EHZ": [1, 2],
               "II.AAK..EHR": [1, 2, 3, 4],
               "II.AAK.10.EHZ": [1, 2],
               "II.AAK..LHZ": [1, 2, 3, 4]}
    BBK_win = {"II.BBK..BHZ": [], "II.BBK..BHR": [1], "II.BBK..BHT": [1],
               "II.BBK.10.BHZ": [1, 2], "II.BBK.10.BHR": [],
               "II.BBK.10.BHT": [1],
               "II.BBK..LHZ": [1, 2, 3, 4], "II.BBK..LHR": [],
               "II.BBK..LHT": [1, 2, 3],
               "II.BBK.10.LHZ": [], "II.BBK.10.LHR": [1]}
    windows = {"II.AAK": AAK_win, "II.BBK": BBK_win}
    return windows


def test_merge_windows():
    windows = get_sample_windows()
    results = wu.merge_windows(windows)
    _true = {"II.BBK": {"II.BBK..LHZ": [1, 2, 3, 4], "II.BBK..LHR": [],
                        "II.BBK..LHT": [1, 2, 3]},
             "II.AAK": {"II.AAK.10.BHZ": [1, 2, 3], "II.AAK.10.BHR": [1, 2],
                        "II.AAK.10.BHT": [1]}}
    assert results == _true


def test_generate_log_content():
    windows = {"II.BBK": {"II.BBK..BHZ": [1, 2, 3, 4], "II.BBK..BHR": [],
                          "II.BBK..BHT": [1, 2, 3]},
               "II.AAK": {"II.AAK.10.BHZ": [1, 2, 3], "II.AAK.10.BHR": [1, 2],
                          "II.AAK.10.BHT": [1]},
               "II.CCK": {}}
    log = wu.generate_log_content(windows)
    _true = {
        'overall': {'windows': 13, 'stations_with_windows': 2,
                    'traces_with_windows': 5, 'traces': 6, 'stations': 3},
        'component': {
            'BHR': {'windows': 2, 'traces_with_windows': 1, 'traces': 2},
            'BHT': {'windows': 4, 'traces_with_windows': 2, 'traces': 2},
            'BHZ': {'windows': 7, 'traces_with_windows': 2, 'traces': 2}}}

    assert log == _true


def test_stats_all_windows(tmpdir):
    windows = {"II.BBK": {"II.BBK..BHZ": [1, 2, 3, 4], "II.BBK..BHR": [],
                          "II.BBK..BHT": [1, 2, 3]},
               "II.AAK": {"II.AAK.10.BHZ": [1, 2, 3], "II.AAK.10.BHR": [1, 2],
                          "II.AAK.10.BHT": [1]},
               "II.CCK": {}}

    outputfile = os.path.join(str(tmpdir), "windows.log.json")
    wu.stats_all_windows(windows, "proc_obsd_17_40", "proc_synt_17_40",
                         True, outputfile)

    log = load_json(outputfile)
    _true = {
        "component": {
            "BHR": {
                "traces": 2, "traces_with_windows": 1, "windows": 2},
            "BHT": {
                "traces": 2, "traces_with_windows": 2, "windows": 4},
            "BHZ": {
                "traces": 2, "traces_with_windows": 2, "windows": 7}},
        "overall": {
            "stations": 3, "stations_with_windows": 2, "traces": 6,
            "traces_with_windows": 5, "windows": 13},
        "instrument_merge_flag": True,
        "obsd_tag": "proc_obsd_17_40",
        "synt_tag": "proc_synt_17_40"
    }
    assert log == _true
