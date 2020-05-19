"""
# functions mainly for pypaw.
"""
import numpy as np
from pytomo3d.utils.io import dump_json


def sort_windows_on_channel_and_location(sta_win):
    """
    functions for merge_instruments_window. the windows
    will be sorted based on [chan][location]

    :param sta_win:
    :return:
    """
    sort_dict = {}
    for trace_id, trace_win in sta_win.items():
        chan = trace_id.split('.')[-1][0:2]
        loc = trace_id.split('.')[-2]
        if chan not in sort_dict:
            sort_dict[chan] = {}
        if loc not in sort_dict[chan]:
            sort_dict[chan][loc] = {"traces": [], "nwins": 0}
        sort_dict[chan][loc]["traces"].append(trace_id)
        sort_dict[chan][loc]["nwins"] += len(trace_win)

    # sort to trace names in order
    for chan, chan_info in sort_dict.items():
        for loc, loc_info in chan_info.items():
            loc_info['traces'] = sorted(loc_info['traces'])

    return sort_dict


def pick_location_with_more_windows(sort_dict):
    choosen = {}
    for chan, chan_info in sort_dict.items():
        if len(chan_info.keys()) == 0:
            continue

        # if multiple locations available, choose the
        # one with most number of windows
        _locs = []
        _nwins = []
        for loc, loc_info in chan_info.items():
            _locs.append(loc)
            _nwins.append(loc_info["nwins"])
        _max_idx = np.array(_nwins).argmax()
        choosen[chan] = _locs[_max_idx]

    return choosen


def merge_instruments_window(sta_win):
    """
    Merge windows from the same channel, for example, if
    there are windows from "00.BH*" and "10.BH*", keep only one
    with the most windows. For example, if "00.BH*" has 10
    windows and "10.BH*" has 20 windows, we will keep the
    "10.BH*" since it has more windows.
    """
    if len(sta_win) == 0:
        return sta_win

    sort_dict = sort_windows_on_channel_and_location(sta_win)
    choosen_locs = pick_location_with_more_windows(sort_dict)

    choosen_wins = {}
    for chan, loc in choosen_locs.items():
        trace_list = sort_dict[chan][loc]["traces"]
        for tr_id in trace_list:
            choosen_wins[tr_id] = sta_win[tr_id]

    return choosen_wins


def sort_windows_on_channel(sta_win):
    """
    Gounp windows from one station into channels and count
    the number of windows in that channel
    :param sta_win:
    :return:
    """
    sort_dict = {}
    for trace_id, trace_win in sta_win.items():
        chan = trace_id.split(".")[-1][0:2]
        if chan not in sort_dict:
            sort_dict[chan] = {"traces": [], "nwins": 0}
        sort_dict[chan]["traces"].append(trace_id)
        sort_dict[chan]["nwins"] += len(trace_win)

    return sort_dict


def pick_channel_with_more_windows(sort_dict):
    max_wins = -1
    max_chan = None
    for chan, chaninfo in sort_dict.items():
        if chaninfo["nwins"] > max_wins:
            max_wins = chaninfo["nwins"]
            max_chan = chan
    return max_chan


def merge_channels_window(sta_win):
    """
    Merge windows from different channels.
    This step should be done after merge instruments windows
    because after that there will only one instrument left
    on one channel.
    For example, if we have "BH" channel with 20 windows and
    "LH" has 10 windows, we will keep only the "BH" channel.
    """
    if len(sta_win) == 0:
        return sta_win

    sort_dict = sort_windows_on_channel(sta_win)
    choosen_chan = pick_channel_with_more_windows(sort_dict)
    choosen_traces = sort_dict[choosen_chan]["traces"]

    choosen_wins = {}
    for _trace_id in choosen_traces:
        choosen_wins[_trace_id] = sta_win[_trace_id]

    return choosen_wins


def merge_station_windows(windows):
    """
    Merge windows for one station.
    For example, you may have "00.BH", "10.BH", "00.LH", "10.LH" from
    different locations and channels. You may only want to keep one
    at the very end. So:
    1) select locations: keep only one location with the most
        number of windows for one channel. For example, in "00.BH" and
        "10.BH" we only keep "10.BH".
    2) select channel: after last step, we keep only "00.BH" and "10.LH",
        but ultimately we only want to keep one. So we may only choose
        "00.BH" since it has more windows.
    """
    w = merge_instruments_window(windows)
    w = merge_channels_window(w)
    return w


def merge_windows(windows):
    """
    Merge the windows(from one event, multiple stations)
    """
    new_windows = {}
    for sta, sta_info in windows.items():
        if sta_info is None:
            continue
        # merge the windows for each station
        new_windows[sta] = merge_station_windows(sta_info)
    return new_windows


def generate_log_content(windows):
    overall_log = {"stations": 0, "stations_with_windows": 0,
                   "windows": 0, "traces": 0, "traces_with_windows": 0}
    comp_log = {}
    for sta_name, sta_win in windows.items():
        if sta_win is None:
            continue
        nwin_sta = 0
        ntraces_with_windows = 0
        for trace_id, trace_win in sta_win.items():
            comp = trace_id.split(".")[-1]
            if comp not in comp_log:
                comp_log[comp] = {
                    "windows": 0, "traces": 0, "traces_with_windows": 0}
            comp_log[comp]["windows"] += len(trace_win)
            if len(trace_win) > 0:
                comp_log[comp]["traces_with_windows"] += 1
                ntraces_with_windows += 1
            comp_log[comp]["traces"] += 1
            nwin_sta += len(trace_win)

        overall_log["stations"] += 1
        overall_log["windows"] += nwin_sta
        overall_log["traces"] += len(sta_win)
        overall_log["traces_with_windows"] += ntraces_with_windows
        if nwin_sta > 0:
            overall_log["stations_with_windows"] += 1

    log = {"component": comp_log, "overall": overall_log}
    return log


def stats_all_windows(windows, obsd_tag, synt_tag,
                      instrument_merge_flag,
                      output_file):
    """
    Generate window statistic information
    """
    log = {"obsd_tag": obsd_tag, "synt_tag": synt_tag,
           "instrument_merge_flag": instrument_merge_flag}

    window_log = generate_log_content(windows)
    log.update(window_log)

    print("Windows statistic log file: %s" % output_file)
    dump_json(log, output_file)

    return log
