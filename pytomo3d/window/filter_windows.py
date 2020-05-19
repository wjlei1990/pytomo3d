#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filter the window based on the sensor type. For example, in long
period band(90-250s), we want to keep only STS-1 instrument windows.
For the input file, it requires 1) sensor type as json file; 2) windows
as json file. For the output, it is going to replace the origin window file
and keep a copy of original windows as "***.origin.json"

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import (absolute_import, division, print_function)
from pprint import pprint
from copy import deepcopy
import numpy as np


def is_right_sensor(sensor, sensor_types):
    """
    Check if sensor in the list of sensor_types
    """
    for stype in sensor_types:
        if stype in sensor:
            return True
    return False


def count_windows(windows):
    """
    Count the number of channels and windows
    :param windows:
    :return:
    """
    nchans = 0
    nwins = 0
    nwins_comp = {}
    for stainfo in windows.values():
        for chan, chaninfo in stainfo.items():
            comp = chan.split(".")[-1][-1]
            _nw = len(chaninfo)
            if _nw == 0:
                continue
            nchans += 1
            nwins += _nw
            if comp not in nwins_comp:
                nwins_comp[comp] = 0
            nwins_comp[comp] += _nw

    return nchans, nwins, nwins_comp


def print_window_filter_summary(old_windows, new_windows):
    nchans_old, nwins_old, nwins_comp_old = count_windows(old_windows)
    nchans_new, nwins_new, nwins_comp_new = count_windows(new_windows)
    if nchans_old:
        nchans_rej = (nchans_old - nchans_new) / nchans_old
    else:
        nchans_rej = 1.0
    if nwins_old:
        nwins_rej = (nwins_old - nwins_new) / nwins_old
    else:
        nwins_rej = 1.0
    print("-" * 10)
    print("Number of channels old and new: %d --> %d(rej: %.2f %%)"
          % (nchans_old, nchans_new, nchans_rej * 100))
    print("Number of windows old and new:  %d --> %d(rej: %.2f %%)"
          % (nwins_old, nwins_new, nwins_rej * 100))
    print("Old Component window counts: %s" % (nwins_comp_old))
    print("New Component window counts: %s" % (nwins_comp_new))
    return {"nchannels_old": nchans_old, "nwindows_old": nwins_old,
            "nchannels_new": nchans_new, "nwindows_new": nwins_new,
            "channel_rejection_percentage": '%.2f' % (nchans_rej * 100),
            "window_rejection_percentage": '%.2f' % (nwins_rej * 100)}


def filter_windows_on_sensors(windows, stations, sensor_types, verbose=False):
    """
    Filter the windows based on sensor types and station information.
    Only sensor types in 'sensor_types' will be kept.
    """
    new_wins = {}
    print("sensor_types: %s" % sensor_types)

    if verbose:
        print("channel name" + " " * 13 + "|" + " " * 30 +
              "sensor type | pick flag | wins | sum(chans, wins) |")

    total_chans = 0
    total_wins = 0
    for sta, sta_info in windows.items():
        sta_wins = {}
        for chan, chan_info in sta_info.items():
            pick_flag = False
            if len(chan_info) == 0:
                # if number of windows is 0
                continue
            try:
                # since windows are on RTZ component and
                # instruments are on NEZ compoennt, so
                # just use Z component instrument information
                zchan = chan[:-1] + "Z"
                _st = stations[zchan]["sensor"]
            except Exception as exp:
                print("Exception raised: {}".format(exp))
                continue
            if is_right_sensor(_st, sensor_types):
                sta_wins[chan] = chan_info
                pick_flag = True
                total_chans += 1
                total_wins += len(chan_info)
            if verbose:
                print("channel(%15s) | %40s |%11s|%9d| (%d, %d)"
                      % (chan, _st[:40], pick_flag, len(chan_info),
                         total_chans, total_wins))

        if len(sta_wins) > 0:
            new_wins[sta] = sta_wins

    return new_wins


def get_measurements_std(measurements):
    """
    Calculate the mean and standard deviation values from the measurements
    """
    comp_dt_meas = {}
    comp_dlna_meas = {}

    for sta_info in measurements.values():
        for chan, chan_info in sta_info.items():
            comp = chan.split(".")[-1][-1]

            dts = [v["dt"] for v in chan_info]
            dlnas = [v["dlna"] for v in chan_info]

            if comp not in comp_dt_meas:
                comp_dt_meas[comp] = []
            comp_dt_meas[comp].extend(dts)

            if comp not in comp_dlna_meas:
                comp_dlna_meas[comp] = []
            comp_dlna_meas[comp].extend(dlnas)

    dt_means = dict((k, np.mean(v)) for (k, v) in comp_dt_meas.items())
    dt_stds = dict((k, np.std(v)) for (k, v) in comp_dt_meas.items())

    dlna_means = dict((k, np.mean(v)) for (k, v) in comp_dlna_meas.items())
    dlna_stds = dict((k, np.std(v)) for (k, v) in comp_dlna_meas.items())

    print("means of dt measurements:")
    pprint(dt_means)
    print("std of dt measurements:")
    pprint(dt_stds)
    print("means of dlna measurements:")
    pprint(dlna_means)
    print("std of dlna measurements:")
    pprint(dlna_stds)

    return dt_means, dt_stds, dlna_means, dlna_stds


def get_user_bound(info):
    """ user specified bound for measurements filter """
    v = [info["tshift_reference"] - info["tshift_acceptance_level"],
         info["tshift_reference"] + info["tshift_acceptance_level"],
         info["dlna_reference"] - info["dlna_acceptance_level"],
         info["dlna_reference"] + info["dlna_acceptance_level"]]
    if v[0] > v[1] or v[2] > v[3]:
        raise ValueError("error on user bound: %s" % v)
    return v


def get_std_bound(mean, std, std_ratio):
    """ bound from mean and std values from measurements """
    b = [mean - std_ratio * std, mean + std_ratio * std]
    if b[0] > b[1]:
        raise ValueError("Error on std bound: %s" % b)
    return b


def filter_measurements_on_bounds(windows, measurements, bounds,
                                  comp_keep_flag=None):
    """
    filter the windows based on its measurements
    """
    def _filter_(chan_wins, chan_meas, _bounds):
        """ filter one channel window and measurements """
        new_wins = []
        new_meas = []
        if len(chan_wins) != len(chan_meas):
            raise ValueError("number of windows is not the same as number"
                             "measurements")
        for win, meas in zip(chan_wins, chan_meas):
            if (meas["dt"] >= _bounds["dt"][0]) \
                    and (meas["dt"] <= _bounds["dt"][1]) \
                    and (meas["dlna"] >= _bounds["dlna"][0]) \
                    and (meas["dlna"] <= _bounds["dlna"][1]):
                new_wins.append(deepcopy(win))
                new_meas.append(deepcopy(meas))

        if len(new_wins) != len(new_meas):
            raise ValueError("length of new_wins(%d) and new_meas(%d) not "
                             "the same!" % (len(new_wins), len(new_meas)))
        return new_wins, new_meas

    new_wins = {}
    new_meas = {}
    for sta, sta_info in windows.items():
        new_sta_wins = {}
        new_sta_meas = {}
        for chan, chan_info in sta_info.items():
            if len(chan_info) == 0:
                continue
            comp = chan.split(".")[-1][-1]
            if comp_keep_flag is not None and not comp_keep_flag[comp]:
                # throw comp if not keep
                continue
            m = measurements[sta][chan]
            w_f, m_f = _filter_(chan_info, m, bounds[comp])
            if len(w_f) == 0:
                continue
            new_sta_wins[chan] = w_f
            new_sta_meas[chan] = m_f

        if len(new_sta_wins) > 0:
            new_wins[sta] = new_sta_wins
            new_meas[sta] = new_sta_meas

    return new_wins, new_meas


def get_component_keep_flag(dt_means, dt_stds, dlna_means, dlna_stds,
                            comp_config):
    """
    Check if the mean and std of dt and dlna are within the range
    provided by the user. The default values are true. If the user
    doesn't provide any parameters, then all values will be kept
    as true
    """
    def _check_all_parameters_valid(_config):
        ts_range = _config["tshift_mean_range"]
        if ts_range[0] >= ts_range[1]:
            raise ValueError("tshift_mean_range error: %s" % ts_range)
        ts_std_level = _config["tshift_std_level"]
        if ts_std_level <= 0:
            raise ValueError("tshift_std_level(%f) <= 0" % ts_std_level)
        dlna_range = _config["dlna_mean_range"]
        if dlna_range[0] >= dlna_range[1]:
            raise ValueError("dlna_mean_range error: %s" % dlna_range)
        dlna_std_level = _config["dlna_std_level"]
        if dlna_std_level <= 0:
            raise ValueError("dlna_std_level(%f) <= 0" % dlna_std_level)

    def _check_in_range(v, vrange):
        if v > vrange[0] and v < vrange[1]:
            return True
        else:
            return False

    flags = {}
    for comp in dt_means:
        flags[comp] = True

    print("-" * 20)
    for comp in dt_means:
        if "tshift_mean_range" not in comp_config[comp]:
            # skip if user does not provide the parameter
            print("Not filtering on mean and std on component(%s) since"
                  "no user parameter(tshift_mean_range) provided" % comp)
            continue
        _config = comp_config[comp]
        _check_all_parameters_valid(_config)
        if not _check_in_range(dt_means[comp], _config["tshift_mean_range"]):
            print("Reject component(%s) due to dt mean(%f) out of range: %s"
                  % (comp, dt_means[comp], _config["tshift_mean_range"]))
            flags[comp] = False
            continue

        if dt_stds[comp] > _config["tshift_std_level"]:
            print("Reject component(%s) due to dt std(%f) larger than "
                  "threshold %f"
                  % (comp, dt_stds[comp], _config["tshift_std_level"]))
            flags[comp] = False
            continue

        if not _check_in_range(dlna_means[comp], _config["dlna_mean_range"]):
            print("Reject component(%s) due to dlna mean(%f) out of range: %s"
                  % (comp, dlna_means[comp], _config["dlna_mean_range"]))
            flags[comp] = False
            continue

        if dlna_stds[comp] > _config["dlna_std_level"]:
            print("Reject component(%s) due to dlna std(%f) larger than "
                  "threshold %f"
                  % (comp, dlna_stds[comp], _config["dlna_std_level"]))
            flags[comp] = False
            continue

    pprint("component keep flag:")
    pprint(flags)
    return flags


def get_measurement_final_bounds(
        comp_config, dt_means, dt_stds, dlna_means, dlna_stds):
    # get the measurement bounds
    final_bounds = {}
    for comp in dt_means:
        print("-" * 20 + "\nComponent: %s" % comp)
        user_bound = get_user_bound(comp_config[comp])
        dt_std_bound = get_std_bound(dt_means[comp], dt_stds[comp],
                                     comp_config[comp]["std_ratio"])
        dlna_std_bound = get_std_bound(dlna_means[comp], dlna_stds[comp],
                                       comp_config[comp]["std_ratio"])
        dt_bound = [max(dt_std_bound[0], user_bound[0]),
                    min(dt_std_bound[1], user_bound[1])]
        dlna_bound = [max(dlna_std_bound[0], user_bound[2]),
                      min(dlna_std_bound[1], user_bound[3])]
        final_bounds[comp] = {"dt": dt_bound, "dlna": dlna_bound}
        print("user specified bound [dt_min, dt_max, dlna_min, dlna_max]: %s"
              % user_bound)
        print("std bound dt: %s -- dlna: %s" % (dt_std_bound, dlna_std_bound))
        print("final dt bound: %s" % dt_bound)
        print("final dlna bound: %s" % dlna_bound)

    return final_bounds


def filter_windows_on_measurements(windows, measurements, measure_config):
    """
    Filter windows based on measurements and threshold specified by
    the user.
    """
    # calculate standard deviation for each component
    pprint("Config:")
    pprint(measure_config)

    if len(windows) == 0:
        return {}, {}

    dt_means, dt_stds, dlna_means, dlna_stds = \
        get_measurements_std(measurements)

    comp_config = measure_config['component']
    comp_keep_flag = get_component_keep_flag(
        dt_means, dt_stds, dlna_means, dlna_stds, comp_config)

    final_bounds = get_measurement_final_bounds(
        comp_config, dt_means, dt_stds, dlna_means, dlna_stds)

    new_wins, new_meas = filter_measurements_on_bounds(
        windows, measurements, final_bounds, comp_keep_flag=comp_keep_flag)

    return new_wins, new_meas


def check_consistency(windows, measurements):
    for sta, stainfo in windows.items():
        for chan, chan_info in stainfo.items():
            nwins_chan = len(chan_info)
            if nwins_chan == 0:
                continue
            try:
                nwins_meas = len(measurements[sta][chan])
            except KeyError as errmsg:
                raise KeyError("Missing %s(%s) in measurements: %s"
                               % (chan, sta, errmsg))
            if nwins_chan != nwins_meas:
                raise ValueError(
                    "Inconsistent between windows(%d) and measurements(%d)"
                    "from %s" % (nwins_chan, nwins_meas, chan))
    print("Pass the consistency check...")


def filter_windows(windows, stations, measurements, config, verbose=False):
    """
    Filter windows based on measurements and station information
    """
    check_consistency(windows, measurements)

    log = {}
    sensor_config = config["sensor"]
    if sensor_config["flag"]:
        print("=" * 10 + "  Filter on sensors  " + "=" * 10)
        windows_sensor = filter_windows_on_sensors(
            windows, stations, sensor_config["sensor_types"],
            verbose=verbose)
    else:
        windows_sensor = windows
    log["sensor"] = print_window_filter_summary(windows, windows_sensor)

    measure_config = config["measurement"]
    if measure_config["flag"]:
        print("=" * 10 + "  Filter on measurements  " + "=" * 10)
        windows_measure, measure_filtered = filter_windows_on_measurements(
            windows_sensor, measurements, measure_config)
    else:
        windows_measure = windows_sensor
        measure_filtered = measurements
    log["measurement"] = print_window_filter_summary(windows_sensor,
                                                     windows_measure)

    return windows_measure, measure_filtered, log
