#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Methods that handles adjoint sources

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import (print_function, division)
import yaml
import numpy as np
import pyadjoint


def load_adjoint_config_yaml(filename):
    """
    load yaml and setup pyadjoint.Config object
    """
    with open(filename) as fh:
        data = yaml.load(fh, Loader=yaml.FullLoader)

    adjsrc_type = data["adj_src_type"]
    data.pop("adj_src_type")

    if adjsrc_type == "multitaper_misfit":
        ConfigClass = pyadjoint.ConfigMultiTaper
    elif adjsrc_type == "cc_traveltime_misfit":
        ConfigClass = pyadjoint.ConfigCrossCorrelation
    elif adjsrc_type == "waveform_misfit":
        ConfigClass = pyadjoint.ConfigWaveForm

    if data["min_period"] > data["max_period"]:
        raise ValueError("min_period is larger than max_period in config "
                         "file: %s" % filename)

    return ConfigClass(**data)


def _extract_window_id(windows):
    """
    Extract obsd id and synt id associated with the windows.
    Windows should come from the same channel.

    :param windows: a list of pyflex.Window
    :return: a two dimension numpy.array of time window, with window
        starttime and endtime
    """
    obs_ids = []
    syn_ids = []
    for _win in windows:
        if isinstance(_win, dict):
            obs_id = _win["channel_id"]
            try:
                syn_id = _win["channel_id_2"]
            except Exception as exp:
                print("Missing channel_id_2: {}".format(exp))
                syn_id = "UNKNOWN"
        else:
            obs_id = _win.channel_id
            try:
                syn_id = _win.channel_id_2
            except Exception as exp:
                print("Missing channel_id_2 property: {}".format(exp))
                syn_id = "UNKNOWN"
        obs_ids.append(obs_id)
        syn_ids.append(syn_id)

    # sanity check for windows in the same channel
    if len(set(obs_ids)) != 1:
        raise ValueError("Windows in for the same channel not consistent for"
                         "obsd id:%s" % obs_ids)
    if len(set(syn_ids)) != 1:
        raise ValueError("Windows in for the same channel not consistent for"
                         "obsd id:%s" % syn_ids)

    obs_id = obs_ids[0]
    syn_id = syn_ids[0]
    # read windows for this trace
    return obs_id, syn_id


def _extract_window_time(windows):
    """
    Extract window time information from a list of windows.
    """
    win_time = []
    for _win in windows:
        if isinstance(_win, dict):
            win_time.append([_win["relative_starttime"],
                             _win["relative_endtime"]])
        else:
            win_time.append([_win.relative_starttime,
                             _win.relative_endtime])
    return np.array(win_time)
