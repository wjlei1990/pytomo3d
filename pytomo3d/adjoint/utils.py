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
import numpy as np
from pyadjoint import AdjointSource


def ensemble_fake_adj(stream):
    """
    Ensemble fake adjoint sources from stream, for test purpose.
    """
    adjsrcs = list()
    comps = ["Z", "R", "T"]
    for comp in comps:
        tr = stream.select(channel="*%s" % comp)[0]
        adj = AdjointSource(
            "waveform_misfit", misfit=0.0, dt=tr.stats.delta,
            min_period=50.0, max_period=100.0,
            component=tr.stats.channel,
            adjoint_source=tr.data, network=tr.stats.network,
            station=tr.stats.station, location=tr.stats.location,
            starttime=tr.stats.starttime)
        adjsrcs.append(adj)

    return adjsrcs


def change_adjsrc_channel_name(adjsrcs, channel):
    """
    Change adjoint source channel name to given string. For example,
    as specfem input, the channel name is "MX". So before writing
    adjoint source out, we need to change the channel name to "MX"
    """
    if len(channel) != 2:
        raise ValueError("channel(%s) must be length of 2" % channel)
    for adj in adjsrcs:
        adj.component = channel + adj.component[-1]


def _stats_channel_window(adjsrcs, windows):
    """
    Determine number of windows on each channel of each component.
    """
    adj_dict = {}
    for idx, adj in enumerate(adjsrcs):
        adj_id = "%s.%s.%s.%s" % (adj.network, adj.station, adj.location,
                                  adj.component)
        adj_dict[adj_id] = idx

    adj_win_dict = {}
    for chan_win in windows.values():
        if len(chan_win) == 0:
            continue
        chan_id = chan_win[0]["channel_id"]
        adj_win_dict[chan_id] = len(chan_win)

    new_win_dict = {}
    for key in adj_win_dict:
        if key in adj_dict:
            new_win_dict[key] = adj_win_dict[key]

    return adj_dict, new_win_dict


def calculate_chan_weight(adjsrcs, windows_sta):
    """
    Calcualte window weights based on adjoint sources and windows

    :param adjsrcs:
    :param windows_sta:
    :return:
    """

    _, adj_win_dict = _stats_channel_window(adjsrcs, windows_sta)

    comp_dict = {}
    for tr_id, nwins in adj_win_dict.items():
        comp = "MX%s" % tr_id.split(".")[-1][-1]
        if comp not in comp_dict:
            comp_dict[comp] = {}
        comp_dict[comp][tr_id] = nwins

    for comp, comp_wins in comp_dict.items():
        ntotal = 0
        for chan_id, chan_win in comp_wins.items():
            ntotal += chan_win
        for chan_id, chan_win in comp_wins.items():
            comp_dict[comp][chan_id] = chan_win / ntotal

    return comp_dict


def check_multiple_instruments(adjsrcs):
    """
    Check if there are mutiple instruments for one component
    This is very important because if there is only one instrument
    for one component, we can define the path shorter and change
    channel name to "MX" to follow the specfem style. For example,
    in adjsrcs, if there are only "II.AAK.00.BHZ" in component Z,
    then we can define the path as "II_AAK_MXZ". If there are
    multiple instruments, then we define the whole path, as
    "II_AAK_00_BHZ"
    """
    name_list = []
    adj_dict = {}
    for adj in adjsrcs:
        cat = adj.component[-1]
        if cat not in adj_dict:
            adj_dict[cat] = []
        adj_id = "%s.%s.%s.%s" % (adj.network, adj.station,
                                  adj.location, adj.component)
        adj_dict[cat].append(adj_id)
        name_list.append(adj_id)

    if len(set(name_list)) != len(name_list):
        raise ValueError("Error on adjoint source(%s.%s) since it has"
                         "duplicate name on adjoint source: %s"
                         % (adj.network, adj.station, name_list))

    _flag = False
    for cat_info in adj_dict.values():
        if len(cat_info) > 1:
            _flag = True
            break
    return _flag


def reshape_adj(adjsrcs, staxml, dtype=np.float32,
                default_specfem_channel="MX"):
    """
    Reshape adjsrcs to a certain structure required by pyasdf writer
    """
    if not isinstance(adjsrcs, list):
        raise ValueError("Input ajdsrcs must be a list of adjoint sources")

    vtype = "AuxiliaryData"
    reshape_list = []
    tag_list = []

    # extract station information
    sta_lat = staxml[0][0].latitude
    sta_lon = staxml[0][0].longitude
    sta_ele = staxml[0][0].elevation
    sta_dep = staxml[0][0][0].depth

    # sanity check to see if there are multiple instruments
    _multiple_flag = check_multiple_instruments(adjsrcs)

    if not _multiple_flag:
        change_adjsrc_channel_name(adjsrcs, default_specfem_channel)

    for adj in adjsrcs:
        adj_array = np.asarray(adj.adjoint_source, dtype=dtype)

        station_id = "%s.%s" % (adj.network, adj.station)

        starttime = "T".join(str(adj.starttime).split())
        parameters = {"dt": adj.dt, "starttime": starttime,
                      "misfit": adj.misfit,
                      "adjoint_source_type": adj.adj_src_type,
                      "min_period": adj.min_period,
                      "max_period": adj.max_period,
                      "latitude": sta_lat, "longitude": sta_lon,
                      "elevation_in_m": sta_ele, "depth_in_m": sta_dep,
                      "station_id": station_id, "component": adj.component,
                      "location": adj.location,
                      "units": "m"}

        if _multiple_flag:
            tag = "%s_%s_%s_%s" % (adj.network, adj.station,
                                   adj.location, adj.component)
        else:
            tag = "%s_%s_%s" % (adj.network, adj.station, adj.component)

        tag_list.append(tag)

        dataset_path = "AdjointSources/%s" % tag

        _reshape = {"object": adj_array, "type": vtype,
                    "path": dataset_path, "parameters": parameters}

        reshape_list.append(_reshape)

    # check if there are different adjoint sources with the same tag. If so,
    # the writer won't be able to write out because of the same dataset path
    if len(set(tag_list)) != len(tag_list):
        raise ValueError("Duplicate tag in adjoint sources list: %s" %
                         tag_list)

    return reshape_list
