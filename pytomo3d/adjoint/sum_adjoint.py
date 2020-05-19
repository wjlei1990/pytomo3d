#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for sum_adjoint in pypaw

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import print_function, division, absolute_import
import numpy as np
import copy
from obspy import UTCDateTime
from pytomo3d.signal.rotate import rotate_one_station_stream
from pytomo3d.adjoint.process_adjsrc import convert_stream_to_adjs
from pytomo3d.adjoint.process_adjsrc import convert_adjs_to_stream
from pytomo3d.adjoint.process_adjsrc import add_missing_components
from pyadjoint import AdjointSource


def check_adj_consistency(adj_base, adj):
    """
    Check the consistency of adj_base and adj
    If passed, return, then adj could be added into adj_base
    If not, raise ValueError
    """
    if adj_base.network != adj.network or \
            adj_base.station != adj.station or \
            adj_base.component != adj.component:
        raise ValueError("Adjoint source network or station is different:"
                         "%s, %s" % (adj_base.id, adj.id))

    if not np.isclose(adj_base.dt, adj.dt):
        raise ValueError("DeltaT of current adjoint source(%f)"
                         "and new added adj(%f) not the same"
                         % (adj_base.dt, adj.dt))

    if np.abs(adj_base.starttime - adj.starttime) > 0.5 * adj.dt:
        raise ValueError("Start time of current adjoint source({})"
                         "and new added adj({}) not the same (dt: {})".format(
                            adj_base.starttime, adj.starttime, adj.dt))

    if len(adj_base.adjoint_source) != len(adj.adjoint_source):
        raise ValueError("Dimension of current adjoint_source(%d)"
                         "and new added adj(%d) not the same" %
                         (len(adj_base.adjoint_source),
                          len(adj.adjoint_source)))


def check_events_consistent(events):
    """
    Check all events are consistent(same with each other)
    """
    fn_base = list(events.keys())[0]
    event_base = events[fn_base]

    diffs = []
    for asdf_fn, event in events.items():
        if event_base != event:
            diffs.append(asdf_fn)

    if len(diffs) != 0:
        raise ValueError("Event information in %s not the same as others: %s"
                         % (diffs, fn_base))


def load_to_adjsrc(adj):
    """
    Load from asdf file adjoint source to pyadjoint.AdjointSources
    """
    starttime = UTCDateTime(adj.parameters["starttime"])
    _id = adj.parameters["station_id"]
    nw, sta = _id.split(".")
    comp = adj.parameters["component"]
    loc = adj.parameters["location"]

    new_adj = AdjointSource(adj.parameters["adjoint_source_type"],
                            adj.parameters["misfit"],
                            adj.parameters["dt"],
                            adj.parameters["min_period"],
                            adj.parameters["max_period"],
                            comp,
                            adjoint_source=np.array(adj.data),
                            network=nw, station=sta,
                            location=loc,
                            starttime=starttime)

    station_info = {"latitude": adj.parameters["latitude"],
                    "longitude": adj.parameters["longitude"],
                    "elevation_in_m": adj.parameters["elevation_in_m"],
                    "depth_in_m": adj.parameters["depth_in_m"],
                    "station": sta, "network": nw,
                    "location": loc}
    return new_adj, station_info


def dump_adjsrc(adj, station_info):
    """
    Combine the adj(pyadjoint.AdjointSource) and station information
    to form the adjoint content for ASDF file
    """
    adj_array = np.asarray(adj.adjoint_source, dtype=np.float32)
    station_id = "%s.%s" % (adj.network, adj.station)

    starttime = "T".join(str(adj.starttime).split())
    parameters = \
        {"dt": adj.dt, "starttime": starttime,
         "misfit": adj.misfit,
         "adjoint_source_type": adj.adj_src_type,
         "min_period": adj.min_period,
         "max_period": adj.max_period,
         "location": adj.location,
         "latitude": station_info["latitude"],
         "longitude": station_info["longitude"],
         "elevation_in_m": station_info["elevation_in_m"],
         "depth_in_m": station_info["depth_in_m"],
         "station_id": station_id, "component": adj.component,
         "units": "m"}

    adj_path = "%s_%s_%s" % (adj.network, adj.station, adj.component)

    return adj_array, adj_path, parameters


def create_weighted_adj(adj, weight):
    new_adj = copy.deepcopy(adj)
    new_adj.adjoint_source *= weight
    new_adj.misfit *= weight
    new_adj.location = ""
    return new_adj


def sum_adj_to_base(adj_base, adj, weight):
    check_adj_consistency(adj_base, adj)
    adj_base.adjoint_source += weight * adj.adjoint_source
    adj_base.misfit += weight * adj.misfit
    adj_base.min_period = min(adj.min_period, adj_base.min_period)
    adj_base.max_period = max(adj.max_period, adj_base.max_period)


def check_station_consistent(sta1, sta2):
    for key in sta1:
        if key == "location":
            # don't check location
            continue
        if key not in sta2:
            return False
        if isinstance(sta1[key], float):
            if not np.isclose(sta1[key], sta2[key]):
                return False
        else:
            if sta1[key] != sta2[key]:
                return False
    return True


def get_station_adjsrcs(adjsrcs, sta_tag):
    """
    Extract three components for a specific sta_tag
    """
    comp_list = ["MXR", "MXT", "MXZ"]
    adj_list = []
    for comp in comp_list:
        adj_name = "%s_%s" % (sta_tag, comp)
        if adj_name in adjsrcs:
            adj_list.append(adjsrcs[adj_name])
    return adj_list


def rotate_one_station_adjsrcs(sta_adjs, slat, slon, elat, elon):
    adj_stream, meta_info = convert_adjs_to_stream(sta_adjs)
    add_missing_components(adj_stream)

    rotate_one_station_stream(
        adj_stream, elat, elon, station_latitude=slat, station_longitude=slon,
        mode="RT->NE")

    new_adjs = convert_stream_to_adjs(adj_stream, meta_info)
    adj_dict = {}
    for _adj in new_adjs:
        adj_id = "%s_%s_%s" % (_adj.network, _adj.station, _adj.component)
        adj_dict[adj_id] = _adj
    return adj_dict


def rotate_adjoint_sources(old_adjs, stations, event_latitude,
                           event_longitude):
    print("="*15 + "\nRotate adjoint sources from RT to EN")
    done_sta_list = []
    new_adjs = {}

    for adj_id, adj in old_adjs.items():
        network = adj.network
        station = adj.station
        sta_tag = "%s_%s" % (network, station)

        if sta_tag not in done_sta_list:
            slat = stations[sta_tag]["latitude"]
            slon = stations[sta_tag]["longitude"]

            sta_adjs = get_station_adjsrcs(old_adjs, sta_tag)
            adj_dict = rotate_one_station_adjsrcs(
                sta_adjs, slat, slon, event_latitude,
                event_longitude)
            new_adjs.update(adj_dict)

    return new_adjs
