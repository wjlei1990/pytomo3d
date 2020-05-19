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
from copy import deepcopy
from obspy import Stream, Trace
from obspy.geodetics import gps2dist_azimuth
from pyadjoint import AdjointSource
from pytomo3d.signal.process import filter_trace
from pytomo3d.signal.rotate import rotate_stream


def calculate_baz(elat, elon, slat, slon):
    """
    Calculate back azimuth(station to event azimuth)

    :param elat: event latitude
    :param elon: event longitude
    :param slat: station latitude
    :param slon: station longitude
    :return: back azimuth
    """
    _, _, baz = gps2dist_azimuth(elat, elon, slat, slon)

    return baz


def change_channel_name(stream, channel_name):
    if not isinstance(channel_name, str):
        raise TypeError("Incorrect type of channel_name: %s"
                        % type(channel_name))
    if len(channel_name) != 2:
        raise ValueError("Length of channel name(%s) should be 2"
                         % channel_name)
    for tr in stream:
        tr.stats.channel = channel_name + tr.stats.channel[-1]


def time_reverse_array(stream):
    """
    Time reverse the data array of stream. Pyadjoint output
    is time-reversed version of adjoint source. However, as
    SPECFEM adjoint input, we need to time reversed again.
    Also, for the zero-paddding and interpolation, it is
    necessary to reverse the data back.
    """
    for tr in stream:
        tr.data = tr.data[::-1]


def convert_adj_to_trace(adj):
    """
    Convert AdjointSource to Trace,for internal use only
    """
    meta = {}

    tr = Trace()
    tr.data = deepcopy(adj.adjoint_source)
    tr.stats.starttime = adj.starttime
    tr.stats.delta = adj.dt

    tr.stats.channel = adj.component
    tr.stats.station = adj.station
    tr.stats.network = adj.network
    tr.stats.location = adj.location

    meta["adj_src_type"] = adj.adj_src_type
    meta["misfit"] = adj.misfit
    meta["min_period"] = adj.min_period
    meta["max_period"] = adj.max_period

    return tr, meta


def convert_adjs_to_stream(adjsrcs):
    """
    Convert adjoint sources into stream. So all obspy
    tools will be available for processing. Return values
    for adjoint stream and other information
    """
    meta_info = {}
    adj_stream = Stream()
    for adj in adjsrcs:
        _tr, _meta = convert_adj_to_trace(adj)
        adj_stream.append(_tr)
        meta_info[_tr.id] = _meta
    return adj_stream, meta_info


def convert_trace_to_adj(tr, meta):
    """
    Convert Trace to AdjointSource, for internal use only, with
    meta data information
    """
    minp = meta["min_period"]
    maxp = meta["max_period"]
    adj_src_type = meta["adj_src_type"]
    misfit = meta["misfit"]

    dt = tr.stats.delta
    component = tr.stats.channel
    adj = AdjointSource(adj_src_type, misfit, dt, minp, maxp, component)

    adj.adjoint_source = deepcopy(tr.data)
    adj.station = tr.stats.station
    adj.network = tr.stats.network
    adj.location = tr.stats.location
    adj.starttime = tr.stats.starttime

    return adj


def convert_stream_to_adjs(stream, meta_info):
    """
    Convert stream to adjoint sources. Be careful about
    the meta information here. If the adjoint source
    has been rotated, from EN to RT, then the misfit
    in meta information would be wrong.
    """
    adjsrcs = []

    _key = list(meta_info.keys())[0]
    _default_meta = meta_info[_key].copy()
    _default_meta["misfit"] = 0.0
    for _tr in stream:
        try:
            _meta = meta_info[_tr.id]
        except KeyError:
            _meta = _default_meta
        adj = convert_trace_to_adj(_tr, _meta)
        adjsrcs.append(adj)
    return adjsrcs


def zero_padding_stream(stream, starttime, endtime):
    """
    Zero padding the stream to time [starttime, endtime]. The time
    might not be precise due to the sampling rate. We usually cut
    one more point to make sure contains the time range. If you
    want to get precise starttim and endtime, you need to zero
    padding first and then interpolate the stream to the exact
    time.
    """
    if starttime > endtime:
        raise ValueError("Starttime is larger than endtime: [%f, %f]"
                         % (starttime, endtime))

    for tr in stream:
        dt = tr.stats.delta
        npts = tr.stats.npts
        tr_starttime = tr.stats.starttime
        tr_endtime = tr.stats.endtime

        npts_before = int((tr_starttime - starttime) / dt) + 1
        npts_before = max(npts_before, 0)
        npts_after = int((endtime - tr_endtime) / dt) + 1
        npts_after = max(npts_after, 0)

        # recalculate the time for padding trace
        padding_starttime = tr_starttime - npts_before * dt
        padding_array = np.zeros(npts_before + npts + npts_after)
        padding_array[npts_before:(npts_before + npts)] = \
            tr.data[:]

        tr.data = padding_array
        tr.stats.starttime = padding_starttime


def sum_adjoint_no_weighting(adj_stream, meta_info):
    """
    Add same components in adjoint source together without
    extra weight, i.e., equal weight.

    :param adj_stream:
    :param meta_info:
    :return:
    """
    new_stream = Stream()
    new_meta = {}
    done_comps = []
    for tr in adj_stream:
        comp = tr.stats.channel[-1]
        # print(comp, done_comps)
        if comp not in done_comps:
            done_comps.append(comp)
            comp_tr = tr.copy()
            comp_tr.stats.location = ""
            comp_tr.stats.channel = "MX" + comp
            new_stream.append(comp_tr)
            new_meta[comp_tr.id] = deepcopy(meta_info[tr.id])
        else:
            comp_tr = new_stream.select(component=comp)[0]
            comp_tr.data += tr.data
            new_meta[comp_tr.id]["misfit"] += meta_info[tr.id]["misfit"]

    return new_stream, new_meta


def sum_adjoint_with_weighting(adj_stream, meta_info, weight_dict):
    new_stream = Stream()
    new_meta = {}
    done_comps = []
    # sum using components weight
    for comp, comp_weights in weight_dict.items():
        for chan_id, chan_weight in comp_weights.items():
            if comp not in done_comps:
                done_comps.append(comp)
                adj_tr = adj_stream.select(id=chan_id)[0]
                comp_tr = adj_tr.copy()
                comp_tr.data *= chan_weight
                comp_tr.stats.location = ""
                comp_tr.stats.channel = comp
                new_stream.append(comp_tr)
                new_meta[comp_tr.id] = meta_info[adj_tr.id].copy()
                new_meta[comp_tr.id]["misfit"] = \
                    chan_weight * meta_info[adj_tr.id]["misfit"]
            else:
                adj_tr = adj_stream.select(id=chan_id)[0]
                comp_tr = new_stream.select(channel="*%s" % comp)[0]
                comp_tr.data += chan_weight * adj_tr.data
                new_meta[comp_tr.id]["misfit"] += \
                    chan_weight * meta_info[adj_tr.id]["misfit"]
    return new_stream, new_meta


def sum_adj_on_component(adj_stream, meta_info, weight_flag=False,
                         weight_dict=None):
    """
    Sum adjoint source on different channels but same component
    together, like "II.AAK.00.BHZ" and "II.AAK.10.BHZ" to form
    "II.AAK.MXZ". Also, misfit values will be added accordingly.
    Please remember after adding, the channel will be renamed to
    "MX" by default if weight_flag is False. If weight_flag is true,
    then the channel is depandant on the dict key.

    :param adj_stream: adjoint source stream
    :param weight_dict: weight dictionary, should be something like
        {"MXZ":{"II.AAK.00.BHZ": 0.5, "II.AAK.10.BHZ": 0.5},
         "MXR":{"II.AAK.00.BHR": 0.3, "II.AAK.10.BHR": 0.7},
         "MXT":{"II.AAK..BHT": 1.0}}
    :return: summed adjoint source stream
    """
    if weight_flag and weight_dict is None:
        raise ValueError("weight_dict should be assigned if you want use"
                         "weighting")
    if weight_flag:
        return sum_adjoint_with_weighting(adj_stream, meta_info, weight_dict)
    else:
        return sum_adjoint_no_weighting(adj_stream, meta_info)


def add_missing_components(stream, component_list=["Z", "R", "T"]):
    """
    Add zero-trace to stream if one component is missing.
    Usually, this shouldn't not be done for a stream. However,
    when using the adjoint source, we don't want to miss any
    measurements, so we just set the missing components to
    zero trace
    """
    done_list = []

    nadds = 0
    for tr in stream:
        nw = tr.stats.network
        sta = tr.stats.station
        loc = tr.stats.location
        station_id = "%s.%s.%s" % (nw, sta, loc)
        if station_id not in done_list:
            stream_sta = stream.select(network=nw, station=sta, location=loc)
        else:
            continue

        missinglist = component_list[:]
        # search for missing location id list
        for tr in stream_sta:
            missinglist.remove(tr.stats.channel[-1])

        tr_template = stream_sta[0]
        for component in missinglist:
            nadds += 1
            zero_trace = tr_template.copy()
            zero_trace.data.fill(0.0)
            zero_trace.stats.channel = \
                "%s%s" % (tr_template.stats.channel[0:2], component)
            stream.append(zero_trace)

    return nadds


def rotate_adj_stream(adj_stream, elatitude, elongitude, inventory):
    """
    Rotate adjoint stream from "RT" to "EN"
    """
    if inventory is None:
        raise ValueError("Station must be provied to rotate the"
                         "adjoint source")
    rotate_stream(adj_stream, elatitude, elongitude, inventory,
                  mode="RT->NE")


def interp_adj_stream(adj_stream, interp_starttime=None, interp_delta=None,
                      interp_npts=None):
    """
    Interpolate the adjoint stream
    """
    # zero padding the adjoint source to the given length
    interp_endtime = interp_starttime + interp_delta * interp_npts
    zero_padding_stream(adj_stream, interp_starttime, interp_endtime)

    # interpolate precisely
    adj_stream.interpolate(sampling_rate=1.0/interp_delta,
                           starttime=interp_starttime,
                           npts=interp_npts)


def process_adjoint(adjsrcs, interp_flag=False, interp_starttime=None,
                    interp_delta=None, interp_npts=None,
                    sum_over_comp_flag=False, weight_flag=False,
                    weight_dict=None,
                    filter_flag=False, pre_filt=None,
                    taper_percentage=0.05, taper_type="hann",
                    add_missing_comp_flag=False,
                    rotate_flag=False, inventory=None, event=None):
    """
    Process adjoint sources function, to fit user's needs. Provide:
    1) zero padding the adjoint sources, and then interpolation
    2) add multiple instrument together
    3) rotate from (R, T) to (N, E)

    :param adjsrcs: adjoint sources list from the same station
    :type adjsrcs: list
    :param adj_starttime: starttime of adjoint sources
    :param adj_starttime: obspy.UTCDateTime
    :param raw_synthetic: raw synthetic from SPECFEM output, as reference
    :type raw_synthetic: obspy.Stream or obspy.Trace
    :param inventory: station inventory
    :type inventory: obspy.Inventory
    :param event: event information
    :type event: obspy.Event
    :param sum_over_comp_flag: sum over component flag
    :param weight_dict: weight dictionary
    """

    if not isinstance(adjsrcs, list):
        raise ValueError("Input adjsrcs should be type of list of adjoint "
                         "sources")

    # transfer AdjointSource type to stream for easy processing
    adj_stream, adj_meta = convert_adjs_to_stream(adjsrcs)

    # time reverse the array
    time_reverse_array(adj_stream)

    if interp_flag:
        interp_adj_stream(adj_stream, interp_starttime, interp_delta,
                          interp_npts)

    # sum multiple instruments
    if sum_over_comp_flag:
        adj_stream, adj_meta = \
            sum_adj_on_component(adj_stream, adj_meta,
                                 weight_flag=weight_flag,
                                 weight_dict=weight_dict)

    if filter_flag:
        # after filtering, taper should be applied to ensure the adjoint
        # source would be zero at two ends
        adj_stream.taper(max_percentage=taper_percentage, type=taper_type)
        # filter each trace
        for tr in adj_stream:
            filter_trace(tr, pre_filt)
        # after filtering, taper should be applied to ensure the adjoint
        # source would be zero at two ends
        adj_stream.taper(max_percentage=taper_percentage, type=taper_type)

    if rotate_flag or add_missing_comp_flag:
        add_missing_components(adj_stream, component_list=["Z", "R", "T"])

    if rotate_flag:
        origin = event.preferred_origin() or event.origins[0]
        rotate_adj_stream(adj_stream, origin.latitude, origin.longitude,
                          inventory)

    # convert the stream back to pyadjoint.AdjointSource
    final_adjsrcs = convert_stream_to_adjs(adj_stream, adj_meta)

    return final_adjsrcs
