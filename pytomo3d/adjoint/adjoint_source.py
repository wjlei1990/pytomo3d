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
import os
from obspy import Stream, Trace
import pyadjoint
from .plot_util import plot_adjoint_source
from .process_adjsrc import process_adjoint
from .io import _extract_window_time, _extract_window_id
from .utils import calculate_chan_weight


def calculate_adjsrc_on_trace(obs, syn, windows, config, adj_src_type,
                              figure_mode=False, figure_dir=None,
                              adjoint_src_flag=True):
    """
    Calculate adjoint source on a pair of trace and windows selected

    :param obs: observed trace
    :type obs: obspy.Trace
    :param syn: synthetic trace
    :type syn: obspy.Trace
    :param window_time: window time information, 2-dimension array, like
        [[win_1_left, win_1_right], [win_2_left, win_2_right], ...]
    :type windows: 2-d list or numpy.array
    :param config: config of pyadjoint
    :type config: pyadjoint.Config
    :param adj_src_type: adjoint source type, options include:
        1) "cc_traveltime_misfit"
        2) "multitaper_misfit"
        3) "waveform_misfit"
    :type adj_src_type: str
    :param adjoint_src_flag: whether calcualte adjoint source or not.
        If False, only make measurements
    :type adjoint_src_flag: bool
    :param plot_flag: whether make plots or not. If True, it will lot
        a adjoint source figure right after calculation
    :type plot_flag:  bool
    :return: adjoint source(pyadjoit.AdjointSource)
    """
    if not isinstance(obs, Trace):
        raise ValueError("Input obs should be obspy.Trace")
    if not isinstance(syn, Trace):
        raise ValueError("Input syn should be obspy.Trace")
    # if not isinstance(config, pyadjoint.Config):
    #    raise ValueError("Input config should be pyadjoint.Config")

    window_time = _extract_window_time(windows)
    if len(window_time.shape) != 2 or window_time.shape[1] != 2:
        raise ValueError("Input windows dimension incorrect, dimension"
                         "(*, 2) expected")

    adjsrc = pyadjoint.calculate_adjoint_source(
        adj_src_type=adj_src_type, observed=obs, synthetic=syn,
        config=config, window=window_time, adjoint_src=adjoint_src_flag,
        plot=figure_mode)

    if figure_mode:
        if figure_dir is None:
            figname = None
        else:
            figname = os.path.join(figure_dir, "%s.pdf" % obs.id)
        plot_adjoint_source(adjsrc, win_times=window_time, obs_tr=obs,
                            syn_tr=syn, figname=figname)

    return adjsrc


def calculate_adjsrc_on_stream(observed, synthetic, windows, config,
                               adj_src_type, figure_mode=False,
                               figure_dir=None, adjoint_src_flag=True):
    """
    calculate adjoint source on a pair of stream and windows selected

    :param observed: observed stream
    :type observed: obspy.Stream
    :param synthetic: observed stream
    :type synthetic: obspy.Stream
    :param windows: list of pyflex windows, like:
        [[Windows(), Windows(), Windows()], [Windows(), Windows()], ...]
        For each element, it contains windows for one channel
    :type windows: list
    :param config: config for calculating adjoint source
    :type config: pyadjoit.Config
    :param adj_src_type: adjoint source type
    :type adj_src_type: str
    :param figure_mode: plot flag. Leave it to True if you want to see adjoint
        plots for every trace
    :type figure_mode: bool
    :param adjoint_src_flag: adjoint source flag. Set it to True if you want
        to calculate adjoint sources
    :type adjoint_src_flag: bool
    :return:
    """
    if not isinstance(observed, Stream):
        raise ValueError("Input observed should be obspy.Stream")
    if not isinstance(synthetic, Stream):
        raise ValueError("Input synthetic should be obspy.Stream")
    if windows is None or len(windows) == 0:
        return None
    # if not isinstance(config, pyadjoint.Config):
    #    raise ValueError("Input config should be pyadjoint.Config")

    adjsrcs_list = []

    for chan_win in windows.values():
        if len(chan_win) == 0:
            continue

        obsd_id, synt_id = _extract_window_id(chan_win)

        try:
            obs = observed.select(id=obsd_id)[0]
        except Exception as exp:
            raise ValueError("Missing observed trace {} for window"
                             "due to {}".format(obsd_id, exp))

        if synt_id == "UNKNOWN":
            syn = synthetic.select(channel="*%s" % obs.stats.channel[-1])[0]
        else:
            syn = synthetic.select(id=synt_id)[0]

        adjsrc = calculate_adjsrc_on_trace(
            obs, syn, windows[obsd_id], config, adj_src_type,
            adjoint_src_flag=adjoint_src_flag,
            figure_mode=figure_mode, figure_dir=figure_dir)

        if adjsrc is None:
            continue
        adjsrcs_list.append(adjsrc)

    return adjsrcs_list


def calculate_and_process_adjsrc_on_stream(
        observed, synthetic, windows, inventory, config, event,
        adj_src_type, postproc_param, figure_mode=False,
        figure_dir=None):
    """
    (API for pypaw)
    Calculate based on config, then process adjoint sources
    based on postproc_param
    """
    # check total number of windows. If total number of
    # window is 0, return None
    nwin_total = 0
    for value in windows.values():
        nwin_total += len(value)
    if nwin_total == 0:
        return

    adjsrcs = calculate_adjsrc_on_stream(
        observed, synthetic, windows, config, adj_src_type,
        figure_mode=figure_mode, figure_dir=figure_dir,
        adjoint_src_flag=True)

    if postproc_param["weight_flag"]:
        chan_weight_dict = calculate_chan_weight(adjsrcs, windows)
    else:
        chan_weight_dict = None

    origin = event.preferred_origin() or event.origins[0]
    focal = event.preferred_focal_mechanism()
    # quakeml stores the event duration. To get the half duration,
    # divide it by 2
    hdr = focal.moment_tensor.source_time_function.duration / 2.0
    # according to SPECFEM starttime convention, the trace
    # start time is -1.5 * hdr + CMT origin start time
    # Please note if the convention in SPECFEM_GLOBE changes,
    # the value(1.5) also needs to be changed.
    time_offset = -1.5 * hdr
    starttime = origin.time + time_offset

    new_adjsrcs = process_adjoint(
        adjsrcs, interp_starttime=starttime,
        inventory=inventory, event=event,
        weight_dict=chan_weight_dict,
        **postproc_param)

    # return new_adjsrcs, time_offset
    return new_adjsrcs


def measure_adjoint_on_stream(
        observed, synthetic, windows, config, adj_src_type,
        figure_mode=False, figure_dir=None):
    """
    (API for pypaw)
    Calculate the measurement of adjoint sources. Only measurments
    are returned(adjoint source is not returned).
    """

    nwin_total = 0
    for value in windows.values():
        nwin_total += len(value)
    if nwin_total == 0:
        return

    adjsrcs = calculate_adjsrc_on_stream(
        observed, synthetic, windows, config, adj_src_type,
        figure_mode=False, figure_dir=None,
        adjoint_src_flag=True)

    results = {}
    for adj in adjsrcs:
        results[adj.id] = adj.measurement
    return results
