#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Methods that handles plotting adjoint sources

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
import matplotlib.pyplot as plt
from obspy import Trace
from pyadjoint import AdjointSource
from matplotlib.patches import Rectangle


def plot_only_adjoint(adjsrc, wintimes=None):
    pass


def plot_adjoint_and_data(adjsrc, win_times, obs_tr, syn_tr):

    plt.figure(figsize=(15, 5))
    times = [obs_tr.stats.delta * i for i in range(obs_tr.stats.npts)]

    plt.subplot(211)
    plt.plot(obs_tr.times(), obs_tr.data, color="0.2", label="Observed",
             lw=2)
    plt.plot(syn_tr.times(), syn_tr.data, color="#bb474f",
             label="Synthetic", lw=2)

    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)
    ylim = max(map(abs, plt.ylim()))
    plt.ylim(-ylim, ylim)
    for win in win_times:
        left = win[0]
        right = win[1]
        re = Rectangle((left, plt.ylim()[0]), right - left,
                       plt.ylim()[1] - plt.ylim()[0], color="blue",
                       alpha=0.4)
        plt.gca().add_patch(re)

    plt.subplot(212)
    plt.plot(times, adjsrc.adjoint_source[::-1], color="#2f8d5b", lw=2,
             label="Adjoint Source")
    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)
    xlim = max(map(abs, plt.xlim()))
    ylim = max(map(abs, plt.ylim()))
    plt.ylim(-ylim, ylim)
    for win in win_times:
        left = win[0]
        right = win[1]
        re = Rectangle((left, plt.ylim()[0]), right - left,
                       plt.ylim()[1] - plt.ylim()[0], color="blue",
                       alpha=0.4)
        plt.gca().add_patch(re)

    plt.text(0.01*xlim, 0.9*ylim, adjsrc.adj_src_name,
             horizontalalignment='left', verticalalignment='top')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


def plot_adjoint_source(adjsrc, win_times=None,
                        obs_tr=None, syn_tr=None,
                        figname=None):
    """
    Plot adjoint source for multiple windows

    :param figname: output figure file name
    :type figname: str
    :param adjsrc: adjoint source
    :type adjsrc: pyadjoint.AdjointSource
    :param adjsrc
    :return:
    """
    if not isinstance(adjsrc, AdjointSource):
        raise ValueError("Input adjsrc should be type of "
                         "pyadjoint.AdjointSource")

    if obs_tr is None or syn_tr is None:
        plot_only_adjoint(adjsrc, win_times)
    else:
        if not isinstance(obs_tr, Trace):
            raise ValueError("Input obs_tr should be type of obspy.Trace")
        if not isinstance(syn_tr, Trace):
            raise ValueError("Input syn_tr should be type of obspy.Trace")
        if win_times is None:
            raise ValueError("Input win_tims should be specified as time "
                             "of windows")
        plot_adjoint_and_data(adjsrc, win_times, obs_tr, syn_tr)

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)
