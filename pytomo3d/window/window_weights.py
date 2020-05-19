#!/usr/bin/env pythoiin
# -*- coding: utf-8 -*-
"""
Calculate the adjoint source weighting based on the station and source
distribution.

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import print_function, division, absolute_import

import os
from collections import defaultdict
import numpy as np
from pprint import pprint

from spaceweight import SpherePoint
from spaceweight import SphereDistRel
from pytomo3d.utils.io import check_dict_keys, load_json


def _receiver_validator(weights, rec_wcounts, cat_wcounts):
    """
    Validate the receiver weights, and make sure it sums to
    category window counts.

    :param weights:
    :param rec_wcounts:
    :param cat_wcounts:
    :return:
    """
    wsum = 0
    for chan, chan_weight in weights.items():
        nwin = rec_wcounts[chan]
        wsum += chan_weight * nwin

    print("Summation of (rec_weights * rec_nwins): %.2f" % wsum)
    if not np.isclose(wsum, cat_wcounts):
        raise ValueError("receiver validator fails: %f, %f" %
                         (wsum, cat_wcounts))


def calculate_receiver_window_counts(windows):
    """
    Given windows, count them based on receiver level and category
    (component) level.

    :param windows:
    :return:
    """
    rec_wcounts = defaultdict(dict)
    cat_wcounts = defaultdict(lambda: 0)
    for sta, sta_window in windows.items():
        for chan, chan_win in sta_window.items():
            comp = chan.split(".")[-1]
            nwin = len(chan_win)
            if nwin == 0:
                continue
            rec_wcounts[comp][chan] = nwin
            cat_wcounts[comp] += nwin

    return rec_wcounts, cat_wcounts


def assign_receiver_to_points(channels, stations):
    """
    Assign the receiver information to SpherePoint

    :param rec_wcounts:
    :param stations:
    :return:
    """
    points = []
    for chan in channels:
        component = chan.split(".")[-1][-1]
        if component == "Z":
            # it is important to set the default weight value to
            # 1.0 here. Becuase if there is no receiver weightings,
            # then this default weights will be used
            point = SpherePoint(stations[chan]["latitude"],
                                stations[chan]["longitude"],
                                tag=chan, weight=1.0)
        else:
            # for R and T component. In station file, there
            # are only `EN` or `12`
            echan = chan[:-1] + "E"
            chan1 = chan[:-1] + "1"
            zchan = chan[:-1] + "Z"

            if echan in stations:
                point = SpherePoint(stations[echan]["latitude"],
                                    stations[echan]["longitude"],
                                    tag=chan, weight=1.0)
            elif chan1 in stations:
                point = SpherePoint(stations[chan1]["latitude"],
                                    stations[chan1]["longitude"],
                                    tag=chan, weight=1.0)
            elif zchan in stations:
                point = SpherePoint(stations[zchan]["latitude"],
                                    stations[zchan]["longitude"],
                                    tag=chan, weight=1.0)
            else:
                raise ValueError("Can't find station information(%s)"
                                 % (chan))
        points.append(point)
    return points


def get_receiver_weights(component, center, points, max_ratio, plot=False,
                         figname_prefix=None):
    """
    Calculate the receiver weights given receiver(points) distribution.

    :param component:
    :param center:
    :param points:
    :param max_ratio:
    :param plot:
    :param figname_prefix:
    :return:
    """
    # calculate weight; otherwise, leave it as default value(1)
    weightobj = SphereDistRel(points, center=center)

    if plot:
        scan_figname = figname_prefix + ".%s.smart_scan.png" % component
    else:
        scan_figname = None

    ref_distance, cond_number = weightobj.smart_scan(
        max_ratio=max_ratio, start=0.5, gap=0.5,
        drop_ratio=0.95, plot=plot,
        figname=scan_figname)

    if plot:
        map_figname = figname_prefix + ".%s.weight.pdf" % component
        weightobj.plot_global_map(figname=map_figname, lon0=180.0)

    return ref_distance, cond_number


def normalize_receiver_weights(points, wcounts):
    """
    Normalize the receiver weights

    :param points:
    :param rec_wcounts:
    :return:
    """
    wsum = 0
    nwins_total = 0
    for point in points:
        nwin = wcounts[point.tag]
        nwins_total += nwin
        wsum += point.weight * nwin
    norm_factor = nwins_total / wsum

    weights = {}
    for point in points:
        weights[point.tag] = point.weight * norm_factor

    return weights


def determine_receiver_weighting(
        src, stations, windows, search_ratio=0.35, weight_flag=True,
        plot_flag=False, figname_prefix=None):
    """
    Given one station and window information, determine the receiver
    weighting
    In one asdf file, there are still 3 components, for example,
    ["BHR", "BHT", "BHZ"]. These three components should be treated
    indepandently and weights will be calculated independantly.

    :return: dict of weights which contains 3 components. Each components
        contains weights values
    """
    center = SpherePoint(src["latitude"], src["longitude"],
                         tag="source")

    rec_wcounts, cat_wcounts = calculate_receiver_window_counts(windows)

    weights = {}
    # in each components, calculate weight
    ref_dists = {}
    cond_nums = {}
    for comp, comp_info in rec_wcounts.items():
        print("-" * 10 + "\nComponent: %s" % comp)
        points = assign_receiver_to_points(comp_info, stations)
        print("Number of receivers: %d" % len(points))
        print("Number of windows: %d" % cat_wcounts[comp])

        if weight_flag:
            ref_dists[comp], cond_nums[comp] = \
                get_receiver_weights(comp, center, points, search_ratio,
                                     plot=plot_flag,
                                     figname_prefix=figname_prefix)
        else:
            # if not weight the receiver, then just leave the weight
            # values all to the default values(1.0). Leave the ref_dists
            # and cond_nums to None since there is no weighting
            # procedures. However, normalization still needs to be
            # applied later.
            ref_dists[comp] = None
            cond_nums[comp] = None

        # normlalize the receiver weights according to the reciever
        # window counts(normalization requirements)
        weights[comp] = normalize_receiver_weights(points, rec_wcounts[comp])

        _receiver_validator(weights[comp], rec_wcounts[comp],
                            cat_wcounts[comp])

    return {"rec_weights": weights, "rec_wcounts": rec_wcounts,
            "cat_wcounts": cat_wcounts, "rec_ref_dists": ref_dists,
            "rec_cond_nums": cond_nums}


def calculate_receiver_weights_interface(
        src_info, path_info, weighting_param, _verbose=True):
    """
    The user interface(API) for calculation the receiver weighting
    in pypaw

    :param src_info: keys contains ["latitude", "longitude"]
    :type src_info: dict
    :param path_info: keys contains ["station_file", "window_file",
        "output_file"]
    :type path_info: dict
    :param weighting_param: keys contains ["flag", "plot", "search_ratio"]
    :type weighting_param: dict
    """
    check_dict_keys(src_info, ["latitude", "longitude", "depth_in_m"])
    check_dict_keys(path_info, ["station_file", "window_file", "output_file"])
    check_dict_keys(weighting_param, ["flag", "plot", "search_ratio"])

    search_ratio = weighting_param["search_ratio"]
    plot_flag = weighting_param["plot"]
    weight_flag = weighting_param["flag"]
    # each file still contains 3-component
    if _verbose:
        print("src_info: %s" % src_info)
        print("path_info:")
        pprint(path_info)
        print("weighting param:")
        pprint(weighting_param)

    station_info = load_json(path_info["station_file"])
    window_info = load_json(path_info["window_file"])

    outputdir = os.path.dirname(path_info["output_file"])
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    figname_prefix = os.path.join(outputdir, "weights")

    _results = determine_receiver_weighting(
        src_info, station_info, window_info,
        search_ratio=search_ratio,
        weight_flag=weight_flag,
        plot_flag=plot_flag, figname_prefix=figname_prefix)

    return _results


def _category_validator(weights, wcounts):
    """
    Validate the category weights

    :param weights:
    :param counts:
    :return:
    """
    wsum = 0.0
    nwins_total = 0
    for p, pinfo in weights.items():
        for c in pinfo:
            wsum += weights[p][c] * wcounts[p][c]
            nwins_total += wcounts[p][c]

    print("Summation of (cat_weight * cat_nwins): %.2f" % wsum)
    if not np.isclose(wsum, nwins_total):
        raise ValueError("Category validator fails: %f, %f" %
                         (wsum, nwins_total))


def normalize_category_weights(category_ratio, cat_wcounts):
    """
    """
    print("category ratio:")
    pprint(category_ratio)
    print("category window counts:")
    pprint(cat_wcounts)
    sumv = 0
    nwins_total = 0
    for p, pinfo in cat_wcounts.items():
        for c in pinfo:
            sumv += cat_wcounts[p][c] * category_ratio[p][c]
            nwins_total += cat_wcounts[p][c]

    if nwins_total == 0 or sumv == 0.0:
        normc = 1.0
    else:
        normc = nwins_total / sumv

    weights = {}
    for p, pinfo in cat_wcounts.items():
        weights[p] = {}
        for c in pinfo:
            weights[p][c] = normc * category_ratio[p][c]

    return weights


def check_category_ratio_consistency(cat_weight_ratio, cat_wcounts):
    """
    check the category weighting ratio(provide by user) has
    the same period band and component as in read seismic
    data.
    """
    err = 0
    # check consistency
    for p, pinfo in cat_weight_ratio.items():
        for c in pinfo:
            try:
                cat_wcounts[p][c]
            except KeyError:
                err = 1
                print("Missing %s.%s" % (p, c))
    if err:
        raise ValueError("category weighting ratio information is not "
                         "consistent with window information")


def calculate_category_weights_interface(category_param, cat_wcounts):
    """
    User interface(API) for calculating category weights

    Calculate the category weighting based on window counts
    in each category. The weighting ratios for different categoies
    are input parameters. So this function only normlizes
    the weights without change the ratio.
    !!! WARNING !!!
    It is the ratio between different categories, not the window
    counts. For example, the value will be mostly likely to be
    set to 1/N_w(for windows, less weights to balance categories)

    :param weight_param: user parameter, which contains the weighting
        ratio for each category.
    :type weight_param: dict
    :param cat_wcounts: category window counts, which should contains
        the same period band as category_param['ratio']
    :type cat_wcounts: dict
    """
    check_dict_keys(category_param, ["flag", "ratio"])
    check_category_ratio_consistency(category_param["ratio"], cat_wcounts)

    weights = normalize_category_weights(
        category_param["ratio"], cat_wcounts)
    _category_validator(weights, cat_wcounts)

    print("Final category weights:")
    pprint(weights)
    return weights


def combine_receiver_and_category_weights(rec_weights, cat_weights):
    """
    Combine weights for receiver weighting and category weighting
    """
    # combine weights
    weights = {}
    for period, period_info in rec_weights.items():
        weights[period] = {}
        for comp, comp_info in period_info.items():
            for chan_id in comp_info:
                rec_weight = comp_info[chan_id]
                cat_weight = cat_weights[period][comp]
                _weight = {"receiver": rec_weight,
                           "category": cat_weight}
                _weight["weight"] = \
                    rec_weight * cat_weight
                weights[period][chan_id] = _weight
    return weights
