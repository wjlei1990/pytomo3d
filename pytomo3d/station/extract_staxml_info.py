#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script extract channel instrument type from obspy.inventory

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""

from __future__ import print_function, division, absolute_import
import os
from collections import defaultdict
from obspy import Inventory
from obspy import read_inventory


def safe_load_staxml(staxmlfile):
    try:
        inv = read_inventory(staxmlfile, format="STATIONXML")
    except Exception as exp:
        raise ValueError("Invalid staxml file({}) due to: "
                         "{}".format(staxmlfile, exp))
    return inv


def extract_staxml_info(staxml):
    """ extract information from staionxml file or obspy.Inventory """
    instruments = defaultdict(dict)

    if isinstance(staxml, Inventory):
        inv = staxml
    else:
        if os.path.isfile(staxml):
            inv = safe_load_staxml(staxml)
        else:
            raise ValueError("Input staxml is neither obspy.Inventory or "
                             "staxml file")
    for nw in inv:
        nw_code = nw.code
        for sta in nw:
            sta_code = sta.code
            for chan in sta:
                chan_code = chan.code
                loc_code = chan.location_code
                key = "%s.%s.%s.%s" % (nw_code, sta_code, loc_code, chan_code)
                instruments[key]["latitude"] = chan.latitude
                instruments[key]["longitude"] = chan.longitude
                instruments[key]["elevation"] = chan.elevation
                instruments[key]["depth"] = chan.depth
                if chan.sensor.description is not None:
                    sensor_type = chan.sensor.description
                elif chan.sensor.type is not None:
                    sensor_type = chan.sensor.type
                else:
                    sensor_type = "None"
                instruments[key]["sensor"] = sensor_type

    return instruments
