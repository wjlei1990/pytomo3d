#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils for seismic data download

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from obspy.clients.fdsn import Client
import os


def read_station_file(station_filename):
    stations = []
    with open(station_filename, "rt") as fh:
        for line in fh:
            line = line.split()
            stations.append((line[1], line[0]))
    return stations


def _parse_station_id(station_id):
    content = station_id.split("_")
    if len(content) == 2:
        nw, sta = content
        loc = "*"
        comp = "*"
    elif len(content) == 4:
        nw, sta, loc, comp = content
    else:
        raise ValueError("Can't not parse station_id: %s" % station_id)
    return nw, sta, loc, comp


def download_waveform(stations, starttime, endtime, outputdir=None,
                      client=None):
    """
    download wavefrom data from IRIS data center

    :param stations: list of stations, should be list of station ids,
        for example, "II.AAK.00.BHZ". Parts could be replaced by "*",
        for example, "II.AAK.*.BH*"
    """
    if client is None:
        client = Client("IRIS")

    if starttime > endtime:
        raise ValueError("Starttime(%s) is larger than endtime(%s)"
                         % (starttime, endtime))

    if not os.path.exists(outputdir):
        raise ValueError("Outputdir not exists: %s" % outputdir)

    _status = {}
    for station_id in stations:
        error_code = "None"
        network, station, location, channel = _parse_station_id(station_id)

        if outputdir is not None:
            filename = os.path.join(outputdir, "%s.mseed" % station_id)
            if os.path.exists(filename):
                os.remove(filename)
        else:
            filename = None

        try:
            st = client.get_waveforms(
                network=network, station=station, location=location,
                channel=channel, starttime=starttime, endtime=endtime)
            if len(st) == 0:
                error_code = "stream empty"
            if filename is not None and len(st) > 0:
                st.write(filename, format="MSEED")
        except Exception as e:
            error_code = "Failed to download waveform '%s' due to: %s" \
                % (station_id, str(e))
            print(error_code)

        _status[station_id] = error_code

    return {"stream": st, "status": _status}


def download_stationxml(stations, starttime, endtime, outputdir=None,
                        client=None, level="response"):

    if client is None:
        client = Client("IRIS")

    if starttime > endtime:
        raise ValueError("Starttime(%s) is larger than endtime(%s)"
                         % (starttime, endtime))

    if not os.path.exists(outputdir):
        raise ValueError("Outputdir not exists: %s" % outputdir)

    _status = {}
    for station_id in stations:
        error_code = "None"
        network, station, location, channel = _parse_station_id(station_id)

        if outputdir is not None:
            filename = os.path.join(outputdir, "%s.xml" % station_id)
            if os.path.exists(filename):
                os.remove(filename)
        else:
            filename = None

        try:
            inv = client.get_stations(
                network=network, station=station, location=location,
                channel=channel, starttime=starttime, endtime=endtime,
                level=level)
            if len(inv) == 0:
                error_code = "Inventory Empty"
            if filename is not None and len(inv) > 0:
                inv.write(filename, format="STATIONXML")
        except Exception as e:
            error_code = "Failed to download StationXML '%s' due to: %s" \
                % (station_id, str(e))
            print(error_code)

        _status[station_id] = error_code

    return {"inventory": inv, "status": _status}
