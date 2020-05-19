#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Methods that handles rotation of seismograms as extension to Obspy.
It can rotate `12`, `EN` and `RT`, forward and backward.

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import print_function
import numpy as np
from obspy.geodetics import gps2dist_azimuth
from obspy import Stream
from .rotate_utils import check_orthogonality, rotate_12_rt


def calculate_baz(elat, elon, slat, slon):
    _, _, baz = gps2dist_azimuth(elat, elon, slat, slon)
    return baz


def ensemble_synthetic_channel_orientation(chan):
    """
    For synthetic data from SPECFEM, just ensemble the orientation
    information.
    """
    orientation = {"Z": (90, 0),
                   "N": (0, 0),
                   "E": (0, 90)}
    try:
        dip, azi = orientation[chan[-1]]
    except KeyError:
        raise Exception("Orientation is not defined for synthetics"
                        " of component: %s" % chan)
    return dip, azi


def extract_channel_orientation(tr, inv):
    """ Extract the dip and azimuth from inventory, given the trace """
    try:
        nw = tr.stats.network
        sta = tr.stats.station
        loc = tr.stats.location
        chan = tr.stats.channel

        if loc == "S3":
            # If channel is synthetic(from specfem), return default value
            # for dip and azimuth.
            dip, azi = ensemble_synthetic_channel_orientation(chan)
        else:
            chan_inv = inv.select(network=nw, station=sta, location=loc,
                                  channel=chan)[0][0][0]
            dip, azi = chan_inv.dip, chan_inv.azimuth
    except Exception as errmsg:
        print("Unable to extract channel orientation information [%s]"
              " due to: %s" % (tr.id, errmsg))
        dip, azi = None, None

    return dip, azi


def extract_station_location(st, inventory):
    """
    Extract the station latitude and longitude from inventory, given
    stream.
    """
    nw = st[0].stats.network
    station = st[0].stats.station
    _inv = inventory.select(network=nw, station=station)
    sta_lat = float(_inv[0][0].latitude)
    sta_lon = float(_inv[0][0].longitude)
    return sta_lat, sta_lon


def check_vertical_inventory_sanity(tr, inventory):
    """
    Check the inventory of vertical(Z) component, check
    if the abs(dip) is 90 and azimuth is 0.
    """
    if tr.stats.channel[-1] != "Z":
        raise ValueError("Function only checks vertical(Z) component(%s)"
                         % tr.stats.channel)
    dip, azi = extract_channel_orientation(tr, inventory)

    if dip is None or azi is None:
        return False

    if np.isclose(abs(dip), 90.0) and np.isclose(abs(azi), 0.0):
        return True
    else:
        return False


def check_horizontal_inventory_sanity(tr1, tr2, inventory):
    """
    Check two horizontal components and see if their dip is 0
    and azimuth is orthogonal to each other.

    :param tr1:
    :param tr2:
    :param inventory:
    :return:
    """
    if tr1.id[:-1] != tr2.id[:-1]:
        raise ValueError("Two horizontal ids should share the same network,"
                         "station, location and channel[0:2]: %s, %s"
                         % (tr1.id, tr2.id))

    if tr1.stats.channel[-1] == "Z" or tr2.stats.channel[-1] == "Z":
        raise ValueError("Functions should check two horizontal component:"
                         "%s, %s" % (tr1.id, tr2.id))
    dip1, azi1 = extract_channel_orientation(tr1, inventory)
    dip2, azi2 = extract_channel_orientation(tr2, inventory)

    if dip1 is None or azi1 is None or dip2 is None or azi2 is None:
        return False

    # check dip
    if not np.isclose(dip1, 0.0) or not np.isclose(dip2, 0.0):
        return False

    # check azimuth
    if not check_orthogonality(azi1, azi2):
        return False

    return True


def check_information_before_rotation(i_1, i_2, inv, sanity_check=False):
    # check starttime, sampling rate
    dt = 0.5 * i_1.stats.delta
    if (len(i_1) != len(i_2)) or \
            (abs(i_1.stats.starttime - i_2.stats.starttime) > dt) \
            or (i_1.stats.sampling_rate != i_2.stats.sampling_rate):
        msg = "All components need to have the same time span."
        raise ValueError(msg)

    # check inventory sanity if required by user
    if sanity_check:
        if not check_horizontal_inventory_sanity(i_1, i_2, inv):
            msg = "Horizontal component are not orthogonal to " \
                  "each other: %s, %s" % (i_1.id, i_2.id)
            raise ValueError(msg)


def rotate_12_rt_func(st, inv, back_azimuth, method="12->RT",
                      sanity_check=False):
    """
    Rotate horizontal component to RT. This function works generally
    for two horizontal and orthogonal components. This function
    supports two method:
        1) "12->RT": which rotates "12" component to "RT", for example,
            "BH1" and "BH2" to "BHR" and "BHT"
        2) "NE->RT": which rotates "NE" component to "RT", for example,
            "BHN" and "BHE" to "BHR" and "BHT"
    The reason why we use our own rotation function is because in obspy
    the inventory information is not checked.

    :param st: input stream
    :param inv: station inventory information
    :param method: rotation method
    :param back_azimuth: back azimuth(station to event azimuth)
    :return: rotated stream
    """
    if method not in ["12->RT", "NE->RT"]:
        raise ValueError("rotate_12_rt_func only supports method:"
                         "['12->RT', 'NE->RT']")

    bad_ids = []
    input_components, output_components = method.split("->")
    if len(input_components) == 2:
        input_1 = st.select(component=input_components[0])
        input_2 = st.select(component=input_components[1])
        for i_1, i_2 in zip(input_1, input_2):
            try:
                check_information_before_rotation(
                    i_1, i_2, inv, sanity_check=sanity_check)
            except Exception as err:
                bad_ids.extend([i_1.id, i_2.id])
                print("Unable to rotate [%s, %s] due to: %s"
                      % (i_1.id, i_2.id, err))
                continue
            inc1, azi1 = extract_channel_orientation(i_1, inv)
            inc2, azi2 = extract_channel_orientation(i_2, inv)
            if azi1 is None or azi2 is None:
                continue
            output_1, output_2 = rotate_12_rt(i_1.data, i_2.data, back_azimuth,
                                              azi1, azi2)
            if output_1 is None or output_2 is None:
                continue
            i_1.data = output_1
            i_2.data = output_2
            # Rename the components
            i_1.stats.channel = i_1.stats.channel[:-1] + output_components[0]
            i_2.stats.channel = i_2.stats.channel[:-1] + output_components[1]
            # Add the azimuth back to stats object
            for comp in (i_1, i_2):
                comp.stats.back_azimuth = back_azimuth
    else:
        raise ValueError("Wrong method: %s" % method)

    # remove trace in bad_ids
    # for bid in bad_ids:
    #    st.remove(st.select(id=bid)[0])

    return st


def rotate_rt_to_ne(st, baz):
    """ Use obspy rotate function to rotate from RT to NE """
    st.rotate(method="RT->NE", back_azimuth=baz)


def remove_bad_z_component(st, inventory):
    """ remove Z component if its inventory does not meet requirements """
    for tr in st:
        if tr.stats.channel[-1] == "Z":
            if not check_vertical_inventory_sanity(tr, inventory):
                st.remove(tr)


def rotate_to_rt(st, baz, inventory, mode, sanity_check=False):
    """
    Rotate horizontal components(12 or NE) to RT directions.

    :param st:
    :param baz:
    :param inventory:
    :param mode:
    :param sanity_check:
    :return:
    """
    components = [tr.stats.channel[-1] for tr in st]

    if sanity_check:
        # remove bad Z component
        remove_bad_z_component(st, inventory)

    if mode in ["12->RT", "ALL->RT"]:
        if "1" in components and "2" in components:
            try:
                rotate_12_rt_func(st, inventory, method="12->RT",
                                  back_azimuth=baz)
            except Exception as errmsg:
                print("Error rotating 12->RT:%s" % errmsg)

    if mode in ["NE->RT", "ALL->RT"]:
        if "N" in components and "E" in components:
            try:
                rotate_12_rt_func(st, inventory, baz, method="NE->RT",
                                  sanity_check=sanity_check)
            except Exception as e:
                print("Error rotating NE->RT:%s" % e)


def rotate_one_station_stream(st, event_latitude, event_longitude,
                              station_latitude=None, station_longitude=None,
                              inventory=None, mode="NE->RT",
                              sanity_check=False):
    """
    Rotate the stream from the same network, station, location and channel
    code, for example the stream should only contains traces whose ids are
    "II.AAK.00.BH*"
    """

    mode = mode.upper()
    mode_options = ["NE->RT", "ALL->RT", "12->RT", "RT->NE"]
    if mode not in mode_options:
        raise ValueError("rotate_stream mode(%s) should be within %s"
                         % (mode, mode_options))
    if mode[0:2] == "RT" and sanity_check:
        raise ValueError("if rotating from RT to NE, then set "
                         "sanity_check to False")

    if inventory is None and \
            (station_latitude is None and station_longitude is None):
        raise ValueError("You need either specify station inventory "
                         "or station location information")

    if mode in ["12->RT", "ALL->RT"] and inventory is None:
        raise ValueError("If you want to rotate 12, you need to specify "
                         "station inventory")

    if len(sort_stream_by_station(st)) != 1:
        raise ValueError("This method only accepts stream that contains "
                         "one station")

    if station_latitude is None or station_longitude is None:
        try:
            station_latitude, station_longitude = \
                extract_station_location(st, inventory)
        except Exception as errmsg:
            # station_latitude and station_longitude is unknown
            # so the stream should be skipped
            print("Error extracting staiton latitude and longitude from "
                  "staiton inventory: %s" % errmsg)
            return

    baz = calculate_baz(event_latitude, event_longitude,
                        station_latitude, station_longitude)

    if mode in ["NE->RT", "12->RT", "ALL->RT"]:
        rotate_to_rt(st, baz, inventory, mode, sanity_check=sanity_check)

    if mode in ["RT->NE"]:
        rotate_rt_to_ne(st, baz)

    return st


def sort_stream_by_station(st):
    """
    Sort the traces in stream. Group the traces from same network,
    station, location, and channel code together
    """
    ntotal = len(st)
    sta_dict = {}
    n_added = 0
    for tr in st:
        nw = tr.stats.network
        station = tr.stats.station
        loc = tr.stats.location
        chan = tr.stats.channel[0:2]
        station_id = "%s.%s.%s.%s" % (nw, station, loc, chan)
        if station_id not in sta_dict:
            sta_dict[station_id] = \
                st.select(network=nw, station=station,
                          location=loc, channel="%s*" % chan)
            n_added += len(sta_dict[station_id])

    if n_added != ntotal:
        raise ValueError("Sort stream by station errors: number of traces "
                         "is inconsistent")
    return sta_dict


def rotate_stream(st, event_latitude, event_longitude,
                  inventory, mode="ALL->RT", sanity_check=False):
    """
    Rotate a stream to radial and transverse components based on the
    station information and event information

    :param st: input stream
    :type st: obspy.Stream
    :param event_latitude: event latitude
    :type event_latitude: float
    :param event_longitude: event longitude
    :type event_longitude: float
    :param inventory: station inventory information. If you want to rotate
    "12" components, you need to provide inventory since only station
    and station_longitude is not enough.
    :type inventory: obspy.Inventory
    :param mode: rotation mode, could be one of:
        1) "NE->RT": rotate only North and East channel to RT
        2) "12->RT": rotate only 1 and 2 channel, like "BH1" and "BH2" to RT
        3) "ALL->RT": rotate all components to RT
        4) "RT->NE": rotate RT to NE
    :param sanity_check: check the sanity of inventory, mianly of the
        orientation of ZNE components.
        1) If rotating observed data from NE(12)to RT, it is recommended
            to set to True; if rotating from RT to NE, then you could set
            it to False;
            ATTENTION: please turn it to True when processing observed
            data since the instrument alignment of observed data might
            be messy.
        2) If rotating synthetic data, you could set it to False since
            I assume for synthetic data, there should not be problem
            associated with orientation.
    :return: rotated stream(obspy.Stream)
    """

    rotated_stream = Stream()

    mode = mode.upper()
    mode_options = ["NE->RT", "ALL->RT", "12->RT", "RT->NE"]
    if mode not in mode_options:
        raise ValueError("rotate_stream mode(%s) should be within %s"
                         % (mode, mode_options))
    if mode[0:2] == "RT" and sanity_check:
        raise ValueError("if rotating from RT to NE, then set "
                         "sanity_check to False")

    if mode in ["12->RT", "ALL->RT"] and inventory is None:
        raise ValueError("Mode %s required inventory(stationxml) "
                         "information provided(to rotate '12')" % mode)

    # the stream might contains multiple stations so each station should
    # be rotated independently, or stations with different channel,
    # (for example, BH and EH), and different locations(such as
    # "00" and "10").
    sorted_st_dict = sort_stream_by_station(st)

    for sta_stream in sorted_st_dict.values():
        # loop over stations
        nw = sta_stream[0].stats.network
        station = sta_stream[0].stats.station
        loc = sta_stream[0].stats.location
        chan = sta_stream[0].stats.channel

        if loc == "S3" or chan[0:2] == "MX":
            # SPECFEM HACK: if the synthetic is generated by SPECFEM
            # and the stationxml is from read data, then the location
            # won't match between synt and staxml
            station_inv = inventory.select(network=nw, station=station)
        else:
            station_inv = inventory.select(network=nw, station=station,
                                           location=loc)

        _st = \
            rotate_one_station_stream(sta_stream, event_latitude,
                                      event_longitude, inventory=station_inv,
                                      mode=mode, sanity_check=sanity_check)
        if _st is not None:
            rotated_stream += _st

    return rotated_stream
