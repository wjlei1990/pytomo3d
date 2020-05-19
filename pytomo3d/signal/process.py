#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Methods that handles signal data processing for stream(Obspy.Stream)

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""

from __future__ import (division, print_function, absolute_import)
from obspy.signal.invsim import cosine_sac_taper
from obspy.signal.util import _npts2nfft
from obspy import Stream, Trace
import numpy as np
from .rotate import rotate_stream


def check_array_order(array, order="ascending"):
    """
    Check whether a array is in ascending order or descending order
    :param array: the input array
    :param order: "ascending" or "descending"
    :return:
    """
    array = np.array(array)
    if order not in ("descending", "ascending"):
        raise ValueError("Order should be either ascending or descending")

    if order == "descending":
        array *= -1

    return (array == sorted(array)).all()


def flex_cut_trace(trace, cut_starttime, cut_endtime, dynamic_npts=0):
    """
    not cut strictly(but also based on the original trace length)

    :param trace: input trace
    :type trace: obspy.Trace
    :param cut_starttime: starttime of cutting
    :type cut_starttime: obspy.UTCDateTime
    :param cut_endtime: endtime of cutting
    :type cut_endtime: obspy.UTCDateTime
    """
    if not isinstance(trace, Trace):
        raise TypeError("flex_cut_trace method only accepts obspy.Trace "
                        "as the first argument")

    delta = trace.stats.delta
    cut_starttime = cut_starttime - dynamic_npts * delta
    cut_endtime = cut_endtime + dynamic_npts * delta
    trace.trim(cut_starttime, cut_endtime)


def flex_cut_stream(st, cut_start, cut_end, dynamic_npts=0):
    """
    Flexible cut stream. But checks for the time.

    :param st: input stream
    :param cut_start: cut starttime
    :param cut_end: cut endtime
    :param dynamic_npts: the dynamic number of points before cut_start
        and after
        cut_end
    :return: the cutted stream
    """
    if not isinstance(st, Stream):
        raise TypeError("flex_cut_stream method only accepts obspy.Stream "
                        "the first Argument")
    new_st = Stream()
    count = 0
    for tr in st:
        flex_cut_trace(tr, cut_start, cut_end, dynamic_npts=dynamic_npts)
        # throw out small piece of data at this step
        if tr.stats.starttime <= cut_start and tr.stats.endtime >= cut_end:
            new_st.append(tr)
            count += 1
    if count == 0:
        raise ValueError("None of traces in Stream satisfy the "
                         "cut time length")
    return new_st


def filter_stream(st, pre_filt):
    """
    Filter a stream

    :param st:
    :param per_filt:
    :return:
    """
    if not isinstance(st, Stream):
        raise TypeError("Input st should be type of Stream")
    for tr in st:
        filter_trace(tr, pre_filt)


def filter_trace(tr, pre_filt):
    """
    Perform a frequency domain taper mimicing the behavior during the
    response removal, without a actual response removal.

    :param tr: input trace
    :param pre_filt: frequency array(Hz) in ascending order, to define
        the four corners of filter, for example, [0.01, 0.1, 0.2, 0.5].
    :type pre_filt: Numpy.array or list
    :return: filtered trace
    """
    if not isinstance(tr, Trace):
        raise TypeError("First Argument should be trace: %s" % type(tr))
    if len(pre_filt) != 4:
        raise ValueError("Length of filter must be 4(corner frequencies)")
    if not check_array_order(pre_filt, order="ascending"):
        raise ValueError("Frequency band should be in ascending order: %s"
                         % pre_filt)

    data = tr.data.astype(np.float64)
    origin_len = len(data)
    if origin_len == 0:
        return

    # smart calculation of nfft dodging large primes
    nfft = _npts2nfft(len(data))

    fy = 1.0 / (tr.stats.delta * 2.0)
    freqs = np.linspace(0, fy, nfft // 2 + 1)

    # Transform data to Frequency domain
    data = np.fft.rfft(data, n=nfft)
    data *= cosine_sac_taper(freqs, flimit=pre_filt)
    data[-1] = abs(data[-1]) + 0.0j
    # transform data back into the time domain
    data = np.fft.irfft(data)[0:origin_len]
    # assign processed data and store processing information
    tr.data = data


def interpolate_stream(stream, sampling_rate, starttime=None, npts=None):
    """
    For a fairly large stream, use stream.interpolate() is not a wise
    choice since if there is one trace fails, then the whole interpolation
    will stop. So it is better to operate interpolation on the trace
    level
    """
    st_new = Stream()
    if not isinstance(stream, Stream):
        raise TypeError("Input stream must be type of obspy.Stream")
    for tr in stream:
        try:
            tr.interpolate(sampling_rate, starttime=starttime, npts=npts)
            st_new.append(tr)
        except ValueError as err:
            print("Error in interpolation on '%s':%s" % (tr.id, err))
    return st_new


def process_stream(st, inventory=None, remove_response_flag=False,
                   water_level=60, filter_flag=False, pre_filt=None,
                   starttime=None, endtime=None,
                   resample_flag=False, sampling_rate=1.0,
                   taper_type="hann", taper_percentage=0.05,
                   rotate_flag=False, event_latitude=None,
                   event_longitude=None, sanity_check=False):
    """
    Stream processing function defined for general purpose of tomography.
    The advantage of using Stream, rather than than Trace, is that rotation
    could be operated if the Stream contains multiple channels. But this
    function also deals with Trace, but you need to set rotate_flag to
    False

    :param st: input stream
    :type st: obspy.Stream
    :param remove_response_flag: flag for remove instrument response. If True,
        then inv should be specified, and filter_flag would not be taken caren
        of. If you want just filter the seismogram, please leave this to False
        and set filter_flag to True.
    :type remove_response_flag: bool
    :param inventory: station inventory information
    :type inventory: obspy.Inventory
    :param water_level: water level used in remove instrument response. The
        default value in obspy is 60.
    :type water_level: float
    :param filter_flag:flag for filter the seismogram
    :type filter_flag: bool
    :param pre_filt: list of tuple of 4 corner frequency for filter,
        in ascending order(unit: Hz)
    :type pre_filt: list, tuple or numpy.array
    :param starttime: starttime of cutting
    :type starttime: obspy.UTCDateTime
    :param endtime: endtime of cutting
    :type endtime: obspy.UTCDateTime
    :param resample_flag: flag for data resampling
    :type resample_flag: bool
    :param sampling_rate: resampling rate(unit: Hz)
    :type sampling_rate: float
    :param taper_type: taper type, options from obspy taper
    :type taper_type: str
    :param taper_percentage: percentage of taper
    :type taper_percentage: float
    :param rotate_flag: rotate flag. If true, both inv and event location
        information should be provided
    :param event_latitude: latitude of event, for rotation usage
    :type event_latitude: float
    :param event_longitude: longitude of event, for rotation usage
    :type event_longitude: float
    :param sanity_check: sanity check the inventory information when
        rotating.
    :type sanity_check: bool
    :return: processed stream
    """
    # check input data type
    if isinstance(st, Trace):
        st = Stream(traces=[st, ])
        _is_trace = True
    elif isinstance(st, Stream):
        _is_trace = False
    else:
        raise TypeError("Input seismogram should be either obspy.Stream "
                        "or obspy.Trace")

    # cut the stream out before processing to reduce computation
    if starttime is not None and endtime is not None:
        st = flex_cut_stream(st, starttime, endtime, dynamic_npts=10)

    if filter_flag or remove_response_flag:
        # detrend ,demean, taper
        st.detrend("linear")
        st.detrend("demean")
        st.taper(max_percentage=taper_percentage, type=taper_type)

    # remove response or filter
    if filter_flag:
        if pre_filt is None or len(pre_filt) != 4:
            raise ValueError("Filter band should be list or tuple with "
                             "length of 4")
        if not check_array_order(pre_filt, order="ascending"):
            raise ValueError("Input pre_filt must be in ascending order: %s"
                             % pre_filt)

    if remove_response_flag:
        # remove response
        if inventory is None:
            raise ValueError("Station information(inv) should be provided if"
                             "you want to remove instrument response")
        st.attach_response(inventory)
        if filter_flag:
            st.remove_response(output="DISP", pre_filt=pre_filt,
                               zero_mean=False, taper=False,
                               water_level=water_level)
        else:
            st.remove_response(output="DISP", zero_mean=False, taper=False)
    elif filter_flag:
        # Perform a frequency domain taper like during the response removal
        # just without an actual response...
        filter_stream(st, pre_filt)

    if filter_flag or remove_response_flag:
        # detrend, demean or taper
        st.detrend("linear")
        st.detrend("demean")
        st.taper(max_percentage=taper_percentage, type=taper_type)

    # resample
    if resample_flag:
        # interpolation
        if sampling_rate is None:
            raise ValueError("sampling rate should be provided if you set"
                             "resample_flag=True")

        if endtime is not None and starttime is not None:
            npts = int((endtime - starttime) * sampling_rate) + 1
            st = interpolate_stream(st, sampling_rate, starttime=starttime,
                                    npts=npts)
        else:
            # it doesn't matter if starttime is None or not, cause
            # obspy will handle this case
            st = interpolate_stream(st, sampling_rate, starttime=starttime)
    else:
        if starttime is not None and endtime is not None:
            # just cut
            st.trim(starttime, endtime)

    # rotate
    if rotate_flag:
        st = rotate_stream(st, event_latitude, event_longitude,
                           inventory=inventory, mode="ALL->RT",
                           sanity_check=sanity_check)

    # Convert to single precision to save space.
    for tr in st:
        tr.data = np.require(tr.data, dtype="float32")

    # transfer back to trace if input type is Trace
    if _is_trace:
        st = st[0]

    return st
