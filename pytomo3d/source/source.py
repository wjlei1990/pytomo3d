#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Source and Receiver classes of Instaseis.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
    Martin van Driel (Martin@vanDriel.de), 2014
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import obspy
import obspy.xseed
import warnings


class CMTSource(object):
    """
    Class to handle a seismic moment tensor source including a source time
    function.
    """
    def __init__(self, origin_time=obspy.UTCDateTime(0),
                 pde_latitude=0.0, pde_longitude=0.0, mb=0.0, ms=0.0,
                 pde_depth_in_m=None, region_tag=None, eventname=None,
                 cmt_time=0.0, half_duration=0.0, latitude=0.0, longitude=0.0,
                 depth_in_m=None, m_rr=0.0, m_tt=0.0, m_pp=0.0, m_rt=0.0,
                 m_rp=0.0, m_tp=0.0):
        """
        :param latitude: latitude of the source in degree
        :param longitude: longitude of the source in degree
        :param depth_in_m: source depth in m
        :param m_rr: moment tensor components in r, theta, phi in Nm
        :param m_tt: moment tensor components in r, theta, phi in Nm
        :param m_pp: moment tensor components in r, theta, phi in Nm
        :param m_rt: moment tensor components in r, theta, phi in Nm
        :param m_rp: moment tensor components in r, theta, phi in Nm
        :param m_tp: moment tensor components in r, theta, phi in Nm
        :param time_shift: correction of the origin time in seconds. only
            useful in the context of finite sources
        :param sliprate: normalized source time function (sliprate)
        :param dt: sampling of the source time function
        :param origin_time: The origin time of the source. This will be the
            time of the first sample in the final seismogram. Be careful to
            adjust it for any time shift or STF (de)convolution effects.

        """
        self.origin_time = origin_time
        self.pde_latitude = pde_latitude
        self.pde_longitude = pde_longitude
        self.pde_depth_in_m = pde_depth_in_m
        self.mb = mb
        self.ms = ms
        self.region_tag = region_tag
        self.eventname = eventname
        self.cmt_time = cmt_time
        self.half_duration = half_duration
        self.latitude = latitude
        self.longitude = longitude
        self.depth_in_m = depth_in_m
        self.m_rr = m_rr
        self.m_tt = m_tt
        self.m_pp = m_pp
        self.m_rt = m_rt
        self.m_rp = m_rp
        self.m_tp = m_tp

    @classmethod
    def from_CMTSOLUTION_file(cls, filename):
        """
        Initialize a source object from a CMTSOLUTION file.

        :param filename: path to the CMTSOLUTION file
        """

        with open(filename, "rt") as f:
            line = f.readline()
            origin_time = line[4:].strip().split()[:6]
            values = list(map(int, origin_time[:-1])) + \
                [float(origin_time[-1])]
            try:
                origin_time = obspy.UTCDateTime(*values)
            except (TypeError, ValueError):
                warnings.warn("Could not determine origin time from line: %s"
                              % line)
                origin_time = obspy.UTCDateTime(0)
            otherinfo = line[4:].strip().split()[6:]
            pde_lat = float(otherinfo[0])
            pde_lon = float(otherinfo[1])
            pde_depth_in_m = float(otherinfo[2]) * 1e3
            mb = float(otherinfo[3])
            ms = float(otherinfo[4])
            region_tag = ' '.join(otherinfo[5:])

            eventname = f.readline().strip().split()[-1]
            time_shift = float(f.readline().strip().split()[-1])
            cmt_time = origin_time + time_shift
            half_duration = float(f.readline().strip().split()[-1])
            latitude = float(f.readline().strip().split()[-1])
            longitude = float(f.readline().strip().split()[-1])
            depth_in_m = float(f.readline().strip().split()[-1]) * 1e3

            # unit: N/m
            m_rr = float(f.readline().strip().split()[-1])  # / 1e7
            m_tt = float(f.readline().strip().split()[-1])  # / 1e7
            m_pp = float(f.readline().strip().split()[-1])  # / 1e7
            m_rt = float(f.readline().strip().split()[-1])  # / 1e7
            m_rp = float(f.readline().strip().split()[-1])  # / 1e7
            m_tp = float(f.readline().strip().split()[-1])  # / 1e7

        return cls(origin_time=origin_time,
                   pde_latitude=pde_lat, pde_longitude=pde_lon, mb=mb, ms=ms,
                   pde_depth_in_m=pde_depth_in_m, region_tag=region_tag,
                   eventname=eventname, cmt_time=cmt_time,
                   half_duration=half_duration, latitude=latitude,
                   longitude=longitude, depth_in_m=depth_in_m,
                   m_rr=m_rr, m_tt=m_tt, m_pp=m_pp, m_rt=m_rt,
                   m_rp=m_rp, m_tp=m_tp)

    @classmethod
    def from_quakeml_file(cls, filename):
        """
        Initizliaze a source object from a quakeml file
        :param filename: path to a quakeml file
        """
        from obspy import read_events
        cat = read_events(filename)
        event = cat[0]
        cmtsolution = event.preferred_origin()
        pdesolution = event.origins[0]

        origin_time = pdesolution.time
        pde_lat = pdesolution.latitude
        pde_lon = pdesolution.longitude
        pde_depth_in_m = pdesolution.depth
        mb = 0.0
        ms = 0.0
        for mag in event.magnitudes:
            if mag.magnitude_type == "mb":
                mb = mag.mag
            elif mag.magnitude_type == "MS":
                ms = mag.mag
        region_tag = event.event_descriptions[0].text

        for descrip in event.event_descriptions:
            if descrip.type == "earthquake name":
                eventname = descrip.text
            else:
                eventname = ""
        cmt_time = cmtsolution.time
        focal_mechanism = event.focal_mechanisms[0]
        half_duration = \
            focal_mechanism.moment_tensor.source_time_function.duration/2.0
        latitude = cmtsolution.latitude
        longitude = cmtsolution.longitude
        depth_in_m = cmtsolution.depth
        tensor = focal_mechanism.moment_tensor.tensor
        m_rr = tensor.m_rr * 1e7
        m_tt = tensor.m_tt * 1e7
        m_pp = tensor.m_pp * 1e7
        m_rt = tensor.m_rt * 1e7
        m_rp = tensor.m_rp * 1e7
        m_tp = tensor.m_tp * 1e7

        return cls(origin_time=origin_time,
                   pde_latitude=pde_lat, pde_longitude=pde_lon, mb=mb, ms=ms,
                   pde_depth_in_m=pde_depth_in_m, region_tag=region_tag,
                   eventname=eventname, cmt_time=cmt_time,
                   half_duration=half_duration, latitude=latitude,
                   longitude=longitude, depth_in_m=depth_in_m,
                   m_rr=m_rr, m_tt=m_tt, m_pp=m_pp, m_rt=m_rt,
                   m_rp=m_rp, m_tp=m_tp)

    def write_CMTSOLUTION_file(self, filename):
        """
        Initialize a source object from a CMTSOLUTION file.

        :param filename: path to the CMTSOLUTION file
        """
        with open(filename, "w") as f:
            # Reconstruct the first line as well as possible. All
            # hypocentral information is missing.
            f.write(
                " PDE %4i %2i %2i %2i %2i %5.2f %8.4f %9.4f %5.1f %.1f %.1f"
                " %s\n" % (
                    self.origin_time.year,
                    self.origin_time.month,
                    self.origin_time.day,
                    self.origin_time.hour,
                    self.origin_time.minute,
                    self.origin_time.second +
                    self.origin_time.microsecond / 1E6,
                    self.pde_latitude,
                    self.pde_longitude,
                    self.pde_depth_in_m / 1e3,
                    self.mb,
                    self.ms,
                    str(self.region_tag)))
            f.write('event name:     %s\n' % (str(self.eventname)))
            f.write('time shift:%12.4f\n' % (self.time_shift,))
            f.write('half duration:%9.4f\n' % (self.half_duration,))
            f.write('latitude:%14.4f\n' % (self.latitude,))
            f.write('longitude:%13.4f\n' % (self.longitude,))
            f.write('depth:%17.4f\n' % (self.depth_in_m / 1e3,))
            f.write('Mrr:%19.6e\n' % self.m_rr)  # * 1e7,))
            f.write('Mtt:%19.6e\n' % self.m_tt)  # * 1e7,))
            f.write('Mpp:%19.6e\n' % self.m_pp)  # * 1e7,))
            f.write('Mrt:%19.6e\n' % self.m_rt)  # * 1e7,))
            f.write('Mrp:%19.6e\n' % self.m_rp)  # * 1e7,))
            f.write('Mtp:%19.6e\n' % self.m_tp)  # * 1e7,))

    @property
    def M0(self):
        """
        Scalar Moment M0 in Nm
        """
        return (self.m_rr ** 2 + self.m_tt ** 2 + self.m_pp ** 2 +
                2 * self.m_rt ** 2 + 2 * self.m_rp ** 2 +
                2 * self.m_tp ** 2) ** 0.5 * 0.5 ** 0.5

    @property
    def moment_magnitude(self):
        """
        Moment magnitude M_w
        """
        return 2.0 / 3.0 * (np.log10(self.M0) - 7.0) - 6.0

    @property
    def time_shift(self):
        """
        Time shift between cmtsolution and pdesolution
        """
        return self.cmt_time - self.origin_time

    @property
    def tensor(self):
        """
        List of moment tensor components in r, theta, phi coordinates:
        [m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]
        """
        return np.array([self.m_rr, self.m_tt, self.m_pp, self.m_rt, self.m_rp,
                         self.m_tp])

    def __str__(self):
        return_str = 'CMTS Source -- %s\n' % self.eventname
        return_str += 'origin time(pde): %s\n' % self.origin_time
        return_str += 'pde location(lat, lon): %f, %f deg\n' \
            % (self.pde_latitude, self.pde_longitude)
        return_str += 'pde depth: %f\n' % self.pde_depth_in_m
        return_str += 'CMT time: %s\n' % self.cmt_time
        return_str += 'CMT location(lat, lon): %f, %f deg\n' \
            % (self.latitude, self.longitude)
        return_str += 'CMT depth: %6.1e km\n' \
                      % (self.depth_in_m / 1e3,)
        return_str += 'half duration: %f\n' % self.half_duration
        return_str += 'moment tensor(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)\n'
        return_str += 'moment tensor: %s\n' \
            % self.tensor
        return_str += 'Magnitude: %4.2f(mw), %4.2f(mb), %4.2f(ms)\n' \
                      % (self.moment_magnitude, self.mb, self.ms)
        return_str += 'region tag: %s' % self.region_tag

        return return_str

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return self.__dict__ == other.__dict__
