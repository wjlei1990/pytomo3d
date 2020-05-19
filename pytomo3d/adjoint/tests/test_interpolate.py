import obspy
import numpy as np


def test_obspy_interpolate():
    tr = obspy.read()[0]
    npts = len(tr.data)
    tr.data = np.zeros(npts)
    tr.interpolate(1.5 * tr.stats.sampling_rate,
                   starttime=tr.stats.starttime+5, npts=npts)
    assert not np.isnan(tr.data).any()
