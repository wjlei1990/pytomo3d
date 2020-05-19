"""
This script generates the adjoint stations from stations.json file and
measurements files. The output file is STATIONS_ADJOINT, which will
be used in the adjoint simulations later.
"""
from .utils import write_stations_file


def extract_usable_stations_from_one_period(measures):
    """ Extract usable stations and channels from measurements file """
    stations = []
    channels = []
    for sta, sta_info in measures.items():
        n_measure_sta = 0
        # append each usable channel
        for chan, chan_info in sta_info.items():
            n_measure_chan = len(chan_info)
            if n_measure_chan > 0:
                channels.append(chan)
            n_measure_sta += n_measure_chan
        # append each usable station
        if n_measure_sta > 0:
            stations.append(sta)

    return stations, channels


def extract_usable_stations_from_measurements(measurements):
    stations = set()
    channels = set()

    for period, measures in measurements.items():
        stations_one_period, channels_one_period = \
            extract_usable_stations_from_one_period(measures)
        print("[Period:%s]Number of stations and channels: %d, %d" %
              (period, len(stations_one_period), len(channels_one_period)))
        stations = stations.union(set(stations_one_period))
        channels = channels.union(set(channels_one_period))

    print("Total number of station and channels: %d, %d" %
          (len(stations), len(channels)))

    return stations, channels


def extract_one_station(chan, stations):
    nw, sta, loc, comp = chan.split(".")
    if comp[-1] == "Z":
        # directly get the station information
        info = stations[chan]
    else:
        # if horizontal components, first try BHE and then try BH1
        new_id_e = "%s.%s.%s.%sE" % (nw, sta, loc, comp[0:2])
        new_id_1 = "%s.%s.%s.%s1" % (nw, sta, loc, comp[0:2])
        if new_id_e in stations:
            info = stations[new_id_e]
        elif new_id_1 in stations:
            info = stations[new_id_1]
        else:
            raise ValueError("Can not locate station(%s) in staitons file"
                             % (chan))
    return info


def prepare_adjoint_station_information(usable_channels, stations):
    """ Based on usable channels, extract adjoint station information """
    adjoint_stations_info = {}

    for chan_id in usable_channels:
        nw, sta, loc, comp = chan_id.split(".")
        info = extract_one_station(chan_id, stations)
        sta_id = "%s.%s" % (nw, sta)
        if sta_id not in adjoint_stations_info:
            # if no previous, just add
            adjoint_stations_info[sta_id] = info
        else:
            # if previous, check if current is Z component
            if comp[-1] == "Z":
                adjoint_stations_info[sta_id] = info
            else:
                continue

    adjoint_stations = {}
    for sta_id, sta_info in adjoint_stations_info.items():
        adjoint_stations[sta_id] = [
            sta_info["latitude"], sta_info["longitude"],
            sta_info["elevation"], sta_info["depth"]]

    return adjoint_stations


def check_adjoint_stations_consistency(adjoint_stations, usable_stations):
    if len(adjoint_stations) != len(usable_stations):
        raise ValueError("Inconsistent between adjoint_stations and "
                         "usable_stations")

    set1 = set(adjoint_stations.keys())
    set2 = set(usable_stations)
    if set1 != set2:
        print("Stations more: %s" % (set1 - set2))
        print("Stations less: %s" % (set2 - set1))
        raise ValueError("Inconsistent between adjoint_stations and "
                         "usable_stations")

    print("Validation check passed")


def benchmark_stations(adjoint_stations):
    """
    Benchmark a few common stations(latitude and longitude)
    just to check if it is correct
    """
    threshold = 0.01

    def is_close(values, true_values):
        for _v1, _v2 in zip(values, true_values):
            if abs(_v1 - _v2) > threshold:
                return False
        return True

    true_values = {
        "II.AAK": [42.6375, 74.4942], "II.ABPO": [-19.0180, 47.2290],
        "II.EFI": [-51.6753, -58.0637], "IU.AFI": [-13.9093, -171.7773],
        "IU.ANMO": [34.9460, -106.4571], "G.CAN": [-35.3187, 148.9963]
    }

    npass = 0
    nfail = 0
    for key in true_values:
        if key not in adjoint_stations:
            continue
        if not is_close(adjoint_stations[key], true_values[key]):
            print("Fails at benchmark station %s" % key)
            nfail += 1
        else:
            npass += 1

    if nfail != 0:
        raise ValueError("Number of benchmark fails: %d" % nfail)

    return npass


def generate_adjoint_stations(measurements, stations, outputfn,
                              benchmark_flag=True):
    """
    This program takes in measurements and stations, and output
    the STATIONS_ADJOINT for stations has adjoint measurements

    :param measurements: dict contains measurements information
        from several period bands, generated by
        pypaw-measure_adjoint_asdf
    :type measurements: dict
    :param stations: dict contains station information, including
        station location and instrument information for all
        channels
    :type stations: dict
    :param outputfn: output STATIONS file
    :type outputfn: str
    :param benchmark_flag: whether benchmark some stations with
        standard locations values
    :type benchmark_flag: bool
    """
    usable_stations, usable_channels = \
        extract_usable_stations_from_measurements(measurements)

    adjoint_stations = prepare_adjoint_station_information(
        usable_channels, stations)

    check_adjoint_stations_consistency(adjoint_stations, usable_stations)

    if benchmark_flag:
        npass = benchmark_stations(adjoint_stations)
        print("Benchmark passed at level: %d" % npass)

    print("Write output station file in: %s" % outputfn)
    write_stations_file(adjoint_stations, outputfn)
