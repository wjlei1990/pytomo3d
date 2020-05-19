import numpy as np
import json
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def dump_json(content, fn):
    with open(fn, 'w') as fh:
        json.dump(content, fh, indent=2, sort_keys=True)


def load_json(fn):
    with open(fn) as fh:
        return json.load(fh)


def plot_basemap(lat=0, lon=180):
    m = Basemap(projection='cyl', lon_0=lon, lat_0=lat,
                resolution='c')

    m.drawcoastlines()
    m.fillcontinents()
    m.drawparallels(np.arange(-90., 120., 30.))
    m.drawmeridians(np.arange(0., 420., 30.))
    m.drawmapboundary()

    return m


def plot_events_map(info, marker="o", s=2, color='r', zorder=10,
                    figname=None):
    plt.figure(figsize=(10, 5))
    m = plot_basemap()

    lats = [info[event]["latitude"] for event in info]
    lons = [info[event]["longitude"] for event in info]
    lons = [(x + 360) % 360 for x in lons]

    x, y = m(lons, lats)
    m.scatter(x, y, marker=marker, s=s, color=color, zorder=zorder)

    if figname is None:
        plt.show()
    else:
        print("figure saved to: %s" % figname)
        plt.savefig(figname)


def _in_box(lat, lon, box):
    if box[0][0] > box[0][1] or box[1][0] > box[1][1]:
        raise ValueError("box bounds is wrong: %s" % box)

    if box[0][0] <= lat and lat <= box[0][1] and box[1][0] <= lon and \
            lon <= box[1][1]:
        return True
    else:
        return False


def get_events_in_boxes(event_info, boxes):
    inbox_events = {}
    for event in event_info:
        lat = event_info[event]["latitude"]
        lon = event_info[event]["longitude"]
        lon = (lon + 360.0) % 360
        for idx, box in enumerate(boxes):
            if _in_box(lat, lon, box):
                inbox_events[event] = idx
                break

    return inbox_events


def stats_box(inbox_events, weights):
    counts = {}
    for event, idx in inbox_events.items():
        if idx not in counts:
            counts[idx] = 0
        counts[idx] += 1

    for idx in range(len(counts)):
        print("[box %d] count: %d -- weight: %f"
              % (idx, counts[idx], weights[idx]))

    print("Number of inbox events: %d" % len(inbox_events))
