import bezier
from dataclasses import asdict, dataclass
import numpy as np
import shapely.geometry as sg
import shapely.affinity as sa
import shapely.ops as so

from shapely import speedups
from tqdm import tqdm

class DistanceConverter(object):

    def __init__(self, d, unit):
        setattr(self, unit, d)

    @property
    def inches(self):
        return self._inches

    @inches.setter
    def inches(self, inches):
        self._inches = inches
        self._mm = 25.4 * inches

    @property
    def mm(self):
        return self._mm

    @mm.setter
    def mm(self, d):
        self._mm = d
        self._inches = d / 25.4


def scalar_to_collection(scalar, length):
    stype = type(scalar)
    return (np.ones(length) * scalar).astype(stype)

def ensure_collection(x, length):
    if np.iterable(x):
        assert len(x) == length
        return x
    else:
        return scalar_to_collection(x, length)
    
def random_split(geoms, n_layers):
    splits = np.random.choice(n_layers, size=len(geoms))
    layers = []

    for i in range(n_layers):
        layers.append([geoms[j] for j in np.nonzero(splits==i)[0]])
    return layers


def merge_LineStrings(mls_list):
    merged_mls = []
    for mls in mls_list:
        if getattr(mls, 'type') == 'MultiLineString':
            merged_mls += list(mls)
        elif getattr(mls, 'type') == 'LineString':
            merged_mls.append(mls)
    return sg.MultiLineString(merged_mls)

def merge_Polygons(mp_list):
    merged_mps = []
    for mp in mp_list:
        if type(mp)==list:
            merged_mps += list(mp)
        elif getattr(mp, 'type') == 'MultiPolygon':
            merged_mps += list(mp)
        elif getattr(mp, 'type') == 'Polygon':
            merged_mps.append(mp)
    return sg.MultiPolygon(merged_mps)