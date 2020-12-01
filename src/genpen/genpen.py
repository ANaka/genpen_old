import bezier
from dataclasses import asdict, dataclass, field
import numpy as np
import shapely.geometry as sg
import shapely.affinity as sa
import shapely.ops as so
from shapely.geometry import box, MultiLineString, Point, MultiPoint, Polygon, MultiPolygon, LineString
from shapely import speedups
from tqdm import tqdm
from typing import List
import vsketch
import functools
from genpen import utils
from shapely.errors import TopologicalError

def get_width(polygon):
    return polygon.bounds[2] - polygon.bounds[0]

def get_height(polygon):
    return polygon.bounds[3] - polygon.bounds[1]

### parameters 
class DataClassBase(object):
    def asdict(self):
        return asdict(self)

# @dataclass
# class ParamSeq(DataClassBase):

class Poly(object):
    def __init__(self, p:sg.Polygon):
        self.p = p
        self.fill = None
        
    @property
    def width(self):
        return get_width(self.p)
    
    @property
    def height(self):
        return get_height(self.p)
    
    def scale_tran(self, d_buffer, d_translate, angle, cap_style=2, join_style=2):
        return scale_tran(self.p)

    def hatch(self, angle, spacing):
        return hatchbox(self.p, angle, spacing)
    
    def fill_scale_trans(self, d_buffers, d_translates, angles, cap_style=2, join_style=2):
        ssps = scale_trans(self.p, d_buffers, d_translates, angles, cap_style=cap_style, join_style=join_style)
        self.fill = merge_LineStrings([p.boundary for p in ssps])
        
    def fill_hatch(self, angle, spacing):
        self.fill = hatchbox(self.p, angle, spacing)
    
    @property
    def intersection_fill(self):
        try:
            ifill = self.fill.intersection(self.p)
        except TopologicalError:
            self.p = self.p.buffer(1e-6)
            ifill = self.fill.intersection(self.p)
        try:
            return collection_to_mls(ifill)
        except:
            return MultiLineString([ifill])


def centered_box(point, width, height):
    return sg.box(point.x-width/2, point.y-height/2, point.x+width/2, point.y+height/2)

def overlay_grid(poly, xstep, ystep):
    xmin, ymin, xmax, ymax = poly.envelope.bounds
    xbins=np.arange(xmin, xmax, xstep)
    ybins=np.arange(ymin, ymax, ystep)
    return xbins, ybins

def get_random_points_in_polygon(polygon, n_points=1, xgen=None, ygen=None):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    if xgen == None:
        xgen = lambda size=None: np.random.uniform(minx, maxx, size)
    if ygen == None:
        ygen = lambda size=None: np.random.uniform(miny, maxy, size)

    n_attempts = 0
    while True:
        point = Point((xgen(), ygen()))
        if polygon.contains(point):
            points.append(point)
        n_attempts += 1
        if n_attempts > (n_points * 20):
            print('too many attempts being rejected in get_random_points_in_polygon')
            break
        if len(points) == n_points:
            return points
        

def get_random_point_in_polygon(polygon):
    return get_random_points_in_polygon(polygon, n_points=1)[0]

def get_rad(circle, use_circumference=False):

    if use_circumference:
        return circle.boundary.length / (np.pi * 2)
    else:
        return (circle.bounds[2] - circle.bounds[0]) / 2
    
    
    
### Shading ###

def scale_tran(p, d_buffer, d_translate, angle, cap_style=2, join_style=2):
    xoff = np.cos(angle) * d_translate
    yoff = np.sin(angle) * d_translate
    bp = p.buffer(d_buffer, cap_style=cap_style, join_style=join_style)
    btp = sa.translate(bp, xoff=xoff, yoff=yoff)
    return btp


def scale_trans(p, d_buffers, d_translates, angles, cap_style=2, join_style=2, return_original=True):
    
    d_translates = utils.ensure_collection(d_translates, length=len(d_buffers))
    angles = utils.ensure_collection(angles, length=len(d_buffers))
    
    ssps = []
    if return_original:
        ssps.append(p)
    
    ssp = p
    for d_buffer, d_translate, angle in zip(d_buffers, d_translates, angles):
        ssp = scale_tran(ssp, d_buffer, d_translate, angle, cap_style, join_style)
        if ssp.area < np.finfo(float).eps:
            break
        ssps.append(ssp)
    return ssps


def hatchbox(rect, angle, spacing):
    """
    returns a Shapely geometry (MULTILINESTRING, or more rarely,
    GEOMETRYCOLLECTION) for a simple hatched rectangle.

    args:
    rect - a Shapely geometry for the outer boundary of the hatch
           Likely most useful if it really is a rectangle

    angle - angle of hatch lines, conventional anticlockwise -ve

    spacing - spacing between hatch lines

    GEOMETRYCOLLECTION case occurs when a hatch line intersects with
    the corner of the clipping rectangle, which produces a point
    along with the usual lines.
    """

    (llx, lly, urx, ury) = rect.bounds
    centre_x = (urx + llx) / 2
    centre_y = (ury + lly) / 2
    diagonal_length = ((urx - llx) ** 2 + (ury - lly) ** 2) ** 0.5
    number_of_lines = 2 + int(diagonal_length / spacing)
    hatch_length = spacing * (number_of_lines - 1)

    # build a square (of side hatch_length) horizontal lines
    # centred on centroid of the bounding box, 'spacing' units apart
    coords = []
    for i in range(number_of_lines):
        if i % 2:
            coords.extend([((centre_x - hatch_length / 2, centre_y
                          - hatch_length / 2 + i * spacing), (centre_x
                          + hatch_length / 2, centre_y - hatch_length
                          / 2 + i * spacing))])
        else:
            coords.extend([((centre_x + hatch_length / 2, centre_y
                          - hatch_length / 2 + i * spacing), (centre_x
                          - hatch_length / 2, centre_y - hatch_length
                          / 2 + i * spacing))])
            
    # turn array into Shapely object
    lines = sg.MultiLineString(coords)
    # Rotate by angle around box centre
    lines = sa.rotate(lines, angle, origin='centroid', use_radians=False)
    # return clipped array
    return rect.intersection(lines)

def connect_hatchlines(hatchlines, dist_thresh):
    linestrings = list(hatchlines)
    merged_linestrings = []

    current_ls = linestrings.pop(0)
    while len(linestrings) > 0:
        next_ls = linestrings.pop(0)
        dist = Point(current_ls.coords[-1]).distance(Point(next_ls.coords[0]))
        if dist <= dist_thresh:
            current_ls = LineString(list(current_ls.coords) + list(next_ls.coords))
        else:
            merged_linestrings.append(current_ls)
            current_ls = next_ls
    merged_linestrings.append(current_ls)
    return merged_linestrings


def connected_hatchbox(rect, angle, spacing, dist_thresh):
    hatches = hatchbox(rect, angle, spacing)
    return hatches




def morsify(ls, buffer_factor=0.01):
    return ls.buffer(buffer_factor).buffer(-buffer_factor).boundary


def add_jittered_midpoints(ls, n_midpoints, xstd, ystd, xbias=0, ybias=0):
    eval_range = np.linspace(0., 1., n_midpoints+2)
    pts = np.stack([ls.interpolate(t, normalized=True) for t in eval_range])
    x_jitter = np.random.randn(n_midpoints, 1) * xstd + xbias
    y_jitter = np.random.randn(n_midpoints, 1) * ystd + ybias
    pts[1:-1] += np.concatenate([x_jitter, y_jitter], axis=1)
    return shg.asLineString(pts)


def LineString_to_jittered_bezier(ls, n_midpoints=1, xbias=0., xstd=0., ybias=0., ystd=0., normalized=True, n_eval_points=50):
    if normalized==True:
        xbias = xbias * ls.length
        xstd = xstd * ls.length
        ybias = ybias * ls.length
        ystd = ystd*ls.length

    jitter_ls = add_jittered_midpoints(ls, n_midpoints=n_midpoints, xbias=xbias, xstd=xstd, ybias=ybias, ystd=ystd,)
    curve1 = bezier.Curve(np.asfortranarray(jitter_ls).T, degree=n_midpoints+1)
    bez = curve1.evaluate_multi(np.linspace(0., 1., n_eval_points))
    return sg.asLineString(bez.T)


def circle_pack_within_poly(poly, rads, max_additions=np.inf, progress_bar=True):
    # max additions limit doesn't really work

    radi = iter(rads)
    if progress_bar:
        pbar = tqdm(total=len(rads))

    # init
    n_additions = 0
    rad = next(radi)
    circles = []
    all_circles = MultiPolygon()
    init_attempts = 0
    while len(circles) == 0:
        if init_attempts > 10:
            break
        try:
            pt = get_random_points_in_polygon(poly.buffer(-rad))[0]
            c = pt.buffer(rad)
            c.rad = rad
            if (not c.intersects(all_circles)) and (poly.contains(c)):
                circles.append(c)
                all_circles = so.unary_union([all_circles, c])
                n_additions += 1
        except:
            init_attempts += 1


    # main loop

    while True:
        try:

            while n_additions <= max_additions:
                at_least_one_addition = False
                circle_order = np.random.permutation(len(circles))
                for i in circle_order:
                    seed_circle = circles[i]
                    seed_rad = seed_circle.rad
                    search_ring = seed_circle.buffer(rad).boundary
                    search_locs = np.arange(0, search_ring.length, (2*rad))
                    scrambled_search_locs = np.random.permutation(search_locs)
                    for sl in scrambled_search_locs:
                        pt = search_ring.interpolate(sl)
                        c = pt.buffer(rad)
                        c.rad = rad
                        if (not c.intersects(all_circles)) and (poly.contains(c)):
                            circles.append(c)
                            all_circles = sho.unary_union([all_circles, c])
                            at_least_one_addition = True
                            n_additions += 1
                if not at_least_one_addition:  # if not working, break and reduce rad
                    break
            rad = next(radi)
            n_additions = 0
            if progress_bar:
                pbar.update()
        except StopIteration:
            return all_circles




@dataclass
class NuzzleParams(DataClassBase):
    dilate_multiplier_min: float = 0.01
    dilate_multiplier_max: float = 0.1
    erode_multiplier:float = -1.1
    n_iters:int = 1
    dilate_join_style:int = 1
    dilate_cap_style:int = 1
    erode_join_style:int = 1
    erode_cap_style:int = 1


def get_rad(circle, use_circumference=False):

    if use_circumference:
        return circle.boundary.length / (np.pi * 2)
    else:
        return (circle.bounds[2] - circle.bounds[0]) / 2


def nuzzle_poly(poly, neighbors,
                dilate_multiplier_min,
                dilate_multiplier_max,
                erode_multiplier=-1.1,
                n_iters=1,
                dilate_join_style=1,
                dilate_cap_style=1,
                erode_join_style=1,
                erode_cap_style=1,
               ):
    rad = get_rad(poly)
    neighbor_union = so.unary_union(neighbors)
    for i in range(n_iters):
        d = rad * np.random.uniform(dilate_multiplier_min, dilate_multiplier_max)
        bc = poly.buffer(d, join_style=dilate_join_style, cap_style=dilate_cap_style)
        if bc.intersects(neighbor_union):
            e = d * erode_multiplier
            bc = bc.difference(neighbor_union).buffer(e,join_style=erode_join_style, cap_style=erode_cap_style)
        poly = bc
    return poly


def nuzzle_em(polys, nuzzle_params, n_iters=20):
    for n in range(n_iters):
        polys = list(polys)
        order = np.random.permutation(len(polys))
        for i in order:
            try:
                poly = polys[i]
                other_polys = MultiPolygon([c for j, c in enumerate(polys) if j!=i])
                new_poly = nuzzle_poly(poly, other_polys, **nuzzle_params.asdict())
                polys[i] = new_poly
            except:
                pass
        polys = merge_Polygons(polys)
    return merge_Polygons(polys)


def random_split(geoms, n_layers):
    splits = np.random.choice(n_layers, size=len(geoms))
    layers = []

    for i in range(n_layers):
        layers.append([geoms[j] for j in np.nonzero(splits==i)[0]])
    return layers



def circle_growth(poly, rads, obj_func, max_additions=np.inf, progress_bar=True):
    # max additions limit doesn't really work

    radi = iter(rads)
    if progress_bar:
        pbar = tqdm(total=len(rads))

    # init
    n_additions = 0
    rad = next(radi)
    circles = []
    all_circles = MultiPolygon()
    while len(circles) == 0:
        pt = get_random_points_in_polygon(poly.buffer(-rad))[0]
        c = pt.buffer(rad)
        c.rad = rad
        if (not c.intersects(all_circles)) and (poly.contains(c)):
            circles.append(c)
            all_circles = so.unary_union([all_circles, c])
            n_additions += 1

    # main loop

    while True:
        try:

            while n_additions <= max_additions:
                at_least_one_addition = False
                circle_order = np.random.permutation(len(circles))
                for i in circle_order:
                    seed_circle = circles[i]
                    seed_rad = seed_circle.rad
                    search_ring = seed_circle.buffer(rad).boundary
                    search_locs = np.arange(0, search_ring.length, (2*rad))
                    scrambled_search_locs = np.random.permutation(search_locs)
                    for sl in scrambled_search_locs:
                        pt = search_ring.interpolate(sl)
                        c = pt.buffer(rad)
                        c.rad = rad
                        if (not c.intersects(all_circles)) and (poly.contains(c)):
                            circles.append(c)
                            all_circles = sho.unary_union([all_circles, c])
                            at_least_one_addition = True
                            n_additions += 1
                if not at_least_one_addition:  # if not working, break and reduce rad
                    break
            rad = next(radi)
            n_additions = 0
            if progress_bar:
                pbar.update()
        except StopIteration:
            return all_circles



# Cell
def overlay_grid(poly, xstep, ystep):
    xmin, ymin, xmax, ymax = poly.envelope.bounds
    xbins=np.arange(xmin, xmax, xstep)
    ybins=np.arange(ymin, ymax, ystep)
    return xbins, ybins

# Cell
class PerlinGrid(object):

    def __init__(self, poly, xstep=0.1, ystep=0.1, lod=4, falloff=None, noiseSeed=71, noise_scale=0.001):

        self.p = poly
        self.xbins, self.ybins = overlay_grid(poly, xstep, ystep)
        self.gxs, self.gys = np.meshgrid(self.xbins, self.ybins)

        self.vsk = vsketch.Vsketch()
        self.lod = lod
        self.falloff = falloff
        self.noiseSeed = noiseSeed
        self.noise_scale = noise_scale
        self.z = self.make_noisegrid()
        self.a = np.interp(self.z, [0, 1], [0, np.pi*2])


    def make_noisegrid(self):
        self.vsk.noiseSeed(self.noiseSeed)
        self.vsk.noiseDetail(lod=self.lod, falloff=self.falloff)
        zs = []
        for x,y in zip(self.gxs.ravel(), self.gys.ravel()):
            x = x * self.noise_scale
            y = y * self.noise_scale
            zs.append(self.vsk.noise(x=x, y=y))
        return np.array(zs).reshape(self.gxs.shape)


# Cell
class Particle(object):

    def __init__(self, pos, grid, stepsize=1):
        self.pos = Point(pos)
        self.grid = grid
        self.stepsize = stepsize
        self.n_step = 0
        self.pts = [self.pos]
        self.in_bounds = True

    @property
    def x(self):
        return self.pos.x
    
    @property
    def y(self):
        return self.pos.y
    
    @property
    def xy(self):
        return np.array([self.x, self.y])

    @property
    def line(self):
        return LineString(self.pts)
    
    def get_closest_bins(self):
        self.xind = np.argmin(abs(self.grid.xbins-self.x))
        self.yind = np.argmin(abs(self.grid.ybins-self.y))

    def get_angle(self):
        self.a = self.grid.a[self.yind, self.xind]

    def check_if_in_bounds(self):
        self.in_bounds = self.grid.p.contains(self.pos)

    def calc_step(self):
        self.get_closest_bins()
        self.get_angle()
        self.dx = np.cos(self.a) * self.stepsize
        self.dy = np.sin(self.a) * self.stepsize


    def step(self):
        self.check_if_in_bounds()
        if self.in_bounds:
            self.calc_step()
            self.pos = sa.translate(self.pos, xoff=self.dx, yoff=self.dy)
            self.pts.append(self.pos)



def buffer_individually(geoms, distance, cap_style=2, join_style=2):
    n_geoms = len(geoms)
    ds = utils.ensure_collection(distance, n_geoms)
    css = utils.ensure_collection(cap_style, n_geoms)
    jss = utils.ensure_collection(join_style, n_geoms)
    bgs = []
    for i in range(n_geoms):
        bg = geoms[i].buffer(ds[i], cap_style=css[i], join_style=jss[i])
        bgs.append(bg)
    return MultiPolygon(bgs)


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


def collection_to_mls(collection):
    lss = [g for g in collection if 'LineString' in g.type]
    lss = [ls for ls in lss if ls.length > np.finfo(float).eps]
    return merge_LineStrings(lss)