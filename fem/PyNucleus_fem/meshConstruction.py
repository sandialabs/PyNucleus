###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
from PyNucleus_base import INDEX, REAL
from . mesh import mesh2d
from meshpy.triangle import MeshInfo, build
from scipy.spatial import cKDTree
import logging

LOGGER = logging.getLogger(__name__)


class segment:
    def __init__(self, points, facets, holes=[]):
        self.points = points
        self.facets = facets
        self.holes = holes
        self.meshTransformations = []

    def __add__(self, other):
        if isinstance(other, (tuple, np.ndarray)):
            newPoints = [(other[0]+p[0], other[1]+p[1]) for p in self.points]
            newHoles = [(other[0]+p[0], other[1]+p[1]) for p in self.holes]
            newSegment = segment(newPoints, self.facets, newHoles)

            for t in self.meshTransformations:
                def transform(x1, x2, xNew):
                    xTemp = xNew-other
                    t(x1-other, x2-other, xTemp)
                    xNew[:] = other+xTemp

                newSegment.meshTransformations.append(transform)

            return newSegment
        elif isinstance(other, segment):
            points = self.points+other.points
            holes = self.holes+other.holes
            facets = []
            offset = len(self.points)
            for f in self.facets:
                facets.append(f)
            for f in other.facets:
                f2 = (f[0]+offset, f[1]+offset)
                facets.append(f2)

            kd = cKDTree(points)
            idx = -np.ones((len(points)), dtype=INDEX)
            idxUnique = -np.ones((len(points)), dtype=INDEX)
            for t in kd.query_pairs(1e-6):
                idx[max(t)] = min(t)
            k = 0
            for i in range(idx.shape[0]):
                if idx[i] == -1:
                    idx[i] = k
                    idxUnique[k] = i
                    k += 1
                else:
                    idx[i] = idx[idx[i]]
            idxUnique = idxUnique[:k]
            points = [points[i] for i in idxUnique]
            facets = [(idx[f[0]], idx[f[1]]) for f in facets]

            sumSeg = segment(points, facets, holes)
            sumSeg.meshTransformations = self.meshTransformations+other.meshTransformations
            return sumSeg
        else:
            raise NotImplementedError(other)

    def __mul__(self, other):
        if isinstance(other, tuple):
            c = np.array(other[0])
            angle = other[1]
            rot = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])

            points = [c+rot.dot(p-c) for p in self.points]
            holes = [c+rot.dot(p-c) for p in self.holes]
            newSegment = segment(points, self.facets, holes)

            for t in self.meshTransformations:
                def transform(x1, x2, xNew):
                    xTemp = c+rot.T.dot(xNew-c)
                    t(c+rot.T.dot(x1-c),
                      c+rot.T.dot(x2-c),
                      xTemp)
                    xNew[:] = c+rot.dot(xTemp-c)

                newSegment.meshTransformations.append(transform)

            return newSegment
        else:
            raise NotImplementedError()

    def plot(self, plotArrows=False):
        import matplotlib.pyplot as plt
        plt.scatter([p[0] for p in self.points], [p[1] for p in self.points])
        for f in self.facets:
            plt.plot([self.points[f[0]][0], self.points[f[1]][0]],
                     [self.points[f[0]][1], self.points[f[1]][1]])
            if plotArrows:
                plt.arrow(self.points[f[0]][0], self.points[f[0]][1],
                          0.5*(self.points[f[1]][0]-self.points[f[0]][0]),
                          0.5*(self.points[f[1]][1]-self.points[f[0]][1]),
                          head_width=0.05, head_length=0.1)

    def get_num_points(self):
        return len(self.points)

    def get_num_facets(self):
        return len(self.facets)

    def get_num_holes(self):
        return len(self.holes)

    def get_num_mesh_transformations(self):
        return len(self.meshTransformations)

    num_points = property(fget=get_num_points)
    num_facets = property(fget=get_num_facets)
    num_holes = property(fget=get_num_holes)
    num_mesh_transformations = property(fget=get_num_mesh_transformations)

    def mesh(self, **kwargs):
        mesh_info = MeshInfo()
        mesh_info.set_points(self.points)
        mesh_info.set_facets(self.facets)
        mesh_info.set_holes(self.holes)

        if 'min_angle' not in kwargs:
            kwargs['min_angle'] = 30

        if 'h' in kwargs:
            h = kwargs.pop('h')
            if 'href' in kwargs:
                href = kwargs.pop('href')
                for k in range(href):
                    fraction = 0.8**k
                    kwargs['max_volume'] = 0.5 * h**2 * fraction
                    mesh_meshpy = build(mesh_info, **kwargs)
                    mesh = mesh2d(np.array(mesh_meshpy.points, dtype=REAL),
                                  np.array(mesh_meshpy.elements, dtype=INDEX))
                    if mesh.h <= h:
                        break
                else:
                    LOGGER.warn("Meshed {} times, but could not achieve h={}. Instead h={}.".format(href, h, mesh.h))
            else:
                kwargs['max_volume'] = 0.5 * h**2
                mesh_meshpy = build(mesh_info, **kwargs)
                mesh = mesh2d(np.array(mesh_meshpy.points, dtype=REAL),
                              np.array(mesh_meshpy.elements, dtype=INDEX))
        else:
            mesh_meshpy = build(mesh_info, **kwargs)
            mesh = mesh2d(np.array(mesh_meshpy.points, dtype=REAL),
                          np.array(mesh_meshpy.elements, dtype=INDEX))
        mesh.setMeshTransformation(self.getMeshTransformer())
        return mesh

    def getMeshTransformer(self):
        from . meshCy import meshTransformer
        from . meshCy import decode_edge_python

        class myMeshTransformer(meshTransformer):
            def __init__(self, meshTransformations):
                self.meshTransformations = meshTransformations

            def __call__(self, mesh, lookup):
                if len(self.meshTransformations) == 0:
                    return
                for encodeVal in lookup:
                    e = decode_edge_python(encodeVal)
                    x1 = mesh.vertices_as_array[e[0], :]
                    x2 = mesh.vertices_as_array[e[1], :]
                    vertexNo = lookup[encodeVal]
                    xNew = mesh.vertices_as_array[vertexNo, :]
                    for t in self.meshTransformations:
                        if t(x1, x2, xNew):
                            break

        return myMeshTransformer(self.meshTransformations)


class circularSegment(segment):
    def __init__(self, center, radius, start_angle, stop_angle, num_points_per_unit_len=None, num_points=None):
        if num_points_per_unit_len is None and num_points is None:
            num_points = 9
        elif num_points is None:
            num_points = int(np.ceil(radius*(stop_angle-start_angle) * num_points_per_unit_len))+1
        if stop_angle-start_angle < 1e-9:
            points = []
            facets = []
        else:
            if abs(stop_angle-start_angle-2*np.pi) < 1e-9:
                points = [(center[0]+radius*np.cos(theta),
                           center[1]+radius*np.sin(theta)) for theta in np.linspace(start_angle, stop_angle, num_points-1, endpoint=False)]
                facets = [(i, i+1) for i in range(num_points-2)]+[(num_points-2, 0)]
            else:
                points = [(center[0]+radius*np.cos(theta),
                           center[1]+radius*np.sin(theta)) for theta in np.linspace(start_angle, stop_angle, num_points)]
                facets = [(i, i+1) for i in range(num_points-1)]
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.stop_angle = stop_angle
        super(circularSegment, self).__init__(points, facets)
        self.meshTransformations = [self.meshTransformation]

    def meshTransformation(self, x1, x2, xNew):
        rNew = np.linalg.norm(xNew-self.center)
        if rNew <= self.radius:
            theta = np.arctan2(xNew[1]-self.center[1],
                               xNew[0]-self.center[0])
            if theta < 0:
                theta += 2*np.pi
            assert 0 <= theta and theta <= 2*np.pi, (theta, 2*np.pi-theta)
            if (self.start_angle <= theta) and (theta <= self.stop_angle):
                if np.vdot(x1-self.center, x2-self.center) <= 0.:
                    return
                r1 = np.linalg.norm(x1-self.center)
                r2 = np.linalg.norm(x2-self.center)
                r = 0.5*r1+0.5*r2
                if r > 2*rNew:
                    print(r, rNew)
                xNew[:] = self.center + (xNew-self.center)*r/rNew


class circle(circularSegment):
    def __init__(self, center, radius, num_points_per_unit_len=None, num_points=None):
        super(circle, self).__init__(center, radius, 0, 2*np.pi, num_points_per_unit_len, num_points)
        self.points.append(center)


class line(segment):
    def __init__(self, start, end, num_points=None, num_points_per_unit_len=None):
        length2 = (end[0]-start[0])**2 + (end[1]-start[1])**2
        if num_points_per_unit_len is None and num_points is None:
            num_points = 2
        elif num_points_per_unit_len is not None:
            length = np.sqrt(length2)
            num_points = int(np.ceil(length*num_points_per_unit_len))+1
        if length2 < 1e-9:
            points = []
            facets = []
        else:
            points = [(start[0]+t*(end[0]-start[0]),
                       start[1]+t*(end[1]-start[1])) for t in np.linspace(0, 1, num_points)]
            facets = [(i, i+1) for i in range(num_points-1)]
        super(line, self).__init__(points, facets)


def polygon(points, doClose=True, num_points=None, num_points_per_unit_len=None):
    if num_points is None:
        num_points = [None]*len(points)
    elif doClose:
        assert len(num_points) == len(points)
    else:
        assert len(num_points) == len(points)-1
    segments = line(points[0], points[1], num_points=num_points[0], num_points_per_unit_len=num_points_per_unit_len)
    for i in range(1, len(points)-1):
        segments += line(points[i], points[i+1], num_points=num_points[i], num_points_per_unit_len=num_points_per_unit_len)
    if doClose:
        segments += line(points[len(points)-1], points[0], num_points=num_points[len(points)-1], num_points_per_unit_len=num_points_per_unit_len)
    return segments


def rectangle(a, b, num_points=None, num_points_per_unit_len=None):
    assert a[0] < b[0]
    assert a[1] < b[1]
    points = [a, (b[0], a[0]), b, (a[0], b[0])]
    rect = polygon(points, doClose=True, num_points=num_points, num_points_per_unit_len=num_points_per_unit_len)

    def meshTransformation(x1, x2, xNew):
        eps = 1e-10
        if ((a[0]-eps <= x1[0] <= b[0]+eps) and (a[1]-eps <= x1[1] <= b[1]+eps) and
                (a[0]-eps <= x2[0] <= b[0]+eps) and (a[1]-eps <= x2[1] <= b[1]+eps)):
            xNew[:] = 0.5*(x1+x2)
            return True

    rect.meshTransformation = [meshTransformation]
    return rect


class transformationRestriction(segment):
    def __init__(self, seg, p1, p2):
        super(transformationRestriction, self).__init__(seg.points, seg.facets)
        for t in seg.meshTransformations:
            def transform(x1, x2, xNew):
                if ((p1[0] <= xNew[0]) and (xNew[0] <= p2[0]) and
                        (p1[1] <= xNew[1]) and (xNew[1] <= p2[1])):
                    t(x1, x2, xNew)
            self.meshTransformations.append(transform)
