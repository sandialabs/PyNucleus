###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from __future__ import division
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
import numpy as np
from PyNucleus_base.factory import factory
from PyNucleus_base.myTypes import INDEX, REAL, TAG
from PyNucleus_base.linear_operators import sparseGraph
from PyNucleus_base import uninitialized, uninitialized_like
from . meshCy import (meshBase,
                      boundaryVertices,
                      boundaryEdges,
                      boundaryFacesWithOrientation,
                      boundaryVerticesFromBoundaryEdges,
                      boundaryEdgesFromBoundaryFaces,
                      radialMeshTransformation)
from . meshPartitioning import (metisMeshPartitioner,
                                regularMeshPartitioner,
                                PartitionerException)
import logging

LOGGER = logging.getLogger(__name__)


# PHYSICAL is the physical boundary of the entire domain
PHYSICAL = TAG(0)
# INTERIOR_NONOVERLAPPING are the interior boundaries of
# non-overlapping subdomains
INTERIOR_NONOVERLAPPING = TAG(-1)
# INTERIOR is the interior boundary of overlapping subdomains
INTERIOR = TAG(-2)
# don't use any boundary
NO_BOUNDARY = np.iinfo(TAG).min

# Types of boundary conditions
DIRICHLET = 0
NEUMANN = 1
HOMOGENEOUS_DIRICHLET = 2
HOMOGENEOUS_NEUMANN = 3
NORM = 4

boundaryConditions = {DIRICHLET: 'Dirichlet',
                      NEUMANN: 'Neumann',
                      HOMOGENEOUS_DIRICHLET: 'homogeneous Dirichlet',
                      HOMOGENEOUS_NEUMANN: 'homogeneous Neumann'}


class meshFactory(factory):
    def __init__(self):
        super(meshFactory, self).__init__()
        self.dims = {}

    def register(self, name, classType, dim, params={}, aliases=[]):
        super(meshFactory, self).register(name, classType, params, aliases)
        name = self.getCanonicalName(name)
        self.dims[name] = dim

    def build(self, name, noRef=0, hTarget=None, surface=False, **kwargs):
        if isinstance(name, meshNd):
            return name
        mesh = super(meshFactory, self).build(name, **kwargs)
        if surface:
            mesh = mesh.get_surface_mesh()
            mesh.removeUnusedVertices()
        from . import P1_DoFMap
        dmTest = P1_DoFMap(mesh, PHYSICAL)
        while dmTest.num_dofs == 0:
            mesh = mesh.refine()
            dmTest = P1_DoFMap(mesh, PHYSICAL)
        if hTarget is None:
            for _ in range(noRef):
                mesh = mesh.refine()
        else:
            assert hTarget > 0
            while mesh.h > hTarget:
                mesh = mesh.refine()
        return mesh

    def getDim(self, name):
        name = self.getCanonicalName(name)
        if name in self.aliases:
            name = self.aliases[name][1]
        return self.dims[name]


def pacman(h=0.1, **kwargs):
    from . meshConstruction import (circularSegment,
                                    line)
    theta = np.pi/5
    center = np.array([0., 0.])
    bottom = np.array([1., 0.])
    top = np.array([np.cos(theta), np.sin(theta)])

    numPointsPerUnitLength = int(np.ceil(1/h))

    domain = (circularSegment(center, 1., theta, 2*np.pi, numPointsPerUnitLength) +
              line(bottom, center) +
              line(center, top))

    mesh = domain.mesh(max_volume=h**2, min_angle=30, **kwargs)
    return mesh


def uniformSquare(N, M=None, ax=0, ay=0, bx=1, by=1, crossed=False, preserveLinesHorizontal=[], preserveLinesVertical=[]):
    if M is None:
        M = max(int(np.around((by-ay)/(bx-ax)))*N, 2)
    assert N >= 2
    assert M >= 2
    xVals = np.linspace(ax, bx, N)
    yVals = np.linspace(ay, by, M)
    x, y = np.meshgrid(xVals, yVals)
    for yVal in preserveLinesHorizontal:
        assert (yVals-yVal).min() < 1e-10
    for xVal in preserveLinesVertical:
        assert (xVals-xVal).min() < 1e-10

    vertices = [np.array([xx, yy]) for xx, yy in
                zip(x.flatten(), y.flatten())]
    cells = []
    if not crossed:
        for i in range(M-1):
            for j in range(N-1):
                # bottom right element
                el = (i*N+j, i*N+j+1, (i+1)*N+j+1)
                cells.append(el)
                # top left element
                el = (i*N+j, (i+1)*N+j+1, (i+1)*N+j)
                cells.append(el)
    else:
        for i in range(M-1):
            for j in range(N-1):
                if i % 2 == 0:
                    if j % 2 == 0:
                        # bottom right element
                        el = (i*N+j, i*N+j+1, (i+1)*N+j+1)
                        cells.append(el)
                        # top left element
                        el = (i*N+j, (i+1)*N+j+1, (i+1)*N+j)
                    else:
                        # bottom left element
                        el = (i*N+j, i*N+j+1, (i+1)*N+j)
                        cells.append(el)
                        # top right element
                        el = (i*N+j+1, (i+1)*N+j+1, (i+1)*N+j)
                else:
                    if j % 2 == 1:
                        # bottom right element
                        el = (i*N+j, i*N+j+1, (i+1)*N+j+1)
                        cells.append(el)
                        # top left element
                        el = (i*N+j, (i+1)*N+j+1, (i+1)*N+j)
                    else:
                        # bottom left element
                        el = (i*N+j, i*N+j+1, (i+1)*N+j)
                        cells.append(el)
                        # top right element
                        el = (i*N+j+1, (i+1)*N+j+1, (i+1)*N+j)
                cells.append(el)

    return mesh2d(np.array(vertices, dtype=REAL),
                  np.array(cells, dtype=INDEX))


def simpleSquare():
    return uniformSquare(2)


def crossSquare():
    return uniformSquare(3, crossed=True)


def gradedSquare(factor=0.6):
    from . meshCy import gradedHypercubeTransformer
    mesh = mesh2d(np.array([[0., 0.],
                            [1., 0.],
                            [0., 1.],
                            [1., 1.]], dtype=REAL),
                  np.array([[0, 1, 3],
                            [3, 2, 0]], dtype=INDEX))
    mesh.setMeshTransformation(gradedHypercubeTransformer(factor))
    mesh = mesh.refine()
    return mesh


def simpleInterval(a=0., b=1., numCells=1):
    vertices = np.zeros((numCells+1, 1), dtype=REAL)
    cells = np.zeros((numCells, 2), dtype=INDEX)
    for i in range(numCells):
        vertices[i, 0] = a+(b-a)*(i/numCells)
        cells[i, 0] = i
        cells[i, 1] = i+1
    vertices[-1, 0] = b
    return mesh1d(vertices, cells)


def disconnectedInterval(sep=0.1):
    vertices = np.array([(0, ),
                         (0.5-sep/2, ),
                         (0.5+sep/2, ),
                         (1., )], dtype=REAL)
    cells = np.array([(0, 1), (2, 3)], dtype=INDEX)
    return mesh1d(vertices, cells)


def getNodes(a, b, horizon, h, strictInteraction=True):
    diam = b-a
    k = INDEX(diam/h)
    if k*h < diam:
        k += 1
    nodes = np.linspace(a, b, k+1, dtype=REAL)
    hInterior = nodes[1]-nodes[0]
    k = INDEX(horizon/hInterior)
    if k*hInterior < horizon-1e-8:
        k += 1
    if not strictInteraction:
        horizon = k*hInterior
    nodes = np.hstack((np.linspace(a-horizon, a, k+1, dtype=REAL)[:-1],
                       nodes,
                       np.linspace(b, b+horizon, k+1, dtype=REAL)[1:]))
    return nodes


def intervalWithInteraction(a, b, horizon, h=None, strictInteraction=True):
    if h is None:
        h = horizon
    nodes = getNodes(a, b, horizon, h, strictInteraction)
    vertices = nodes[:, np.newaxis]
    num_vertices = vertices.shape[0]
    cells = uninitialized((num_vertices-1, 2), dtype=INDEX)
    cells[:, 0] = np.arange(0, num_vertices-1, dtype=INDEX)
    cells[:, 1] = np.arange(1, num_vertices, dtype=INDEX)
    return mesh1d(vertices, cells)


def doubleIntervalWithInteractions(a=0., b=1., c=2.,
                                   horizon1=0.1, horizon2=0.2,
                                   h=None):

    def getNumCells(l, r):
        eps = 1e-8
        return int(np.ceil((r-l-eps)/h))

    assert horizon2 >= horizon1
    assert horizon1 >= 0
    if h is None:
        if horizon1 > 0:
            h = horizon1
        elif horizon2 > 0:
            h = horizon2
        else:
            h = 0.5
    else:
        if horizon1 > 0:
            h = min([h, horizon1, horizon2])
        elif horizon2 > 0:
            h = min([h, horizon2])

    nodes = []
    if horizon1 > 0:
        nodes.append(a-horizon1)
    nodes.append(a)
    if horizon2 > 0:
        nodes.append(b-horizon2)
        if horizon1 != horizon2:
            nodes.append(b-horizon1)
    nodes.append(b)
    if horizon2 > 0:
        if horizon1 != horizon2:
            nodes.append(b+horizon1)
        nodes.append(b+horizon2)
    nodes.append(c)
    if horizon2 > 0:
        nodes.append(c+horizon2)
    vertices = []
    i = 0
    k = getNumCells(nodes[i], nodes[i+1])
    vertices.append(np.linspace(nodes[i], nodes[i+1], k+1))
    for i in range(1, len(nodes)-1):
        k = getNumCells(nodes[i], nodes[i+1])
        vertices.append(np.linspace(nodes[i], nodes[i+1], k+1)[1:])
    vertices = np.hstack(vertices)
    vertices = vertices[:, np.newaxis]
    num_vertices = vertices.shape[0]
    cells = uninitialized((num_vertices-1, 2), dtype=INDEX)
    cells[:, 0] = np.arange(0, num_vertices-1, dtype=INDEX)
    cells[:, 1] = np.arange(1, num_vertices, dtype=INDEX)
    return mesh1d(vertices, cells)


def squareWithInteractions(ax, ay, bx, by,
                           horizon,
                           h=None,
                           uniform=False,
                           strictInteraction=True,
                           innerRadius=-1,
                           preserveLinesHorizontal=[],
                           preserveLinesVertical=[],
                           **kwargs):
    if h is None:
        h = horizon
    if innerRadius > 0:
        uniform = False
    if not uniform:
        from . meshConstruction import (circularSegment,
                                        line,
                                        transformationRestriction)
        if h is None:
            h = horizon
        bottomLeft = np.array([ax, ay])
        center = np.array([(ax+bx)/2, (ay+by)/2])

        numPointsPerUnitLength = int(np.ceil(1/h))

        assert len(preserveLinesVertical) == 0 or len(preserveLinesHorizontal) == 0
        if len(preserveLinesVertical)+len(preserveLinesHorizontal) > 0:
            preserve = preserveLinesVertical+preserveLinesHorizontal

            c1 = circularSegment(bottomLeft, horizon, np.pi, 3/2*np.pi, numPointsPerUnitLength)

            x1 = preserve[0]
            c2 = line((ax, ay), (x1, ay))
            for k in range(len(preserve)-1):
                x1 = preserve[k]
                x2 = preserve[k+1]
                c2 = c2+line((x1, ay), (x2, ay))
            x2 = preserve[-1]
            c2 = c2+line((x2, ay), (bx, ay))
            c1 = c1 + (c2+(0., -horizon))
        else:
            c1 = circularSegment(bottomLeft, horizon, np.pi, 3/2*np.pi, numPointsPerUnitLength)
            c2 = line((ax, ay), (bx, ay))
            c1 = c1 + (c2+(0., -horizon))
        c3 = line((ax, ay), (ax, ay-horizon))
        c4 = line((ax, ay), (ax-horizon, ay))
        c = c1+c2+c3+c4

        frame = (c + (c*(center, np.pi/2)) + (c*(center, np.pi)) + (c*(center, -np.pi/2)))

        if len(preserveLinesVertical) > 0:
            d = line((0, ay-horizon), (0, ay))
            x1 = preserve[0]
            d = d + line((0, ay), (0, x1))
            for k in range(len(preserve)-1):
                x1 = preserve[k]
                x2 = preserve[k+1]
                d += line((0, x1), (0, x2))
            x2 = preserve[-1]
            d = d + line((0, x2), (0, by))
            d = d + line((0, by), (0, by+horizon))
            for x in preserveLinesVertical:
                assert ax <= x <= bx
                frame += (d+(x, 0.))
        if len(preserveLinesHorizontal) > 0:
            d = line((ax-horizon, 0), (ax, 0))+line((ax, 0), (bx, 0))+line((bx, 0), (bx+horizon, 0))

            d = line((ax-horizon, 0), (ax, 0))
            x1 = preserve[0]
            d = d + line((ax, 0), (x1, 0))
            for k in range(len(preserve)-1):
                x1 = preserve[k]
                x2 = preserve[k+1]
                d += line((x1, 0), (x2, 0))
            x2 = preserve[-1]
            d = d + line((x2, 0), (bx, 0))
            d = d + line((bx, 0), (bx+horizon, 0))

            for y in preserveLinesHorizontal:
                assert ay <= y <= by
                frame += (d+(0, y))

        if innerRadius > 0:
            frame += transformationRestriction(circularSegment(center, innerRadius, 0, 2*np.pi, numPointsPerUnitLength),
                                               center-(innerRadius, innerRadius),
                                               center+(innerRadius, innerRadius))
            mesh = frame.mesh(max_volume=h**2, min_angle=30, **kwargs)
        else:
            frame.holes.append(center)
            mesh = frame.mesh(max_volume=h**2, min_angle=30, **kwargs)

            eps = 1e-10
            N1 = np.logical_and(np.absolute(mesh.vertices_as_array[:, 0]-ax) < eps,
                                np.logical_and(mesh.vertices_as_array[:, 1] >= ay-eps,
                                               mesh.vertices_as_array[:, 1] <= by+eps)).sum()
            N2 = np.logical_and(np.absolute(mesh.vertices_as_array[:, 0]-bx) < eps,
                                np.logical_and(mesh.vertices_as_array[:, 1] >= ay-eps,
                                               mesh.vertices_as_array[:, 1] <= by+eps)).sum()
            M1 = np.logical_and(np.absolute(mesh.vertices_as_array[:, 1]-ay) < eps,
                                np.logical_and(mesh.vertices_as_array[:, 0] >= ax-eps,
                                               mesh.vertices_as_array[:, 0] <= bx+eps)).sum()
            M2 = np.logical_and(np.absolute(mesh.vertices_as_array[:, 1]-by) < eps,
                                np.logical_and(mesh.vertices_as_array[:, 0] >= ax-eps,
                                               mesh.vertices_as_array[:, 0] <= bx+eps)).sum()
            assert N1 == N2, (N1, N2)
            assert M1 == M2, (M1, M2)
            mesh2 = uniformSquare(N=N1, M=M1, ax=ax, ay=ay, bx=bx, by=by)
            mesh = snapMeshes(mesh, mesh2)

        location = uninitialized((mesh.num_vertices), dtype=INDEX)
        eps = 1e-9
        for x in preserveLinesVertical:
            for vertexNo in range(mesh.num_vertices):
                if mesh.vertices[vertexNo, 0] < x-eps:
                    location[vertexNo] = 0
                elif mesh.vertices[vertexNo, 0] > x+eps:
                    location[vertexNo] = 2
                else:
                    location[vertexNo] = 1
            for cellNo in range(mesh.num_cells):
                cellLoc = set()
                for vertexNo in range(mesh.dim+1):
                    cellLoc.add(location[mesh.cells[cellNo, vertexNo]])
                assert max(cellLoc)-min(cellLoc) <= 1, (mesh.vertices_as_array[mesh.cells_as_array[cellNo, :], :], cellLoc)
        for y in preserveLinesHorizontal:
            for vertexNo in range(mesh.num_vertices):
                if mesh.vertices[vertexNo, 1] < y-eps:
                    location[vertexNo] = 0
                elif mesh.vertices[vertexNo, 1] > y+eps:
                    location[vertexNo] = 2
                else:
                    location[vertexNo] = 1
            for cellNo in range(mesh.num_cells):
                cellLoc = set()
                for vertexNo in range(mesh.dim+1):
                    cellLoc.add(location[mesh.cells[cellNo, vertexNo]])
                assert max(cellLoc)-min(cellLoc) <= 1, mesh.vertices_as_array[mesh.cells_as_array[cellNo, :], :]
    else:
        x = getNodes(ax, bx, horizon, h, strictInteraction)
        y = getNodes(ay, by, horizon, h, strictInteraction)
        M = x.shape[0]
        N = y.shape[0]
        vertices = []
        for i in range(M):
            for j in range(N):
                vertices.append((x[i], y[j]))
        cells = []
        for i in range(M-1):
            for j in range(N-1):
                # bottom right element
                el = (i*N+j, i*N+j+1, (i+1)*N+j+1)
                cells.append(el)
                # top left element
                el = (i*N+j, (i+1)*N+j+1, (i+1)*N+j)
                cells.append(el)
        mesh = mesh2d(np.array(vertices, dtype=REAL),
                      np.array(cells, dtype=INDEX))
    return mesh


def doubleSquareWithInteractions(ax=0., ay=0., bx=1., by=1., cx=2., cy=1.,
                                 horizon1=0.1, horizon2=0.2,
                                 h=None,
                                 returnSketch=False,
                                 **kwargs):
    from . meshConstruction import (circularSegment,
                                    line,
                                    polygon,
                                    transformationRestriction)
    assert horizon2 >= horizon1
    assert horizon1 >= 0
    if h is None:
        if horizon1 > 0:
            h = horizon1
        elif horizon2 > 0:
            h = horizon2
        else:
            h = 0.5
    else:
        if horizon1 > 0:
            h = min([h, horizon1, horizon2])
        elif horizon2 > 0:
            h = min([h, horizon2])

    bottomLeft = np.array([ax, ay])
    bottomMid = np.array([bx, ay])
    bottomRight = np.array([cx, ay])
    topLeft = np.array([ax, by])
    topMid = np.array([bx, by])
    topRight = np.array([cx, by])

    centerLeft = np.array([(ax+bx)/2, (ay+by)/2])
    centerRight = np.array([(bx+cx)/2, (ay+cy)/2])

    for k in range(10):
        numPointsPerUnitLength = int(np.ceil(1/(h*0.8**(k/2))))

        if horizon2 > 0:
            magicAngle = 0.5*np.pi-np.arcsin(horizon1/horizon2)
            magicLen = horizon2*np.cos(0.5*np.pi-magicAngle)

            # the four/six inner squares
            inner = polygon([bottomLeft, bottomMid-(horizon2, 0),
                             topMid-(horizon2, 0), topLeft], num_points_per_unit_len=numPointsPerUnitLength)
            if horizon1 < horizon2:
                inner += polygon([bottomMid-(horizon2, 0), bottomMid-(horizon1, 0),
                                  topMid-(horizon1, 0), topMid-(horizon2, 0)], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)
                inner += polygon([bottomMid-(horizon1, 0), bottomMid,
                                  topMid, topMid-(horizon1, 0)], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)
                inner += polygon([bottomMid, bottomMid+(horizon1, 0),
                                  topMid+(horizon1, 0), topMid], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)
                inner += polygon([bottomMid+(horizon1, 0), bottomMid+(horizon2, 0),
                                  topMid+(horizon2, 0), topMid+(horizon1, 0)], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)
            else:
                inner += polygon([bottomMid-(horizon2, 0), bottomMid,
                                  topMid, topMid-(horizon2, 0)], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)
                inner += polygon([bottomMid, bottomMid+(horizon2, 0),
                                  topMid+(horizon2, 0), topMid], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)
            inner += polygon([bottomMid+(horizon2, 0), bottomRight,
                              topRight, topMid+(horizon2, 0)], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)
        else:
            inner = polygon([bottomLeft, bottomMid, topMid, topLeft], num_points_per_unit_len=numPointsPerUnitLength)
            inner += polygon([bottomMid, bottomRight, topRight, topMid], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)

            mesh = inner.mesh(h=h*0.8**(k/2), **kwargs)
            frame = inner

        if horizon2 > 0:
            # interaction domain for right domain
            d1 = (line(bottomMid, bottomRight)+(0, -horizon2) + circularSegment(bottomRight, horizon2, 1.5*np.pi, 2*np.pi, numPointsPerUnitLength))
            d2 = (line(bottomRight, topRight)+(horizon2, 0) + circularSegment(topRight, horizon2, 0, 0.5*np.pi, numPointsPerUnitLength))
            d3 = ((line(topRight, topMid)+(0, horizon2)) +
                  transformationRestriction(circularSegment(topMid, horizon2, 0.5*np.pi, 0.5*np.pi+magicAngle, numPointsPerUnitLength),
                                            topMid+(-horizon2, horizon1+1e-9),
                                            topMid+(0, horizon2)) +
                  transformationRestriction(circularSegment(topMid, horizon2, 0.5*np.pi + magicAngle, np.pi, numPointsPerUnitLength),
                                            topMid+(-horizon2, 0),
                                            topMid+(-magicLen-1e-9, horizon1)))
            d4 = (transformationRestriction(circularSegment(bottomMid, horizon2, np.pi, np.pi + (0.5*np.pi-magicAngle), numPointsPerUnitLength),
                                            bottomMid+(-horizon2, -horizon1+1e-9),
                                            bottomMid+(-magicLen, 0)) +
                  transformationRestriction(circularSegment(bottomMid, horizon2, np.pi + (0.5*np.pi-magicAngle), 1.5*np.pi, numPointsPerUnitLength),
                                            bottomMid+(-horizon2, -horizon2),
                                            bottomMid+(0, -horizon1-1e-9)))
            outer = d1+d2+d3+d4

            # two right corners
            c6 = line(bottomRight, bottomRight-(0, horizon2)) + line(bottomRight, bottomRight+(horizon2, 0))
            c6 = c6 + (c6*(centerRight, 0.5*np.pi))
            outer += c6

            # the two mid corners
            c7 = line(topMid+(0, horizon2), topMid+(0, horizon1)) + line(topMid+(0, horizon1), topMid)
            c8 = line(bottomMid, bottomMid-(0, horizon1)) + line(bottomMid-(0, horizon1), bottomMid-(0, horizon2))
            outer += c7+c8

            if horizon1 > 0:
                # interaction domain for left domain
                e1 = circularSegment(topMid, horizon1, 0, 0.5*np.pi, num_points_per_unit_len=numPointsPerUnitLength)
                e2 = (line(topMid, topMid-(magicLen, 0)) + (0, horizon1)) + (line(topMid-(magicLen, 0), topLeft) + (0, horizon1))
                e3 = circularSegment(topLeft, horizon1, 0.5*np.pi, np.pi, num_points_per_unit_len=numPointsPerUnitLength)
                e4 = line(topLeft, bottomLeft)+(-horizon1, 0)
                e5 = circularSegment(bottomLeft, horizon1, np.pi, 1.5*np.pi, num_points_per_unit_len=numPointsPerUnitLength)
                e6 = (line(bottomLeft, bottomMid-(magicLen, 0))+(0, -horizon1)) + (line(bottomMid-(magicLen, 0), bottomMid)+(0, -horizon1))
                e7 = circularSegment(bottomMid, horizon1, 1.5*np.pi, 2*np.pi, num_points_per_unit_len=numPointsPerUnitLength)
                outer += e1+e2+e3+e4+e5+e6+e7

            # preserve right angles near corners
            if horizon1 > 0:
                # two left corners
                c5 = line(topLeft, topLeft+(0, horizon1))+line(topLeft, topLeft-(horizon1, 0))
                c5 = c5 + (c5*(centerLeft, 0.5*np.pi))
                outer += c5

            frame = inner+outer
            mesh = frame.mesh(h=h*0.8**(k/2), **kwargs)

        if mesh.h <= h:
            if returnSketch:
                return mesh, frame
            else:
                return mesh
    if returnSketch:
        return mesh, frame
    else:
        return mesh


def doubleSquareWithInteractionsCorners(ax=0., ay=0., bx=1., by=1., cx=2., cy=1.,
                                        horizon1=0.1, horizon2=0.2,
                                        h=None,
                                        returnSketch=False,
                                        **kwargs):
    from PyNucleus.fem.meshConstruction import (line,
                                                polygon)
    assert horizon2 >= horizon1
    assert horizon1 >= 0
    if h is None:
        if horizon1 > 0:
            h = horizon1
        elif horizon2 > 0:
            h = horizon2
        else:
            h = 0.5
    else:
        if horizon1 > 0:
            h = min([h, horizon1, horizon2])
        elif horizon2 > 0:
            h = min([h, horizon2])

    bottomLeft = np.array([ax, ay])
    bottomMid = np.array([bx, ay])
    bottomRight = np.array([cx, ay])
    topLeft = np.array([ax, by])
    topMid = np.array([bx, by])
    topRight = np.array([cx, by])

    centerLeft = np.array([(ax+bx)/2, (ay+by)/2])
    centerRight = np.array([(bx+cx)/2, (ay+cy)/2])

    for k in range(10):
        numPointsPerUnitLength = int(np.ceil(1/(h*0.8**(k/2))))

        if horizon2 > 0:

            # the four/six inner squares
            inner = polygon([bottomLeft, bottomMid-(horizon2, 0),
                             topMid-(horizon2, 0), topLeft], num_points_per_unit_len=numPointsPerUnitLength)
            if horizon1 < horizon2:
                inner += polygon([bottomMid-(horizon2, 0), bottomMid-(horizon1, 0),
                                  topMid-(horizon1, 0), topMid-(horizon2, 0)], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)
                inner += polygon([bottomMid-(horizon1, 0), bottomMid,
                                  topMid, topMid-(horizon1, 0)], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)
                inner += polygon([bottomMid, bottomMid+(horizon1, 0),
                                  topMid+(horizon1, 0), topMid], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)
                inner += polygon([bottomMid+(horizon1, 0), bottomMid+(horizon2, 0),
                                  topMid+(horizon2, 0), topMid+(horizon1, 0)], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)
            else:
                inner += polygon([bottomMid-(horizon2, 0), bottomMid,
                                  topMid, topMid-(horizon2, 0)], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)
                inner += polygon([bottomMid, bottomMid+(horizon2, 0),
                                  topMid+(horizon2, 0), topMid], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)
            inner += polygon([bottomMid+(horizon2, 0), bottomRight,
                              topRight, topMid+(horizon2, 0)], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)
        else:
            inner = polygon([bottomLeft, bottomMid, topMid, topLeft], num_points_per_unit_len=numPointsPerUnitLength)
            inner += polygon([bottomMid, bottomRight, topRight, topMid], doClose=False, num_points_per_unit_len=numPointsPerUnitLength)

            mesh = inner.mesh(h=h*0.8**(k/2), **kwargs)
            frame = inner

        if horizon2 > 0:
            # interaction domain for right domain

            outer = polygon([np.array([bx-horizon2, ay]),
                             np.array([bx-horizon2, ay-horizon1]),
                             np.array([bx-horizon2, ay-horizon2]),
                             np.array([cx, ay-horizon2]),
                             np.array([cx+horizon2, ay-horizon2]),
                             np.array([cx+horizon2, ay]),
                             np.array([cx+horizon2, cy]),
                             np.array([cx+horizon2, cy+horizon2]),
                             np.array([cx, cy+horizon2]),
                             np.array([bx-horizon2, by+horizon2]),
                             np.array([bx-horizon2, by+horizon1]),
                             np.array([bx-horizon2, by])],
                            doClose=False)
            # two right corners
            c6 = line(bottomRight, bottomRight-(0, horizon2)) + line(bottomRight, bottomRight+(horizon2, 0))
            c6 = c6 + (c6*(centerRight, 0.5*np.pi))
            outer += c6

            if horizon1 > 0:
                # interaction domain for left domain
                outer += polygon([np.array([bx+horizon1, by+horizon1]),
                                  np.array([bx-horizon2, by+horizon1]),
                                  np.array([ax, by+horizon1]),
                                  np.array([ax-horizon1, by+horizon1]),
                                  np.array([ax-horizon1, by]),
                                  np.array([ax-horizon1, ay]),
                                  np.array([ax-horizon1, ay-horizon1]),
                                  np.array([ax, ay-horizon1]),
                                  np.array([bx-horizon2, ay-horizon1]),
                                  np.array([bx+horizon1, ay-horizon1])])

            # preserve right angles near corners
            if horizon1 > 0:
                # two left corners
                c5 = line(topLeft, topLeft+(0, horizon1))+line(topLeft, topLeft-(horizon1, 0))
                c5 = c5 + (c5*(centerLeft, 0.5*np.pi))
                outer += c5

            frame = inner+outer
            mesh = frame.mesh(h=h*0.8**(k/2), **kwargs)

        if mesh.h <= h:
            if returnSketch:
                return mesh, frame
            else:
                return mesh
    if returnSketch:
        return mesh, frame
    else:
        return mesh


def discWithInteraction(radius, horizon, h=0.25, max_volume=None, projectNodeToOrigin=True):
    if max_volume is None:
        max_volume = h**2
    n = int(np.around(2*np.pi*radius/h))
    if horizon > 0:
        outerRadius = radius + horizon
        if h > horizon:
            LOGGER.warn("h = {} > horizon = {}. Using h=horizon instead.".format(h, horizon))
            h = horizon
        return circleWithInnerRadius(n,
                                     radius=outerRadius,
                                     innerRadius=radius,
                                     max_volume=max_volume)
    else:
        return circle(n,
                      radius=radius,
                      max_volume=max_volume,
                      projectNodeToOrigin=projectNodeToOrigin)


def gradedDiscWithInteraction(radius, horizon, mu=2., h=0.25, max_volume=None, projectNodeToOrigin=True):
    if max_volume is None:
        max_volume = h**2
    n = int(np.around(2*np.pi*radius/h))
    if horizon > 0:
        raise NotImplementedError()
    else:
        return graded_circle(n,
                             mu=mu,
                             radius=radius,
                             max_volume=max_volume)


def discWithIslands(horizon=0., radius=1., islandOffCenter=0.35, islandDiam=0.5):
    from . meshConstruction import circle, rectangle
    numPointsPerLength = 4
    assert islandOffCenter > islandDiam/2
    assert np.sqrt(2)*(islandOffCenter+islandDiam/2) < radius
    assert horizon >= 0.
    c = circle((0, 0), radius, num_points_per_unit_len=numPointsPerLength)
    if horizon > 0:
        c += circle((0, 0), radius+horizon, num_points_per_unit_len=numPointsPerLength)
    island = rectangle((-islandDiam/2, -islandDiam/2), (islandDiam/2, islandDiam/2))
    c += (island+(islandOffCenter, islandOffCenter))
    c += (island+(-islandOffCenter, islandOffCenter))
    c += (island+(islandOffCenter, -islandOffCenter))
    c += (island+(-islandOffCenter, -islandOffCenter))
    mesh = c.mesh(min_angle=30)
    return mesh


def simpleBox():
    vertices = np.array([(0, 0, 0),
                         (1, 0, 0),
                         (1, 1, 0),
                         (0, 1, 0),
                         (0, 0, 1),
                         (1, 0, 1),
                         (1, 1, 1),
                         (0, 1, 1)], dtype=REAL)
    cells = np.array([(0, 1, 6, 5),
                      (0, 1, 2, 6),
                      (0, 4, 5, 6),
                      (0, 4, 6, 7),
                      (0, 2, 3, 6),
                      (0, 3, 7, 6)], dtype=INDEX)
    return mesh3d(vertices, cells)


def box(ax=0., ay=0., az=0., bx=1., by=1., bz=1., Nx=2, Ny=2, Nz=2):
    x = np.linspace(ax, bx, Nx)
    y = np.linspace(ay, by, Ny)
    z = np.linspace(az, bz, Nz)

    vertices = []
    for kz in range(Nz):
        for ky in range(Ny):
            for kx in range(Nx):
                vertices.append(np.array([x[kx], y[ky], z[kz]]))

    def getVertexNo(kx, ky, kz):
        return Ny*Nx*kz + Nx*ky + kx

    def boxCells(a, b, c, d, e, f, g, h):
        return [(a, b, g, f),
                (a, b, c, g),
                (a, e, f, g),
                (a, e, g, h),
                (a, c, d, g),
                (a, d, h, g)]

    cells = []
    for kz in range(Nz-1):
        for ky in range(Ny-1):
            for kx in range(Nx-1):
                a = getVertexNo(kx, ky, kz)
                b = getVertexNo(kx+1, ky, kz)
                c = getVertexNo(kx+1, ky+1, kz)
                d = getVertexNo(kx, ky+1, kz)
                e = getVertexNo(kx, ky, kz+1)
                f = getVertexNo(kx+1, ky, kz+1)
                g = getVertexNo(kx+1, ky+1, kz+1)
                h = getVertexNo(kx, ky+1, kz+1)

                cells += boxCells(a, b, c, d, e, f, g, h)
    return mesh3d(np.array(vertices, dtype=REAL),
                  np.array(cells, dtype=INDEX))


def boxWithInteractions(horizon, ax=0., ay=0., az=0., bx=1., by=1., bz=1., Nx=2, Ny=2, Nz=2):
    Nx2 = max(int(np.ceil((bx-ax+2*horizon)/horizon))+1, int(np.ceil((bx-ax+2*horizon)/(bx-ax)*Nx)))
    Ny2 = max(int(np.ceil((by-ay+2*horizon)/horizon))+1, int(np.ceil((by-ay+2*horizon)/(by-ay)*Nx)))
    Nz2 = max(int(np.ceil((bz-az+2*horizon)/horizon))+1, int(np.ceil((bz-az+2*horizon)/(bz-az)*Nx)))
    return box(ax-horizon, ay-horizon, az-horizon,
               bx+horizon, by+horizon, bz+horizon,
               Nx2, Ny2, Nz2)


def gradedBox(factor=0.6):
    from . meshCy import gradedHypercubeTransformer
    mesh = simpleBox()
    mesh.setMeshTransformation(gradedHypercubeTransformer(factor))
    mesh = mesh.refine()
    return mesh


def standardSimplex(d):
    vertices = np.zeros((d+1, d), dtype=REAL)
    cells = np.zeros((1, d+1), dtype=INDEX)
    for i in range(d):
        vertices[i+1, i] = 1.
        cells[0, i+1] = i+1
    if d == 1:
        return mesh1d(vertices, cells)
    elif d == 2:
        return mesh2d(vertices, cells)
    elif d == 3:
        return mesh3d(vertices, cells)
    else:
        raise NotImplementedError()


def standardSimplex2D():
    return standardSimplex(2)


def standardSimplex3D():
    return standardSimplex(3)


def simpleFicheraCube():
    vertices = np.array([(0, 0, 0),
                         (1, 0, 0),
                         (1, 1, 0),
                         (0, 1, 0),
                         (0, 0, 1),
                         (1, 0, 1),
                         (1, 1, 1),
                         (0, 1, 1),
                         #
                         (2, 0, 0),
                         (2, 1, 0),
                         (2, 0, 1),
                         (2, 1, 1),
                         #
                         (0, 0, 2),
                         (1, 0, 2),
                         (1, 1, 2),
                         (0, 1, 2),
                         #
                         (0, 2, 0),
                         (1, 2, 0),
                         (2, 2, 0),
                         (2, 2, 1),
                         (1, 2, 1),
                         (0, 2, 1),
                         (2, 2, 2),
                         (1, 2, 2),
                         (0, 2, 2),
                         (2, 1, 2)], dtype=REAL)

    def boxCells(a, b, c, d, e, f, g, h):
        return np.array([(a, b, g, f),
                         (a, b, c, g),
                         (a, e, f, g),
                         (a, e, g, h),
                         (a, c, d, g),
                         (a, d, h, g)], dtype=INDEX)

    cells = np.vstack((boxCells(0, 1, 2, 3, 4, 5, 6, 7),
                       boxCells(1, 8, 9, 2, 5, 10, 11, 6),
                       boxCells(4, 5, 6, 7, 12, 13, 14, 15),
                       boxCells(3, 2, 17, 16, 7, 6, 20, 21),
                       boxCells(2, 9, 18, 17, 6, 11, 19, 20),
                       boxCells(7, 6, 20, 21, 15, 14, 23, 24),
                       boxCells(6, 11, 19, 20, 14, 25, 22, 23)))
    return mesh3d(vertices, cells)


def simpleLshape():
    vertices = np.array([(0, 0),  # 0
                         (1, 0),  # 1
                         (2, 0),  # 2
                         (2, 1),  # 3
                         (1, 1),  # 4
                         (0, 1),  # 5
                         (0, 2),  # 6
                         (1, 2)], dtype=REAL)  # 7

    cells = np.array([(0, 1, 4), (0, 4, 5), (1, 2, 3),
                      (1, 3, 4), (5, 4, 7), (5, 7, 6)], dtype=INDEX)
    return mesh2d(vertices, cells)


def disconnectedDomain(sep=0.1):
    vertices = np.array([(0, 0),
                         (1, 0),
                         (1, 0.5-sep/2),
                         (0, 0.5-sep/2),
                         (0, 0.5+sep/2),
                         (1, 0.5+sep/2),
                         (1, 1),
                         (0, 1)], dtype=REAL)

    cells = np.array([(0, 1, 2), (0, 2, 3),
                      (4, 5, 6), (4, 6, 7)], dtype=INDEX)
    return mesh2d(vertices, cells)


def Lshape(d):
    from mshr import Rectangle, generate_mesh
    from dolfin import Point
    domain = (Rectangle(Point(0, 0), Point(2, 2))
              - Rectangle(Point(1, 1), Point(2, 2)))
    mesh = generate_mesh(domain, d)
    vertices = [x for x in mesh.coordinates()]
    cells = mesh.cells()
    return mesh2d(vertices, cells)


def circle(n, radius=1., returnFacets=False, projectNodeToOrigin=True, **kwargs):
    from meshpy.triangle import MeshInfo, build
    mesh_info = MeshInfo()

    if 'min_angle' not in kwargs:
        kwargs['min_angle'] = 30

    points = []
    facets = []
    for i in range(n):
        points.append((radius*np.cos(i*2*np.pi/n), radius*np.sin(i*2*np.pi/n)))
    for i in range(1, n):
        facets.append((i-1, i))
    facets.append((n-1, 0))

    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    mesh_meshpy = build(mesh_info, **kwargs)
    mesh = mesh2d(np.array(mesh_meshpy.points, dtype=REAL),
                  np.array(mesh_meshpy.elements, dtype=INDEX))
    if projectNodeToOrigin:
        # Make sure that one node is on the origin.
        # Otherwise the radialMeshTransformation does weird stuff
        k = np.linalg.norm(mesh.vertices_as_array, axis=1).argmin()
        mesh.vertices[k, :] = 0.
        mesh.resetMeshInfo()
        assert mesh.delta < 10., (mesh, mesh.hmin, mesh.h, mesh.delta)
    from . meshCy import radialMeshTransformer
    mesh.setMeshTransformation(radialMeshTransformer())
    if returnFacets:
        return mesh, np.array(points), np.array(facets)
    else:
        return mesh


def circleWithInnerRadius(n, radius=2., innerRadius=1., returnFacets=False, **kwargs):
    from meshpy.triangle import MeshInfo, build
    mesh_info = MeshInfo()

    if 'min_angle' not in kwargs:
        kwargs['min_angle'] = 30

    points = []
    facets = []
    for i in range(n):
        points.append((radius*np.cos(i*2*np.pi/n),
                       radius*np.sin(i*2*np.pi/n)))
    for i in range(1, n):
        facets.append((i-1, i))
    facets.append((n-1, 0))

    nInner = int(round(n*innerRadius/radius))

    for i in range(nInner):
        points.append((innerRadius*np.cos(i*2*np.pi/nInner),
                       innerRadius*np.sin(i*2*np.pi/nInner)))
    for i in range(1, nInner):
        facets.append((n+i-1, n+i))
    facets.append((n-1+nInner, n))

    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    mesh_meshpy = build(mesh_info, **kwargs)
    mesh = mesh2d(np.array(mesh_meshpy.points, dtype=REAL),
                  np.array(mesh_meshpy.elements, dtype=INDEX))
    # Make sure that one node is on the origin.
    # Otherwise the radialMeshTransformation does weird stuff
    k = np.linalg.norm(mesh.vertices_as_array, axis=1).argmin()
    mesh.vertices[k, :] = 0.
    mesh.resetMeshInfo()
    assert mesh.delta < 10.
    from . meshCy import radialMeshTransformer
    mesh.setMeshTransformation(radialMeshTransformer())
    if returnFacets:
        return mesh, np.array(points), np.array(facets)
    else:
        return mesh


def squareWithCircularCutout(ax=-3., ay=-3., bx=3., by=3., radius=1., num_points_per_unit_len=2):
    from . meshConstruction import polygon, circle
    square = polygon([(ax, ay), (bx, ay), (bx, by), (ax, by)])
    frame = square+circle((0, 0), radius, num_points_per_unit_len=num_points_per_unit_len)
    frame.holes.append((0, 0))
    return frame.mesh()


def boxWithBallCutout(ax=-3., ay=-3., az=-3., bx=3., by=3., bz=3.,
                      radius=1., points=4, radial_subdiv=None, **kwargs):
    from meshpy.tet import MeshInfo, build  # Options
    from meshpy.geometry import generate_surface_of_revolution, EXT_OPEN, GeometryBuilder, make_box

    if radial_subdiv is None:
        radial_subdiv = 2*points+2

    dphi = np.pi/points

    def truncate(r):
        if abs(r) < 1e-10:
            return 0
        else:
            return r

    rz = [(truncate(radius*np.sin(i*dphi)), radius*np.cos(i*dphi)) for i in range(points+1)]

    geob = GeometryBuilder()
    geob.add_geometry(*generate_surface_of_revolution(rz,
                                                      closure=EXT_OPEN,
                                                      radial_subdiv=radial_subdiv))
    points, facets, _, facet_markers = make_box((ax, ay, az), (bx, by, bz))
    geob.add_geometry(points, facets, facet_markers=facet_markers)
    mesh_info = MeshInfo()
    geob.set(mesh_info)
    mesh_info.set_holes([(0., 0., 0.)])
    mesh_meshpy = build(mesh_info, **kwargs)  # , options=Options(switches='pq1.2/10')
    mesh = mesh3d(np.array(mesh_meshpy.points, dtype=REAL),
                  np.array(mesh_meshpy.elements, dtype=INDEX))
    from PyNucleus_fem.meshCy import radialMeshTransformer
    mesh.setMeshTransformation(radialMeshTransformer(radius))
    return mesh


def gradedIntervals(intervals, h):

    intervals = list(sorted(intervals, key=lambda int: int[0]))

    Ms = np.zeros((2*len(intervals)), dtype=INDEX)
    for intNo, interval in enumerate(intervals):
        mu1 = interval[2]
        mu2 = interval[3]
        if mu1 is None:
            if mu2 is None:
                raise NotImplementedError()
            else:
                radius = interval[1]-interval[0]
                Ms[2*intNo] = 0
                Ms[2*intNo+1] = max(int(np.ceil(1/(1-(1-h/radius)**(1/mu2)))), 1)
        else:
            if mu2 is None:
                radius = interval[1]-interval[0]
                Ms[2*intNo] = max(int(np.ceil(1/(1-(1-h/radius)**(1/mu1)))), 1)
                Ms[2*intNo+1] = 0
            else:
                radius = 0.5*(interval[1]-interval[0])
                Ms[2*intNo] = max(int(np.ceil(1/(1-(1-h/radius)**(1/mu1)))), 1)
                Ms[2*intNo+1] = max(int(np.ceil(1/(1-(1-h/radius)**(1/mu2)))), 1)
    points = np.zeros((Ms.sum()+1, 1), dtype=REAL)

    for intNo, interval in enumerate(intervals):
        mu1 = interval[2]
        mu2 = interval[3]
        M1 = Ms[2*intNo]
        M2 = Ms[2*intNo+1]
        if M1 > 0 and M2 > 0:
            radius = 0.5*(interval[1]-interval[0])
            center = 0.5*(interval[0]+interval[1])
        else:
            radius = interval[1]-interval[0]
            if M1 == 0:
                center = interval[0]
            else:
                center = interval[1]

        indexCenter = Ms[:2*intNo+1].sum()
        points[indexCenter, 0] = center
        M = Ms[2*intNo]
        for j in range(1, M+1):
            points[indexCenter-j, 0] = center - radius*(1 - (1-j/M)**mu1)
        M = Ms[2*intNo+1]
        for j in range(1, M+1):
            points[indexCenter+j, 0] = center + radius*(1 - (1-j/M)**mu2)

    cells = np.empty((Ms.sum(), 2), dtype=INDEX)
    cells[:, 0] = np.arange(0, Ms.sum(), dtype=INDEX)
    cells[:, 1] = np.arange(1, Ms.sum()+1, dtype=INDEX)

    mesh = mesh1d(points, cells)
    from . meshCy import multiIntervalMeshTransformer
    mesh.setMeshTransformation(multiIntervalMeshTransformer(intervals))
    return mesh


def graded_interval(h, mu=2., mu2=None, radius=1.):
    if mu2 is None:
        mu2 = mu
    intervals = [(-radius, radius, mu, mu2)]
    return gradedIntervals(intervals, h)


def double_graded_interval(h, mu_ll=2., mu_rr=2., mu_lr=None, mu_rl=None, a=-1., b=1.):
    if mu_lr is None:
        mu_lr = mu_ll
    if mu_rl is None:
        mu_rl = mu_rr
    intervals = [(a, 0., mu_ll, mu_lr), (0., b, mu_rl, mu_rr)]
    return gradedIntervals(intervals, h)


def double_graded_interval_with_interaction(horizon, h=None, mu_ll=2., mu_rr=2., mu_lr=None, mu_rl=None, a=-1., b=1.):
    if h is None:
        h = horizon/2
    else:
        h = min(horizon/2, h)
    if mu_lr is None:
        mu_lr = mu_ll
    if mu_rl is None:
        mu_rl = mu_rr
    intervals = [(a-horizon, a, None, mu_ll), (a, 0., mu_ll, mu_lr), (0., b, mu_rl, mu_rr), (b, b+horizon, mu_rr, None)]
    return gradedIntervals(intervals, h)


def graded_circle(M, mu=2., radius=1., returnFacets=False, **kwargs):
    from meshpy.triangle import MeshInfo, build
    mesh_info = MeshInfo()

    points = []
    facets = []

    points.append((0, 0))
    rold = 0
    for j in range(1, M+1):
        rj = radius*(1 - (1-j/M)**mu)
        hj = rj-rold
        n = int(np.floor(2*np.pi*rj/hj))
        for i in range(n):
            points.append((rj*np.cos(i*2*np.pi/n), rj*np.sin(i*2*np.pi/n)))
        rold = rj
        for i in range(len(points)-n+1, len(points)):
            facets.append((i-1, i))
        facets.append((len(points)-1, len(points)-n))

    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    mesh_meshpy = build(mesh_info, **kwargs)
    mesh = mesh2d(np.array(mesh_meshpy.points, dtype=REAL),
                  np.array(mesh_meshpy.elements, dtype=INDEX))
    if returnFacets:
        return mesh, np.array(points), np.array(facets)
    else:
        return mesh


def double_graded_circle(M,
                         muInterior=2., muExterior=2.,
                         rInterior=1., rExterior=2.,
                         returnFacets=False, **kwargs):
    from meshpy.triangle import MeshInfo, build
    mesh_info = MeshInfo()

    points = []
    facets = []

    points.append((0, 0))
    rold = 0
    for j in range(1, M+1):
        rj = rInterior*(1 - (1-j/M)**muInterior)
        # print(rj)
        hj = rj-rold
        n = int(np.floor(2*np.pi*rj/hj))
        for i in range(n):
            points.append((rj*np.cos(i*2*np.pi/n), rj*np.sin(i*2*np.pi/n)))
        rold = rj
        for i in range(len(points)-n+1, len(points)):
            facets.append((i-1, i))
        facets.append((len(points)-1, len(points)-n))

    # rold = rInterior
    # M = int(((rExterior-rInterior)/rInterior)*M)
    for j in range(1, M+1):
        rj = rInterior + (rExterior-rInterior)*(j/M)**muExterior
        # print(rj)
        hj = rj-rold
        n = int(np.floor(2*np.pi*rj/hj))
        for i in range(n):
            points.append((rj*np.cos(i*2*np.pi/n), rj*np.sin(i*2*np.pi/n)))
        rold = rj
        for i in range(len(points)-n+1, len(points)):
            facets.append((i-1, i))
        facets.append((len(points)-1, len(points)-n))

    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    mesh_meshpy = build(mesh_info, **kwargs)
    mesh = mesh2d(np.array(mesh_meshpy.points, dtype=REAL),
                  np.array(mesh_meshpy.elements, dtype=INDEX))
    if returnFacets:
        return mesh, np.array(points), np.array(facets)
    else:
        return mesh


def cutoutCircle(n, radius=1., cutoutAngle=np.pi/2.,
                 returnFacets=False, minAngle=30, **kwargs):
    from meshpy.triangle import MeshInfo, build
    mesh_info = MeshInfo()
    n = n-1

    points = [(0., 0.)]
    facets = []
    for i in range(n+1):
        points.append((radius*np.cos(i*(2*np.pi-cutoutAngle)/n),
                       radius*np.sin(i*(2*np.pi-cutoutAngle)/n)))
    for i in range(1, n+2):
        facets.append((i-1, i))
    facets.append((n+1, 0))

    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    mesh_meshpy = build(mesh_info, min_angle=minAngle, **kwargs)
    mesh = mesh2d(np.array(mesh_meshpy.points, dtype=REAL),
                  np.array(mesh_meshpy.elements, dtype=INDEX))
    if returnFacets:
        return mesh, np.array(points), np.array(facets)
    else:
        return mesh


def twinDisc(n, radius=1., sep=0.1, **kwargs):
    from . meshConstruction import circle
    return (circle((sep/2+radius, 0), radius, num_points=n+1) +
            circle((-sep/2-radius, 0), radius, num_points=n+1)).mesh()


def dumbbell(n=8, radius=1., barAngle=np.pi/4, barLength=3,
             returnFacets=False, minAngle=30, **kwargs):
    from meshpy.triangle import MeshInfo, build
    mesh_info = MeshInfo()

    points = []
    facets = []
    for i in range(n):
        points.append((-barLength/2 +
                       radius*np.cos(barAngle/2+i*(2*np.pi-barAngle)/(n-1)),
                       radius*np.sin(barAngle/2+i*(2*np.pi-barAngle)/(n-1))))
    for i in range(n):
        points.append((barLength/2 +
                       radius*np.cos(np.pi+barAngle/2+i*(2*np.pi-barAngle)/(n-1)),
                       radius*np.sin(np.pi+barAngle/2+i*(2*np.pi-barAngle)/(n-1))))
    for i in range(1, 2*n):
        facets.append((i-1, i))
    facets.append((2*n-1, 0))

    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    mesh_meshpy = build(mesh_info, min_angle=minAngle, **kwargs)
    mesh = mesh2d(np.array(mesh_meshpy.points, dtype=REAL),
                  np.array(mesh_meshpy.elements, dtype=INDEX))
    if returnFacets:
        return mesh, np.array(points), np.array(facets)
    else:
        return mesh


def wrench(n=8, radius=0.17, radius2=0.3, barLength=2, returnFacets=False, minAngle=30, **kwargs):
    from meshpy.triangle import MeshInfo, build
    mesh_info = MeshInfo()

    points = []
    facets = []
    n = 2
    for i in range(n+1):
        points.append((barLength +
                       radius*np.cos(i*(np.pi/2)/n),
                       radius*np.sin(i*(np.pi/2)/n)))
    n = 3
    for i in range(n+1):
        points.append((-radius2 +
                       radius2*np.cos(i*np.pi/n),
                       radius+radius2*np.sin(i*np.pi/n)))

    r = np.sqrt((1.5*radius2)**2 + radius**2)
    th = np.arctan2(radius, 1.5*radius2)
    n = 1
    for i in range(n+1):
        points.append((-2.5*radius2+r*np.cos(th-th*i/n),
                       r*np.sin(th-th*i/n)))

    for p in reversed(points[1:-1]):
        q = p[0], -p[1]
        points.append(q)

    for i in range(1, len(points)):
        facets.append((i-1, i))
    facets.append((len(points)-1, 0))

    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    mesh_meshpy = build(mesh_info, min_angle=minAngle, **kwargs)
    mesh = mesh2d(np.array(mesh_meshpy.points, dtype=REAL),
                  np.array(mesh_meshpy.elements, dtype=INDEX))
    if returnFacets:
        return mesh, np.array(points), np.array(facets)
    else:
        return mesh


def rectangle(nx, ny, bx=1., by=1., ax=0., ay=0., **kwargs):
    from . meshConstruction import rectangle
    frame = rectangle((ax, ay), (bx, by), num_points=[nx+1, ny+1, nx+1, ny+1])
    mesh = frame.mesh(**kwargs)
    return mesh


def Hshape(a=1., b=1., c=0.3, h=0.2, returnFacets=False):
    from meshpy.triangle import MeshInfo, build
    mesh_info = MeshInfo()

    points = [(0., 0.), (a, 0.), (a, b), (a+c, b), (a+c, 0.), (a+c+a, 0.),
              (a+c+a, b+b+h), (a+c, b+b+h), (a+c, b+h), (a, b+h),
              (a, b+b+h), (0, b+b+h)]
    facets = []
    for i in range(1, len(points)):
        facets.append((i-1, i))
    facets.append((len(points)-1, 0))

    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    mesh_meshpy = build(mesh_info, min_angle=30)
    mesh = mesh2d(np.array(mesh_meshpy.points, dtype=REAL),
                  np.array(mesh_meshpy.elements, dtype=INDEX))
    if returnFacets:
        return mesh, np.array(points), np.array(facets)
    else:
        return mesh


def ball2(radius=1.):
    from meshpy.tet import MeshInfo, build
    mesh_info = MeshInfo()

    points = [(radius, 0, 0), (0, radius, 0), (-radius, 0, 0), (0, -radius, 0),
              (0, 0, radius), (0, 0, -radius)]
    facets = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4),
              (1, 0, 5), (2, 1, 5), (3, 2, 5), (0, 3, 5)]

    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    mesh_meshpy = build(mesh_info)
    mesh = mesh3d(np.array(mesh_meshpy.points, dtype=REAL),
                  np.array(mesh_meshpy.elements, dtype=INDEX))
    from . meshCy import radialMeshTransformer
    mesh.setMeshTransformation(radialMeshTransformer())
    return mesh


def ball(radius=1., points=4, radial_subdiv=None, **kwargs):
    """
    Build mesh for 3D ball as surface of revolution.
    points         determines the number of points on the curve.
    radial_subdiv  determines the number of steps in the rotation.
    """
    from meshpy.tet import MeshInfo, build  # Options
    from meshpy.geometry import generate_surface_of_revolution, EXT_OPEN, GeometryBuilder
    # from meshpy.geometry import make_ball

    if radial_subdiv is None:
        radial_subdiv = 2*points+2

    dphi = np.pi/points

    def truncate(r):
        if abs(r) < 1e-10:
            return 0
        else:
            return r

    rz = [(truncate(radius*np.sin(i*dphi)), radius*np.cos(i*dphi)) for i in range(points+1)]

    geob = GeometryBuilder()
    geob.add_geometry(*generate_surface_of_revolution(rz,
                                                      closure=EXT_OPEN,
                                                      radial_subdiv=radial_subdiv))
    # geob.add_geometry(*make_ball(radius, radial_subdiv))
    mesh_info = MeshInfo()
    geob.set(mesh_info)
    mesh_meshpy = build(mesh_info, **kwargs)  # , options=Options(switches='pq1.2/10')
    mesh = mesh3d(np.array(mesh_meshpy.points, dtype=REAL),
                  np.array(mesh_meshpy.elements, dtype=INDEX))
    from . meshCy import radialMeshTransformer
    mesh.setMeshTransformation(radialMeshTransformer())
    return mesh


def ballNd(dim, radius, h):
    if dim == 1:
        mesh = simpleInterval(-radius, radius)
        while mesh.h > h:
            mesh, lookup = mesh.refine(returnLookup=True)
            radialMeshTransformation(mesh, lookup)
        return mesh
    elif dim == 2:
        return circle(int(np.ceil(2*np.pi*radius/h)), radius, max_volume=0.5*h**2)
    elif dim == 3:
        mesh = ball(radius)
        while mesh.h > h:
            mesh, lookup = mesh.refine(returnLookup=True)
            radialMeshTransformation(mesh, lookup)
        return mesh
    else:
        raise NotImplementedError()


def gradeMesh(mesh, grading):
    vertices = mesh.vertices_as_array
    norms = np.linalg.norm(vertices, axis=1)
    for i in range(vertices.shape[0]):
        n = norms[i]
        if n > 0:
            vertices[i, :] *= grading(n)/n
    mesh.resetMeshInfo()


def gradeUniformBall(mesh,
                     muInterior=2., muExterior=2.,
                     rInterior=1., rExterior=None, rExteriorInitial=None):
    if rExteriorInitial is None:
        rExteriorInitial = np.linalg.norm(mesh.vertices, axis=1).max()
    assert rInterior < rExteriorInitial
    if rExterior is None:
        rExterior = rExteriorInitial

    def grading(r):
        if r <= rInterior:
            return rInterior*(1-(1-r/rInterior)**muInterior)
        else:
            return rInterior + (rExterior-rInterior)*((r-rInterior)/(rExteriorInitial-rInterior))**muExterior

    gradeMesh(mesh, grading)


class meshNd(meshBase):
    def __init__(self, vertices, cells):
        super(meshNd, self).__init__(vertices, cells)

    def __getstate__(self):
        if hasattr(self, '_boundaryVertices'):
            boundaryVertices = self.boundaryVertices
            boundaryVertexTags = self.boundaryVertexTags
        else:
            boundaryVertices = None
            boundaryVertexTags = None
        if hasattr(self, '_boundaryEdges'):
            boundaryEdges = self.boundaryEdges
            boundaryEdgeTags = self.boundaryEdgeTags
        else:
            boundaryEdges = None
            boundaryEdgeTags = None
        if hasattr(self, '_boundaryFaces'):
            boundaryFaces = self.boundaryFaces
            boundaryFaceTags = self.boundaryFaceTags
        else:
            boundaryFaces = None
            boundaryFaceTags = None
        return (super(meshNd, self).__getstate__(),
                boundaryVertices, boundaryVertexTags,
                boundaryEdges, boundaryEdgeTags,
                boundaryFaces, boundaryFaceTags)

    def __setstate__(self, state):
        super(meshNd, self).__setstate__(state[0])
        if state[1] is not None:
            self._boundaryVertices = state[1]
            self._boundaryVertexTags = state[2]
        if state[3] is not None:
            self._boundaryEdges = state[3]
            self._boundaryEdgeTags = state[4]
        if state[5] is not None:
            self._boundaryFaces = state[5]
            self._boundaryFaceTags = state[6]

    def get_boundary_vertices(self):
        if not hasattr(self, '_boundaryVertices'):
            if self.manifold_dim >= 2:
                self._boundaryVertices = boundaryVerticesFromBoundaryEdges(self.boundaryEdges)
            else:
                self._boundaryVertices = boundaryVertices(self.cells)
            return self._boundaryVertices
        else:
            return self._boundaryVertices

    def set_boundary_vertices(self, value):
        self._boundaryVertices = value

    boundaryVertices = property(fget=get_boundary_vertices,
                                fset=set_boundary_vertices)

    def get_boundary_edges(self):
        if not hasattr(self, '_boundaryEdges'):
            if self.manifold_dim == 1:
                self._boundaryEdges = uninitialized((0, 2), dtype=INDEX)
            elif self.manifold_dim == 2:
                self._boundaryEdges = boundaryEdges(self.cells)
            elif self.manifold_dim == 3:
                self._boundaryEdges = boundaryEdgesFromBoundaryFaces(self.boundaryFaces)
            return self._boundaryEdges
        else:
            return self._boundaryEdges

    def set_boundary_edges(self, value):
        assert value.shape[1] == 2
        assert value.dtype == INDEX
        self._boundaryEdges = value

    boundaryEdges = property(fget=get_boundary_edges,
                             fset=set_boundary_edges)

    def get_boundary_faces(self):
        if not hasattr(self, '_boundaryFaces'):
            if self.dim <= 2:
                self._boundaryFaces = uninitialized((0, 3), dtype=INDEX)
            elif self.dim == 3:
                self._boundaryFaces = boundaryFacesWithOrientation(self.vertices, self.cells)
            return self._boundaryFaces
        else:
            return self._boundaryFaces

    def set_boundary_faces(self, value):
        assert value.shape[1] == 3
        self._boundaryFaces = value

    boundaryFaces = property(fget=get_boundary_faces,
                             fset=set_boundary_faces)

    def get_boundary_cells(self):
        if not hasattr(self, '_boundaryCells'):
            if self.manifold_dim == 2:
                self._boundaryEdges, self._boundaryCells = boundaryEdges(self.cells, returnBoundaryCells=True)
            else:
                raise NotImplementedError()
        return self._boundaryCells

    def set_boundary_cells(self, value):
        assert value.ndim == 1
        self._boundaryCells = value

    boundaryCells = property(fget=get_boundary_cells,
                             fset=set_boundary_cells)

    def get_interiorVertices(self):
        if not hasattr(self, '_interiorVertices'):
            temp = np.ones(self.vertices.shape[0], dtype=np.bool)
            temp[self.boundaryVertices] = 0
            self._interiorVertices = temp.nonzero()[0]
            return self._interiorVertices
        else:
            return self._interiorVertices

    def getInteriorVerticesByTag(self, tag=None):
        if not isinstance(tag, list) and tag == NO_BOUNDARY:
            return np.arange(self.num_vertices, dtype=INDEX)
        else:
            bv = self.getBoundaryVerticesByTag(tag)
            idx = np.ones(self.num_vertices, dtype=np.bool)
            idx[bv] = False
            return np.nonzero(idx)[0].astype(INDEX)

    def get_diam(self):
        from numpy.linalg import norm
        vertices = self.vertices_as_array
        return norm(vertices.max(axis=0)-vertices.min(axis=0), 2)

    interiorVertices = property(fget=get_interiorVertices)
    diam = property(fget=get_diam)

    def copy(self):
        newMesh = super(meshNd, self).copy()
        if hasattr(self, '_boundaryVertices'):
            newMesh._boundaryVertices = self._boundaryVertices.copy()
        if hasattr(self, '_boundaryVertexTags'):
            newMesh._boundaryVertexTags = self._boundaryVertexTags.copy()
        if hasattr(self, '_boundaryEdges'):
            newMesh._boundaryEdges = self._boundaryEdges.copy()
        if hasattr(self, '_boundaryEdgeTags'):
            newMesh._boundaryEdgeTags = self._boundaryEdgeTags.copy()
        if hasattr(self, '_boundaryFaces'):
            newMesh._boundaryFaces = self._boundaryFaces.copy()
        if hasattr(self, '_boundaryFaceTags'):
            newMesh._boundaryFaceTags = self._boundaryFaceTags.copy()
        return newMesh

    def __repr__(self):
        return ('{} with {:,} vertices '
                + 'and {:,} cells').format(self.__class__.__name__,
                                           self.num_vertices,
                                           self.num_cells)

    def get_boundary_vertex_tags(self):
        if not hasattr(self, '_boundaryVertexTags'):
            self._boundaryVertexTags = PHYSICAL*np.zeros((self.boundaryVertices.shape[0]),
                                                         dtype=TAG)
        return self._boundaryVertexTags

    def set_boundary_vertex_tags(self, value):
        assert value.shape[0] == self.boundaryVertices.shape[0]
        assert value.dtype == TAG
        self._boundaryVertexTags = value

    boundaryVertexTags = property(fset=set_boundary_vertex_tags,
                                  fget=get_boundary_vertex_tags)

    def tagBoundaryVertices(self, tagFunc):
        boundaryVertexTags = uninitialized((self.boundaryVertices.shape[0]),
                                           dtype=TAG)
        for i, j in enumerate(self.boundaryVertices):
            v = self.vertices[j, :]
            boundaryVertexTags[i] = tagFunc(v)
        self.boundaryVertexTags = boundaryVertexTags

    def replaceBoundaryVertexTags(self, tagFunc, tagsToReplace=set()):
        boundaryVertexTags = uninitialized((self.boundaryVertices.shape[0]),
                                           dtype=TAG)
        for i, j in enumerate(self.boundaryVertices):
            if self.boundaryVertexTags[i] in tagsToReplace:
                v = self.vertices[j, :]
                boundaryVertexTags[i] = tagFunc(v)
            else:
                boundaryVertexTags[i] = self.boundaryVertexTags[i]
        self.boundaryVertexTags = boundaryVertexTags

    def getBoundaryVerticesByTag(self, tag=None, sorted=False):
        if tag is None:
            bv = self.boundaryVertices
        elif isinstance(tag, list) and tag[0] is None:
            bv = self.boundaryVertices
        elif isinstance(tag, list):
            idx = (self.boundaryVertexTags == tag[0])
            for t in tag[1:]:
                idx = np.logical_or(idx, (self.boundaryVertexTags == t))
            bv = self.boundaryVertices[idx]
        else:
            bv = self.boundaryVertices[self.boundaryVertexTags == tag]
        if sorted:
            bv.sort()
        return bv

    def get_boundary_edge_tags(self):
        if not hasattr(self, '_boundaryEdgeTags'):
            self._boundaryEdgeTags = PHYSICAL*np.ones(self.boundaryEdges.shape[0],
                                                      dtype=TAG)
        return self._boundaryEdgeTags

    def set_boundary_edge_tags(self, value):
        assert value.shape[0] == self.boundaryEdges.shape[0]
        self._boundaryEdgeTags = value

    boundaryEdgeTags = property(fset=set_boundary_edge_tags,
                                fget=get_boundary_edge_tags)

    def tagBoundaryEdges(self, tagFunc):
        boundaryEdgeTags = uninitialized(self.boundaryEdges.shape[0],
                                         dtype=TAG)
        for i in range(self.boundaryEdges.shape[0]):
            e = self.boundaryEdges[i, :]
            v0 = self.vertices[e[0]]
            v1 = self.vertices[e[1]]
            boundaryEdgeTags[i] = tagFunc(v0, v1)
        self.boundaryEdgeTags = boundaryEdgeTags

    def replaceBoundaryEdgeTags(self, tagFunc, tagsToReplace=set()):
        boundaryEdgeTags = uninitialized((self.boundaryEdges.shape[0]),
                                         dtype=TAG)
        for i in range(self.boundaryEdges.shape[0]):
            if self.boundaryEdgeTags[i] in tagsToReplace:
                e = self.boundaryEdges[i, :]
                v0 = self.vertices[e[0]]
                v1 = self.vertices[e[1]]
                boundaryEdgeTags[i] = tagFunc(v0, v1)
            else:
                boundaryEdgeTags[i] = self.boundaryEdgeTags[i]
        self.boundaryEdgeTags = boundaryEdgeTags

    def getBoundaryEdgesByTag(self, tag=None, returnBoundaryCells=False):
        if tag is None:
            if not returnBoundaryCells:
                return self.boundaryEdges
            else:
                assert self.dim == 2
                return self.boundaryEdges, self.boundaryCells
        else:
            if not type(tag) is list:
                tag = [tag]
            idx = (self.boundaryEdgeTags == tag[0])
            for t in tag[1:]:
                idx = np.logical_or(idx, (self.boundaryEdgeTags == t))
            if not returnBoundaryCells:
                return self.boundaryEdges[idx, :]
            else:
                return self.boundaryEdges[idx, :], self.boundaryCells[idx]

    def get_boundary_face_tags(self):
        if not hasattr(self, '_boundaryFaceTags'):
            self._boundaryFaceTags = PHYSICAL*np.ones(self.boundaryFaces.shape[0],
                                                      dtype=TAG)
        return self._boundaryFaceTags

    def set_boundary_face_tags(self, value):
        assert value.shape[0] == self.boundaryFaces.shape[0]
        self._boundaryFaceTags = value

    boundaryFaceTags = property(fset=set_boundary_face_tags,
                                fget=get_boundary_face_tags)

    def tagBoundaryFaces(self, tagFunc):
        boundaryFaceTags = uninitialized(self.boundaryFaces.shape[0],
                                         dtype=TAG)
        for i in range(self.boundaryFaces.shape[0]):
            f = self.boundaryFaces[i, :]
            v0 = self.vertices[f[0]]
            v1 = self.vertices[f[1]]
            v2 = self.vertices[f[2]]
            boundaryFaceTags[i] = tagFunc(v0, v1, v2)
        self.boundaryFaceTags = boundaryFaceTags

    def getBoundaryFacesByTag(self, tag=None):
        if tag is None:
            return self.boundaryFaces
        elif type(tag) is list:
            idx = (self.boundaryFaceTags == tag[0])
            for t in tag[1:]:
                idx = np.logical_or(idx, (self.boundaryFaceTags == t))
            return self.boundaryFaces[idx]
        else:
            return self.boundaryFaces[self.boundaryFaceTags == tag]

    def HDF5write(self, node):
        COMPRESSION = 'gzip'
        node.create_dataset('vertices', data=self.vertices,
                            compression=COMPRESSION)
        node.create_dataset('cells', data=self.cells,
                            compression=COMPRESSION)
        if hasattr(self, '_boundaryVertices'):
            node.create_dataset('boundaryVertices',
                                data=self.boundaryVertices,
                                compression=COMPRESSION)
        if hasattr(self, '_boundaryVertexTags'):
            node.create_dataset('boundaryVertexTags',
                                data=self.boundaryVertexTags,
                                compression=COMPRESSION)
        if hasattr(self, '_boundaryEdges'):
            node.create_dataset('boundaryEdges',
                                data=self.boundaryEdges,
                                compression=COMPRESSION)
        if hasattr(self, '_boundaryEdgeTags'):
            node.create_dataset('boundaryEdgeTags',
                                data=self.boundaryEdgeTags,
                                compression=COMPRESSION)
        if hasattr(self, '_boundaryFaces'):
            node.create_dataset('boundaryFaces',
                                data=self.boundaryFaces,
                                compression=COMPRESSION)
        if hasattr(self, '_boundaryFaceTags'):
            node.create_dataset('boundaryFaceTags',
                                data=self.boundaryFaceTags,
                                compression=COMPRESSION)
        node.attrs['dim'] = self.dim

    @staticmethod
    def HDF5read(node):
        dim = node.attrs['dim']
        vertices = np.array(node['vertices'], dtype=REAL)
        cells = np.array(node['cells'], dtype=INDEX)
        if dim == 1:
            mesh = mesh1d(vertices, cells)
        elif dim == 2:
            mesh = mesh2d(vertices, cells)
        elif dim == 3:
            mesh = mesh3d(vertices, cells)
        if 'boundaryVertices' in node:
            mesh.boundaryVertices = np.array(node['boundaryVertices'],
                                             dtype=INDEX)
        if 'boundaryVertexTags' in node:
            mesh.boundaryVertexTags = np.array(node['boundaryVertexTags'],
                                               dtype=TAG)
        if 'boundaryEdges' in node:
            mesh.boundaryEdges = np.array(node['boundaryEdges'],
                                          dtype=INDEX)
        if 'boundaryEdgeTags' in node:
            mesh.boundaryEdgeTags = np.array(node['boundaryEdgeTags'],
                                             dtype=TAG)
        if 'boundaryFaces' in node:
            mesh.boundaryFaces = np.array(node['boundaryFaces'],
                                          dtype=INDEX)
        if 'boundaryFaceTags' in node:
            mesh.boundaryFaceTags = np.array(node['boundaryFaceTags'],
                                             dtype=TAG)
        return mesh

    def exportVTK(self, filename, cell_data=None):
        import meshio
        if self.manifold_dim == 1:
            cell_type = 'line'
        elif self.manifold_dim == 2:
            cell_type = 'triangle'
        elif self.manifold_dim == 3:
            cell_type = 'tetra'
        else:
            raise NotImplementedError()
        vertices = np.zeros((self.num_vertices, 3), dtype=REAL)
        vertices[:, 3-self.dim:] = self.vertices_as_array
        meshio.write(filename,
                     meshio.Mesh(vertices,
                                 {cell_type: self.cells_as_array},
                                 cell_data=cell_data),
                     file_format='vtk')

    def exportSolutionVTK(self, x, filename, labels='solution', cell_data={}):
        import meshio
        from . DoFMaps import Product_DoFMap, P0_DoFMap
        if not isinstance(x, (list, tuple)):
            x = [x]
            labels = [labels]
        else:
            assert len(x) == len(labels)
        point_data = {}
        for xx, label in zip(x, labels):
            if isinstance(xx.dm, P0_DoFMap):
                cell_data[label] = [xx.toarray()]
            else:
                sol = xx.linearPart()

                if isinstance(xx.dm, Product_DoFMap):
                    v2d = -np.ones((self.num_vertices, 1), dtype=INDEX)
                    sol.dm.getVertexDoFs(v2d)
                    sol2 = np.zeros((self.num_vertices, sol.dm.numComponents), dtype=REAL)
                    for component in range(sol.dm.numComponents):
                        R, _ = sol.dm.getRestrictionProlongation(component)
                        for i in range(self.num_vertices):
                            dof = v2d[i, 0]
                            if dof >= 0:
                                sol2[i, component] = (R*sol)[dof]
                    point_data[label] = sol2
                else:
                    v2d = -np.ones((self.num_vertices, 1), dtype=INDEX)
                    sol.dm.getVertexDoFs(v2d)
                    sol2 = np.zeros((self.num_vertices), dtype=REAL)
                    for i in range(self.num_vertices):
                        dof = v2d[i, 0]
                        if dof >= 0:
                            sol2[i] = sol[dof]
                    point_data[label] = np.array(sol2)
        if self.manifold_dim == 1:
            cell_type = 'line'
        elif self.manifold_dim == 2:
            cell_type = 'triangle'
        elif self.manifold_dim == 3:
            cell_type = 'tetra'
        else:
            raise NotImplementedError()
        vertices = np.zeros((self.num_vertices, 3), dtype=REAL)
        vertices[:, 3-self.dim:] = self.vertices_as_array
        meshio.write(filename,
                     meshio.Mesh(vertices,
                                 {cell_type: self.cells_as_array},
                                 point_data=point_data,
                                 cell_data=cell_data),
                     file_format='vtk')

    @staticmethod
    def readMesh(filename, file_format=None):
        import meshio
        mesh = meshio.read(filename, file_format)
        vertices = mesh.points.astype(REAL)
        dim = vertices.shape[1]
        assert len(mesh.cells)
        cell_type = mesh.cells[0].type
        if cell_type == 'line':
            dim = 1
            meshType = mesh1d
        elif cell_type == 'triangle':
            dim = 2
            meshType = mesh2d
        elif cell_type == 'tetra':
            dim = 3
            meshType = mesh3d
        else:
            raise NotImplementedError()
        vertices = np.ascontiguousarray(vertices[:, :dim])
        cells = mesh.cells[0].data.astype(INDEX)
        return meshType(vertices, cells)

    def getPartitions(self, numPartitions, partitioner='metis', partitionerParams={}):
        # partition mesh cells
        if partitioner == 'regular':
            mP = regularMeshPartitioner(self)
            defaultParams = {'partitionedDimensions': self.dim}
            if 'regular' in partitionerParams:
                defaultParams.update(partitionerParams['regular'])
            part, actualNumPartitions = mP.partitionCells(numPartitions,
                                                          partitionedDimensions=defaultParams['partitionedDimensions'])
        elif partitioner == 'metis':
            mP = metisMeshPartitioner(self)
            defaultParams = {'partition_weights': None}
            if 'metis' in partitionerParams:
                defaultParams.update(partitionerParams['metis'])
            part, actualNumPartitions = mP.partitionCells(numPartitions,
                                                          partition_weights=defaultParams['partition_weights'])
        else:
            raise NotImplementedError()
        if not actualNumPartitions == numPartitions:
            raise PartitionerException('Partitioner returned {} partitions instead of {}.'.format(actualNumPartitions, numPartitions))
        return part

    def getCuthillMckeeVertexOrder(self):
        from PyNucleus_base.linear_operators import sparseGraph
        from PyNucleus_base.sparseGraph import cuthill_mckee
        from . import P1_DoFMap
        dm = P1_DoFMap(self, -10)
        A = dm.buildSparsityPattern(self.cells)
        graph = sparseGraph(A.indices, A.indptr, A.shape[0], A.shape[1])
        idx = uninitialized((dm.num_dofs), dtype=INDEX)
        cuthill_mckee(graph, idx)
        return idx

    def global_h(self, comm):
        h = self.h
        if comm is None:
            return h
        else:
            return comm.allreduce(h, op=MPI.MAX)

    def global_hmin(self, comm):
        hmin = self.hmin
        if comm is None:
            return hmin
        else:
            return comm.allreduce(hmin, op=MPI.MIN)

    def global_volume(self, comm):
        vol = self.volume
        if comm is None:
            return vol
        else:
            return comm.allreduce(vol, op=MPI.SUM)

    def global_diam(self, comm):
        if comm is None:
            return self.diam()
        from numpy.linalg import norm
        m = self.vertices.min(axis=0)
        M = self.vertices.max(axis=0)
        comm.Allreduce(m, MPI.IN_PLACE, op=MPI.MIN)
        comm.Allreduce(M, MPI.IN_PLACE, op=MPI.MAX)
        return norm(M-m, 2)

    def get_surface(self):
        if self.dim == 1:
            return 1.0
        else:
            return self.get_surface_mesh().volume

    surface = property(fget=get_surface)

    def get_surface_mesh(self, tag=None):
        if self.dim == 1:
            bv = self.getBoundaryVerticesByTag(tag)
            cells = uninitialized((len(bv), 1), dtype=INDEX)
            cells[:, 0] = bv
            surface = mesh0d(self.vertices, cells)
        elif self.dim == 2:
            surface = mesh1d(self.vertices, self.getBoundaryEdgesByTag(tag))
        elif self.dim == 3:
            surface = mesh2d(self.vertices, self.getBoundaryFacesByTag(tag))
        else:
            raise NotImplementedError()
        surface.setMeshTransformation(self.transformer)
        return surface

    def reorderVertices(self, idx):
        invidx = uninitialized_like(idx)
        invidx[idx] = np.arange(self.num_vertices, dtype=INDEX)
        self.vertices = self.vertices_as_array[idx, :]
        if hasattr(self, '_boundaryVertices'):
            self._boundaryVertices = invidx[self._boundaryVertices].astype(INDEX)
        if hasattr(self, '_boundaryEdges'):
            self._boundaryEdges = invidx[self._boundaryEdges].astype(INDEX)
        if hasattr(self, '_boundaryFaces'):
            self._boundaryEdges = invidx[self._boundaryEdges].astype(INDEX)
        self.cells = invidx[self.cells_as_array[:, :]].astype(INDEX)


class mesh0d(meshNd):
    pass


class mesh1d(meshNd):
    def plot(self, vertices=True, boundary=None, info=False):
        import matplotlib.pyplot as plt
        X = np.array([v[0] for v in self.vertices])
        if self.vertices.shape[1] == 1:
            Y = np.zeros_like(X)
            lenX = X.max()-X.min()
            plt.xlim([X.min()-lenX*0.1, X.max()+lenX*0.1])
            plt.plot(X, Y, 'o-' if vertices else '-', zorder=1)
        else:
            v = self.vertices_as_array
            c = self.cells_as_array
            plt.plot([v[c[:, 0], 0],
                      v[c[:, 1], 0]],
                     [v[c[:, 0], 1],
                      v[c[:, 1], 1]],
                     c='k')
            if vertices:
                plt.scatter(self.vertices_as_array[:, 0], self.vertices_as_array[:, 1])
            lenX = v[:, 0].max()-v[:, 0].min()
            plt.xlim([v[:, 0].min()-lenX*0.1, v[:, 0].max()+lenX*0.1])
            lenY = v[:, 1].max()-v[:, 1].min()
            plt.ylim([v[:, 1].min()-lenY*0.1, v[:, 1].max()+lenY*0.1])
            plt.axis('equal')
        if info:
            tags = set(self.boundaryEdgeTags)
            tags = tags.union(self.boundaryVertexTags)
            cm = plt.get_cmap('gist_rainbow')
            num_colors = len(tags)
            colors = {tag: cm(i/num_colors) for i, tag in enumerate(tags)}
            for i, c in enumerate(self.cells):
                midpoint = (self.vertices_as_array[c[0], :]
                            + self.vertices_as_array[c[1], :])/2
                if midpoint.shape[0] == 1:
                    plt.text(midpoint[0], 0, str(i), style='italic')
                else:
                    plt.text(midpoint[0], midpoint[1], str(i), style='italic')
            for i, v in enumerate(self.vertices_as_array):
                if v.shape[0] == 1:
                    plt.text(v, 0, i)
                else:
                    plt.text(v[0], v[1], i)
            for vno, tag in zip(self.boundaryVertices,
                                self.boundaryVertexTags):
                v = self.vertices_as_array[vno, :]
                if v.shape[0] == 1:
                    plt.text(v[0], 0, tag, horizontalalignment='right',
                             verticalalignment='top', color=colors[tag])
                else:
                    plt.text(v[0], v[1], tag, horizontalalignment='right',
                             verticalalignment='top', color=colors[tag])
            for i, (e, tag) in enumerate(zip(self.boundaryEdges,
                                             self.boundaryEdgeTags)):
                v = (self.vertices_as_array[e[0], :]+self.vertices_as_array[e[1], :])/2
                if v.shape[0] == 1:
                    plt.text(v[0], 0, tag, color=colors[tag])
                else:
                    plt.text(v[0], v[1], tag, color=colors[tag])

    def plotPrepocess(self, x, DoFMap):
        from . DoFMaps import P0_DoFMap
        if not isinstance(DoFMap, P0_DoFMap):
            positions = uninitialized((DoFMap.num_dofs+DoFMap.num_boundary_dofs), dtype=REAL)
            dof2pos = np.full((DoFMap.num_boundary_dofs), dtype=INDEX, fill_value=-1)
            bDoF = DoFMap.num_dofs
            simplex = uninitialized((self.dim+1, self.dim), dtype=REAL)
            for cellNo in range(self.num_cells):
                self.getSimplex_py(cellNo, simplex)
                for i in range(DoFMap.dofs_per_element):
                    dof = DoFMap.cell2dof_py(cellNo, i)
                    pos = np.dot(DoFMap.nodes[i, :], simplex)
                    if dof >= 0:
                        positions[dof] = pos[0]
                    else:
                        p = dof2pos[-dof-1]
                        if p == -1:
                            p = dof2pos[-dof-1] = bDoF
                            bDoF += 1
                        positions[p] = pos[0]
            if x.ndim == 1:
                xx = np.zeros((DoFMap.num_dofs+DoFMap.num_boundary_dofs), dtype=REAL)
                xx[:DoFMap.num_dofs] = x
            else:
                xx = np.zeros((x.shape[0], self.num_vertices), dtype=REAL)
                xx[:, :DoFMap.num_dofs] = x
        else:
            positions = uninitialized((2*(DoFMap.num_dofs+DoFMap.num_boundary_dofs)), dtype=REAL)
            dof2pos = np.full((DoFMap.num_boundary_dofs), dtype=INDEX, fill_value=-1)
            bDoF = DoFMap.num_dofs
            simplex = uninitialized((self.dim+1, self.dim), dtype=REAL)
            for cellNo in range(self.num_cells):
                self.getSimplex_py(cellNo, simplex)
                for i in range(DoFMap.dofs_per_element):
                    dof = DoFMap.cell2dof_py(cellNo, i)
                    if dof >= 0:
                        positions[2*dof] = min(simplex[0, 0], simplex[1, 0])+1e-9
                        positions[2*dof+1] = max(simplex[0, 0], simplex[1, 0])-1e-9
                    else:
                        p = dof2pos[-dof-1]
                        if p == -1:
                            p = dof2pos[-dof-1] = bDoF
                            bDoF += 1
                        positions[2*p] = min(simplex[0, 0], simplex[1, 0])+1e-9
                        positions[2*p+1] = max(simplex[0, 0], simplex[1, 0])-1e-9
            if x.ndim == 1:
                xx = np.zeros((2*(DoFMap.num_dofs+DoFMap.num_boundary_dofs)), dtype=REAL)
                xx[:2*DoFMap.num_dofs-1:2] = x
                xx[1:2*DoFMap.num_dofs:2] = x
            else:
                xx = np.zeros((x.shape[0], 2*(DoFMap.num_dofs+DoFMap.num_boundary_dofs)), dtype=REAL)
                xx[:, :2*DoFMap.num_dofs-1:2] = x
                xx[:, 1:2*DoFMap.num_dofs:2] = x
            positions = np.concatenate((positions, self.vertices_as_array[:, 0]))
            if x.ndim == 1:
                shape = (self.num_vertices, )
            else:
                shape = (x.shape[0], self.num_vertices)
            xx = np.hstack((xx, np.full(shape, fill_value=np.nan, dtype=REAL)))
        idx = np.argsort(positions)
        positions = positions[idx]
        if x.ndim == 1:
            xx = xx[idx]
        else:
            xx = xx[:, idx]
        return positions, xx

    def plotFunction(self, x, DoFMap=None, tag=0, flat=False, yvals=None, fig=None, ax=None, update=None, **kwargs):
        import matplotlib.pyplot as plt
        if fig is None:
            fig = plt.gcf()
        if ax is None:
            ax = fig.gca()
        if DoFMap:
            positions, sol = self.plotPrepocess(x, DoFMap)
        else:
            if x.shape[0] == self.num_cells:
                from . DoFMaps import P0_DoFMap
                dm = P0_DoFMap(self)
                positions, sol = self.plotPrepocess(x, dm)
            elif x.shape[0] < self.num_vertices:
                positions = self.vertices_as_array[:, 0]
                sol = np.zeros((self.num_vertices))
                sol[self.getInteriorVerticesByTag(tag)] = x
            else:
                positions = self.vertices_as_array[:, 0]
                sol = x
            idx = np.argsort(positions)
            positions = positions[idx]
            sol = sol[idx]

        if sol.ndim == 1:
            if update is None:
                return ax.plot(positions, sol, **kwargs)[0]
            else:
                update.set_data(positions, sol)
        else:
            from matplotlib import cm
            assert yvals is not None
            X, Y = np.meshgrid(positions, yvals)
            if flat:
                ax.pcolor(X, Y,
                          sol, cmap=cm.jet,
                          **kwargs)
            else:
                fig = plt.gcf()
                fig.delaxes(fig.gca())
                ax = fig.add_subplot(projection='3d')
                ax.plot_surface(X, Y, sol, cmap=cm.jet, **kwargs)

    def plotDoFMap(self, DoFMap, printDoFIndices=True):
        "Plot the DoF numbers on the mesh."
        import matplotlib.pyplot as plt
        from matplotlib import rc_context
        self.plot()
        pos = DoFMap.getDoFCoordinates()
        if printDoFIndices:
            with rc_context({'text.usetex': False}):
                for dof in range(DoFMap.num_dofs):
                    plt.text(pos[dof, 0], 0, str(dof))
        else:
            plt.scatter(pos[:, 0], np.zeros((pos.shape[0])), marker='x', s=60)

    def plotMeshOverlap(self, overlap):
        "Plot a single mesh overlap."
        from . meshOverlaps import meshOverlap
        assert isinstance(overlap, meshOverlap)
        import matplotlib.pyplot as plt
        # self.plot(boundary=True)
        self.plot(boundary=False)
        for i in range(overlap.num_vertices):
            v = self.cells[overlap.vertices[i, 0], overlap.vertices[i, 1]]
            plt.text(self.vertices[v, 0], self.vertices[v, 1], str(i))
        for i in range(overlap.num_cells):
            cellNo = overlap.cells[i]
            simplex = self.vertices[self.cells[cellNo, :], :]
            XY = simplex.mean(axis=0)
            plt.text(XY[0], 0, str(i))

    def plotOverlapManager(self, overlap):
        "Plot all mesh overlaps in an overlap manager."
        from . meshOverlaps import overlapManager
        assert isinstance(overlap, overlapManager)
        import matplotlib.pyplot as plt
        self.plot()
        x = np.zeros((self.num_cells), dtype=REAL)
        for subdomain in overlap.overlaps:
            for cellNo in overlap.overlaps[subdomain].cells:
                x[cellNo] += 1
        for cellNo in range(self.num_cells):
            plt.text(self.vertices[self.cells[cellNo, :], 0].mean(), 0, str(x[cellNo]))
        plt.axis('equal')

    def plotAlgebraicOverlap(self, DoFMap, overlap):
        "Plot a single algebraic overlap."
        from . algebraicOverlaps import algebraicOverlap
        assert isinstance(overlap, algebraicOverlap)
        import matplotlib.pyplot as plt
        self.plot(boundary=True)
        dofDict = {}
        for i, dof in enumerate(overlap.shared_dofs):
            dofDict[dof] = i
        for cellNo in range(self.num_cells):
            simplex = self.vertices[self.cells[cellNo, :], :]
            for i in range(DoFMap.dofs_per_element):
                dof = DoFMap.cell2dof_py(cellNo, i)
                try:
                    pos = np.dot(DoFMap.nodes[i, :], simplex)
                    plt.text(pos[0], 0, str(dofDict[dof]))
                except:
                    pass

    def plotAlgebraicOverlapManager(self, DoFMap, overlap):
        from . algebraicOverlaps import algebraicOverlapManager
        assert isinstance(overlap, algebraicOverlapManager)
        self.plot(boundary=True)
        x = np.zeros((DoFMap.num_dofs), dtype=REAL)
        for subdomainNo in overlap.overlaps:
            for i, dof in enumerate(overlap.overlaps[subdomainNo].shared_dofs):
                x[dof] += 1
        self.plotFunctionDoFMap(DoFMap, x)

    def plotFunctionDoFMap(self, DoFMap, x):
        "Display function values for every DoF."
        import matplotlib.pyplot as plt
        self.plot()
        for cellNo in range(self.num_cells):
            simplex = self.vertices[self.cells[cellNo, :], :]
            for i in range(DoFMap.dofs_per_element):
                dof = DoFMap.cell2dof_py(cellNo, i)
                if dof >= 0:
                    pos = np.dot(DoFMap.nodes[i, :], simplex)
                    plt.text(pos[0], 0, '{:.2}'.format(x[dof]))

    def sortVertices(self):
        idx = np.argsort(self.vertices_as_array, axis=0).ravel()
        self.reorderVertices(idx)


class mesh2d(meshNd):
    """
    2D mesh

    Attributes:
    vertices
    cells
    boundaryVertices
    boundaryEdges
    boundaryVertexTags
    boundaryEdgeTags
    """

    def getInteriorMap(self, tag):
        """
        Returns a map from the vertex numbers of the mesh
        to the interior vertices.
        """
        bdofs = self.getBoundaryVerticesByTag(tag)
        mapping = -1*np.ones((self.num_vertices), dtype=INDEX)
        iV = np.ones(self.num_vertices, dtype=np.bool)
        iV[bdofs] = 0
        iV = iV.nonzero()[0]
        mapping[iV] = np.arange(len(iV), dtype=INDEX)
        return mapping

    def plot(self, boundary=None, info=False, padding=0.1, fill=False, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        vertices = self.vertices_as_array
        X, Y = vertices[:, 0], vertices[:, 1]
        triangles = self.cells
        lenX = X.max()-X.min()
        lenY = Y.max()-Y.min()
        plt.axis('equal')
        plt.xlim([X.min()-lenX*padding, X.max()+lenX*padding])
        plt.ylim([Y.min()-lenY*padding, Y.max()+lenY*padding])
        if fill:
            plt.tripcolor(X, Y, triangles, np.ones(triangles.shape[0]), 'k-', zorder=1, alpha=0.3 if boundary else 1., **kwargs)
        else:
            if 'alpha' not in kwargs:
                kwargs['alpha'] = 0.3 if boundary else 1.
            plt.triplot(X, Y, triangles, 'k-', zorder=1, **kwargs)
        if boundary:
            tags = set(self.boundaryEdgeTags)
            tags = tags.union(self.boundaryVertexTags)
            cm = plt.get_cmap('gist_rainbow')
            num_colors = len(tags)
            colors = {tag: cm(i/(num_colors)) for i, tag in enumerate(sorted(tags))}
            vertices = self.vertices_as_array
            for bv, tag in zip(self.boundaryVertices, self.boundaryVertexTags):
                XY = vertices[bv, :]
                plt.plot([XY[0]], [XY[1]], '-o',
                         linewidth=0*rcParams["lines.linewidth"],
                         markersize=10,
                         color=colors[tag],
                         zorder=3)
            for be, tag in zip(self.boundaryEdges, self.boundaryEdgeTags):
                XY = vertices[be, :]
                plt.plot(XY[:, 0], XY[:, 1], 'k-',
                         linewidth=3*rcParams["lines.linewidth"],
                         color=colors[tag],
                         zorder=2)

        if info:
            tags = set(self.boundaryEdgeTags)
            tags = tags.union(self.boundaryVertexTags)
            cm = plt.get_cmap('gist_rainbow')
            num_colors = len(tags)
            colors = {tag: cm(i/num_colors) for i, tag in enumerate(tags)}
            vertices = self.vertices_as_array
            for i, c in enumerate(self.cells):
                midpoint = (vertices[c[0]]
                            + vertices[c[1]]
                            + vertices[c[2]])/3
                plt.text(midpoint[0], midpoint[1], str(i), style='italic')
            for i, v in enumerate(vertices):
                plt.text(v[0], v[1], i)
            for vno, tag in zip(self.boundaryVertices,
                                self.boundaryVertexTags):
                v = self.vertices[vno, :]
                plt.text(v[0], v[1], tag, horizontalalignment='right',
                         verticalalignment='top', color=colors[tag])
            for i, (e, tag) in enumerate(zip(self.boundaryEdges,
                                             self.boundaryEdgeTags)):
                v = (vertices[e[0]]+vertices[e[1]])/2
                plt.text(v[0], v[1], tag, color=colors[tag])

    def plotPrepocess(self, x, DoFMap=None, tag=0):
        from . DoFMaps import P1_DoFMap, P0_DoFMap
        if DoFMap is not None and hasattr(x, 'dm'):
            DoFMap = x.dm
        if DoFMap is not None:
            if isinstance(DoFMap, P0_DoFMap):
                if DoFMap.num_dofs < self.num_cells:
                    from . DoFMaps import getSubMapRestrictionProlongation
                    dm = P0_DoFMap(self, -10)
                    _, P = getSubMapRestrictionProlongation(dm, DoFMap)
                    y = P*x
                    return self.plotPrepocess(y)
                else:
                    return self.plotPrepocess(x)
            elif not isinstance(DoFMap, P1_DoFMap):
                return self.plotPrepocess(DoFMap.linearPart(x)[0])
            elif isinstance(DoFMap, P1_DoFMap):
                v = self.vertices_as_array
                X, Y = v[:, 0], v[:, 1]
                sol = DoFMap.getValuesAtVertices(x)
                return X, Y, sol
        else:
            v = self.vertices_as_array
            X, Y = v[:, 0], v[:, 1]
            if x.shape[0] == self.num_vertices:
                sol = x
            elif x.shape[0] == self.num_cells:
                sol = x
            else:
                sol = np.zeros(self.num_vertices)
                if DoFMap is not None:
                    tag = DoFMap.tag
                sol[self.getInteriorVerticesByTag(tag)] = x
            return X, Y, sol

    def plotFunction(self, x, flat=False, DoFMap=None, tag=0, update=None, contour=False, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        X, Y, sol = self.plotPrepocess(x, DoFMap, tag)
        if flat:
            plt.axis('equal')
            if update is None:
                try:
                    cb = plt.gca().collections[-1].colorbar
                    cb.remove()
                except:
                    pass
                update = plt.tripcolor(X, Y, self.cells, sol, cmap=cm.jet, linewidth=0, **kwargs)
                plt.colorbar()
                if contour:
                    update2 = plt.tricontour(X, Y, self.cells, sol, colors=['k'])
                    update = [update, update2]
                return update
            else:
                if contour:
                    update[0].set_array(sol)
                    for cp in update[1].collections:
                        cp.remove()
                    update[1] = plt.tricontour(X, Y, self.cells, sol, colors=['k'])
                else:
                    if sol.shape[0] != update.get_array().shape[0]:
                        sol = sol[self.cells_as_array].mean(axis=1)
                    assert sol.shape[0] == update.get_array().shape[0]
                    update.set_array(sol)
        else:
            from . DoFMaps import P0_DoFMap
            if isinstance(DoFMap, P0_DoFMap):
                assert self.num_cells == sol.shape[0]
                newVertices = uninitialized(((self.dim+1)*self.num_cells, self.dim),
                                            dtype=REAL)
                newCells = uninitialized((self.num_cells, self.dim+1),
                                         dtype=INDEX)
                newSol = uninitialized(((self.dim+1)*self.num_cells, ),
                                       dtype=REAL)
                k = 0
                for cellNo in range(self.num_cells):
                    for vertexNo in range(self.dim+1):
                        vertex = self.cells[cellNo, vertexNo]
                        for j in range(self.dim):
                            newVertices[k, j] = self.vertices[vertex, j]
                        newCells[cellNo, vertexNo] = k
                        newSol[(self.dim+1)*cellNo+vertexNo] = sol[cellNo]
                        k += 1
                X, Y = newVertices[:, 0], newVertices[:, 1]
                sol = newSol
                cells = newCells
            else:
                cells = self.cells
            if ax is None:
                fig = plt.gcf()
                fig.delaxes(fig.gca())
                ax = fig.add_subplot(projection='3d')
            ax.plot_trisurf(X, Y, cells, sol, cmap=cm.jet, linewidth=0, **kwargs)
            return ax

    def plotDoFMap(self, DoFMap, printDoFIndices=True):
        "Plot the DoF numbers on the mesh."
        import matplotlib.pyplot as plt
        from matplotlib import rc_context
        self.plot(alpha=0.3)
        pos = DoFMap.getDoFCoordinates()
        if printDoFIndices:
            with rc_context({'text.usetex': False}):
                for dof in range(DoFMap.num_dofs):
                    plt.text(pos[dof, 0], pos[dof, 1], str(dof),
                             horizontalalignment='center',
                             verticalalignment='center')
        else:
            plt.scatter(pos[:, 0], pos[:, 1])

    def plotFunctionDoFMap(self, DoFMap, x):
        "Display function values for every DoF."
        import matplotlib.pyplot as plt
        self.plot()
        for cellNo in range(self.num_cells):
            simplex = self.vertices_as_array[self.cells[cellNo, :], :]
            for i in range(DoFMap.dofs_per_element):
                dof = DoFMap.cell2dof_py(cellNo, i)
                if dof >= 0:
                    pos = np.dot(DoFMap.nodes[i, :], simplex)
                    plt.text(pos[0], pos[1], '{:.2}'.format(x[dof]))

    def plotInterface(self, interface):
        "Plot a single mesh interface."
        import matplotlib.pyplot as plt
        from . meshOverlaps import meshInterface
        assert isinstance(interface, meshInterface)
        self.plot()
        for i in range(interface.num_edges):
            cellNo = interface.edges[i, 0]
            edgeNo = interface.edges[i, 1]
            order = interface.edges[i, 2]
            simplex = self.vertices[self.cells[cellNo, :], :]
            if edgeNo == 0:
                idx = (0, 1)
            elif edgeNo == 1:
                idx = (1, 2)
            else:
                idx = (2, 0)
            if order != 0:
                idx = (idx[1], idx[0])
            XY = simplex[idx, :]
            plt.plot(XY[:, 0], XY[:, 1], 'k-',
                     linewidth=3,
                     # color=colors[tag],
                     zorder=2)
            plt.text(XY[:, 0].mean(), XY[:, 1].mean(), str(i))

    def plotMeshOverlap(self, overlap):
        "Plot a single mesh overlap."
        from . meshOverlaps import meshOverlap
        assert isinstance(overlap, meshOverlap)
        import matplotlib.pyplot as plt
        # self.plot(boundary=True)
        self.plot(boundary=False)
        for i in range(overlap.num_vertices):
            v = self.cells[overlap.vertices[i, 0], overlap.vertices[i, 1]]
            plt.text(self.vertices[v, 0], self.vertices[v, 1], str(i))
        for i in range(overlap.num_cells):
            cellNo = overlap.cells[i]
            simplex = self.vertices_as_array[self.cells[cellNo, :], :]
            XY = simplex.mean(axis=0)
            plt.text(XY[0], XY[1], str(i))
        plt.title('Overlap of subdomain {} with {}'.format(overlap.mySubdomainNo, overlap.otherSubdomainNo))

    def plotOverlapManager(self, overlap):
        "Plot all mesh overlaps in an overlap manager."
        from . meshOverlaps import overlapManager
        assert isinstance(overlap, overlapManager)
        import matplotlib.pyplot as plt
        self.plot()
        x = np.zeros((self.num_cells), dtype=REAL)
        for subdomain in overlap.overlaps:
            for cellNo in overlap.overlaps[subdomain].cells:
                x[cellNo] += subdomain+1
        plt.tripcolor(self.vertices[:, 0], self.vertices[:, 1],
                      self.cells, x)
        plt.axis('equal')

    def plotAlgebraicOverlap(self, DoFMap, overlap):
        "Plot a single algebraic overlap."
        from . algebraicOverlaps import algebraicOverlap
        assert isinstance(overlap, algebraicOverlap)
        import matplotlib.pyplot as plt
        self.plot(boundary=True)
        dofDict = {}
        for i, dof in enumerate(overlap.shared_dofs):
            dofDict[dof] = i
        for cellNo in range(self.num_cells):
            simplex = self.vertices_as_array[self.cells[cellNo, :], :]
            for i in range(DoFMap.dofs_per_element):
                dof = DoFMap.cell2dof_py(cellNo, i)
                try:
                    pos = np.dot(DoFMap.nodes[i, :], simplex)
                    plt.text(pos[0], pos[1], str(dofDict[dof]))
                except:
                    pass

    def plotAlgebraicOverlapManager(self, DoFMap, overlap):
        from . algebraicOverlaps import algebraicOverlapManager
        assert isinstance(overlap, algebraicOverlapManager)
        self.plot(boundary=True)
        x = np.zeros((DoFMap.num_dofs), dtype=REAL)
        for subdomainNo in overlap.overlaps:
            for i, dof in enumerate(overlap.overlaps[subdomainNo].shared_dofs):
                x[dof] += 1
        self.plotFunctionDoFMap(DoFMap, x)

    def plotVertexPartitions(self, numPartitions, partitioner='metis',
                             interior=False, padding=0.1):
        import matplotlib.pyplot as plt
        if isinstance(partitioner, str):
            if partitioner == 'metis':
                partitioner = metisMeshPartitioner(self)
            elif partitioner == 'regular':
                partitioner = regularMeshPartitioner(self)
            else:
                raise NotImplementedError()
            part, numPartitions = partitioner.partitionVertices(numPartitions,
                                                                interior)
        elif isinstance(partitioner, sparseGraph):
            part = np.zeros((partitioner.nnz))
            for p in range(partitioner.num_rows):
                for jj in range(partitioner.indptr[p], partitioner.indptr[p+1]):
                    part[partitioner.indices[jj]] = p
            numPartitions = partitioner.shape[0]
        else:
            raise NotImplementedError()
        self.plot()
        cm = plt.get_cmap('gist_rainbow')
        X, Y = self.vertices[:, 0], self.vertices[:, 1]
        lenX = X.max()-X.min()
        lenY = Y.max()-Y.min()
        plt.axis('equal')
        plt.xlim([X.min()-lenX*padding, X.max()+lenX*padding])
        plt.ylim([Y.min()-lenY*padding, Y.max()+lenY*padding])
        if not X.shape[0] == part.shape[0]:
            part2 = -np.ones((X.shape[0]))
            part2[self.interiorVertices] = part
            part = part2
        for i in range(numPartitions):
            plt.tricontourf(X, Y,
                            part == i,
                            levels=[0.7, 1.1],
                            colors=[cm(i/numPartitions)])

    def plotCellPartitions(self, numPartitions, partitioner='metis'):
        import matplotlib.pyplot as plt
        if isinstance(partitioner, str):
            if partitioner == 'metis':
                partitioner = metisMeshPartitioner(self)
            elif partitioner == 'regular':
                partitioner = regularMeshPartitioner(self)
            else:
                raise NotImplementedError()
        part, numPartitions = partitioner.partitionCells(numPartitions)
        plt.tripcolor(self.vertices[:, 0], self.vertices[:, 1],
                      self.cells, part)
        plt.triplot(self.vertices[:, 0], self.vertices[:, 1],
                    self.cells, '-', zorder=1)

    def plotGraph(self, A, dofmap):
        from PyNucleus_base.linear_operators import CSR_LinearOperator
        import matplotlib.pyplot as plt
        assert isinstance(A, CSR_LinearOperator)
        for cellNo in range(self.num_cells):
            simplex = self.vertices[self.cells[cellNo, :], :]
            coords = dofmap.getNodalCoordinates_py(simplex)
            dofs = []
            for j in range(dofmap.dofs_per_element):
                dofs.append(dofmap.cell2dof_py(cellNo, j))
            for i, dof1 in enumerate(dofs):
                if dof1 < 0:
                    continue
                for j, dof2 in enumerate(dofs):
                    if dof2 < 0:
                        continue
                    if A.getEntry_py(dof1, dof2) != 0.:
                        if i == j:
                            plt.plot([coords[i, 0], coords[j, 0]],
                                     [coords[i, 1], coords[j, 1]],
                                     marker='o',
                                     ms=8,
                                     c='r', lw=4)
                        else:
                            plt.plot([coords[i, 0], coords[j, 0]],
                                     [coords[i, 1], coords[j, 1]],
                                     c='g', lw=4)

    def sortVertices(self):
        idx = np.argsort(self.vertices_as_array.view('d,d'), order=['f1', 'f0'], axis=0).flat[:self.vertices.shape[0]]
        self.reorderVertices(idx)


class mesh3d(meshNd):
    """
    3D mesh

    Attributes:
    vertices
    cells
    boundaryVertices
    boundaryEdges
    boundaryFaces
    boundaryVertexTags
    boundaryEdgeTags
    boundaryFaceTags
    """

    def plot(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from itertools import combinations
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(self.cells.shape[0]):
            for j, k in combinations(range(4), 2):
                u = self.vertices[self.cells[i, j], :]
                v = self.vertices[self.cells[i, k], :]
                ax.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]], 'k')

    def plot_surface(self, boundary=False):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        # from matplotlib import rcParams
        # from itertools import combinations
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # for i in range(self.boundaryFaces.shape[0]):
        #     for j, k in combinations(range(3), 2):
        #         u = self.vertices[self.boundaryFaces[i, j], :]
        #         v = self.vertices[self.boundaryFaces[i, k], :]
        #         ax.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]], 'k', zorder=-1)
        tags = set(self.boundaryFaceTags)
        tags = tags.union(self.boundaryEdgeTags)
        tags = tags.union(self.boundaryVertexTags)
        cm = plt.get_cmap('gist_rainbow')
        num_colors = len(tags)
        colors = {tag: cm(i/num_colors) for i, tag in enumerate(tags)}
        tri = Poly3DCollection([self.vertices_as_array[self.boundaryFaces[i, :], :]
                                for i in range(self.boundaryFaces.shape[0])],
                               facecolors=[colors[t] for t in self.boundaryFaceTags],
                               edgecolors=(0, 0, 0, 1), lw=1)
        ax.add_collection3d(tri)
        if boundary:
            scatterDict = {}
            for bv, tag in zip(self.boundaryVertices, self.boundaryVertexTags):
                XY = self.vertices[bv, :]
                try:
                    scatterDict[tag].append(XY)
                except KeyError:
                    scatterDict[tag] = [XY]
            for tag in scatterDict:
                XY = np.vstack(scatterDict[tag])
                print(XY.shape, colors[tag])
                plt.scatter(XY[:, 0], XY[:, 1], zs=XY[:, 2],
                            s=100,
                            c=colors[tag],
                            zorder=3,
                            depthshade=False)
            # for be, tag in zip(self.boundaryEdges, self.boundaryEdgeTags):
            #     XY = self.vertices[be, :]
            #     plt.plot(XY[:, 0], XY[:, 1], 'k-', zs=XY[:, 2],
            #              linewidth=3*rcParams["lines.linewidth"],
            #              color=colors[tag],
            #              zorder=2)

    def plotVTK(self, boundary=False, opacity=1.0):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        import matplotlib.pyplot as plt

        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(self.vertices, deep=1))

        cm = plt.get_cmap('gist_rainbow')
        tags = set(self.boundaryFaceTags)
        tags = tags.union(self.boundaryEdgeTags)
        tags = tags.union(self.boundaryVertexTags)
        num_colors = len(tags)
        ccs = {tag: cm(i/num_colors) for i, tag in enumerate(tags)}

        if boundary:
            toPlot = [
                (self.boundaryFaces, self.boundaryFaceTags),
                (self.boundaryEdges, self.boundaryEdgeTags),
                (self.boundaryVertices[:, np.newaxis], self.boundaryVertexTags)
            ]
        else:
            toPlot = [(self.cells, np.zeros((self.num_cells), dtype=TAG))]

        colors = vtk.vtkUnsignedCharArray()
        colors.SetName("Colors")
        colors.SetNumberOfComponents(3)
        colors.SetNumberOfTuples(sum([cells.shape[0] for cells, _ in toPlot]))
        myCells = []
        myCellTypes = []
        numCells = 0
        k = 0
        for cells, tags in toPlot:
            if cells.shape[1] == 1:
                cellType = vtk.VTK_VERTEX
            elif cells.shape[1] == 2:
                cellType = vtk.VTK_LINE
            elif cells.shape[1] == 3:
                cellType = vtk.VTK_TRIANGLE
            elif cells.shape[1] == 4:
                cellType = vtk.VTK_TETRA
            else:
                raise NotImplementedError()
            myCellTypes.append(cellType*np.ones((cells.shape[0]), dtype=np.int))
            myCells.append(np.hstack((cells.shape[1]*np.ones((cells.shape[0], 1), dtype=np.int64),
                                      cells.astype(np.int64))).ravel())
            numCells += cells.shape[0]
            for i in range(cells.shape[0]):
                c = ccs[tags[i]]
                colors.InsertTuple3(k, 255*c[0], 255*c[1], 255*c[2])
                k += 1
        c3 = np.concatenate(myCells)
        c2 = numpy_to_vtkIdTypeArray(c3, deep=1)
        c = vtk.vtkCellArray()
        c.SetCells(numCells, c2)

        ugrid = vtk.vtkUnstructuredGrid()
        cellTypes = np.concatenate(myCellTypes)
        ugrid.SetCells(cellTypes, c)
        ugrid.SetPoints(points)
        ugrid.GetCellData().SetScalars(colors)

        ugridMapper = vtk.vtkDataSetMapper()
        ugridMapper.SetInputData(ugrid)

        ugridActor = vtk.vtkActor()
        ugridActor.SetMapper(ugridMapper)
        if not boundary:
            ugridActor.GetProperty().EdgeVisibilityOn()
        else:
            ugridActor.GetProperty().SetLineWidth(10)
            ugridActor.GetProperty().SetPointSize(30)
        ugridActor.GetProperty().SetOpacity(opacity)

        return ugridActor

    def plotInterfaceVTK(self, interface):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        from . meshOverlaps import sharedMesh, simplexMapper3D
        assert isinstance(interface, sharedMesh)

        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(self.vertices, deep=1))

        sM = simplexMapper3D(self)

        cellsCells = self.cells[interface.cells, :]
        cellsFaces = uninitialized((interface.num_faces, 3), dtype=INDEX)
        for i in range(interface.num_faces):
            cellsFaces[i, :] = sM.getFaceInCell_py(interface.faces[i, 0],
                                                   interface.faces[i, 1])
        cellsEdges = uninitialized((interface.num_edges, 2), dtype=INDEX)
        for i in range(interface.num_edges):
            cellsEdges[i, :] = sM.getEdgeInCell_py(interface.edges[i, 0],
                                                   interface.edges[i, 1])
        cellsVertices = uninitialized((interface.num_vertices, 1), dtype=INDEX)
        for i in range(interface.num_vertices):
            cellsVertices[i, 0] = sM.getVertexInCell_py(interface.vertices[i, 0],
                                                        interface.vertices[i, 1])

        toPlot = [
            cellsCells, cellsFaces, cellsEdges, cellsVertices
        ]

        myCells = []
        myCellTypes = []
        numCells = 0
        for cells in toPlot:
            if cells.shape[1] == 1:
                cellType = vtk.VTK_VERTEX
            elif cells.shape[1] == 2:
                cellType = vtk.VTK_LINE
            elif cells.shape[1] == 3:
                cellType = vtk.VTK_TRIANGLE
            elif cells.shape[1] == 4:
                cellType = vtk.VTK_TETRA
            else:
                raise NotImplementedError()
            myCellTypes.append(cellType*np.ones((cells.shape[0]), dtype=np.int))
            myCells.append(np.hstack((cells.shape[1]*np.ones((cells.shape[0], 1), dtype=np.int64),
                                      cells.astype(np.int64))).ravel())
            numCells += cells.shape[0]
        c3 = np.concatenate(myCells)
        c2 = numpy_to_vtkIdTypeArray(c3, deep=1)
        c = vtk.vtkCellArray()
        c.SetCells(numCells, c2)

        ugrid = vtk.vtkUnstructuredGrid()
        cellTypes = np.concatenate(myCellTypes)
        ugrid.SetCells(cellTypes, c)
        ugrid.SetPoints(points)

        ugridMapper = vtk.vtkDataSetMapper()
        ugridMapper.SetInputData(ugrid)

        ugridActor = vtk.vtkActor()
        ugridActor.SetMapper(ugridMapper)
        ugridActor.GetProperty().SetLineWidth(10)
        ugridActor.GetProperty().SetPointSize(30)

        return ugridActor

    def checkDoFMap(self, DoFMap):
        "Plot the DoF numbers on the mesh."
        recorderdDofs = {}
        for cellNo in range(self.num_cells):
            simplex = self.vertices[self.cells[cellNo, :], :]
            for i in range(DoFMap.dofs_per_element):
                dof = DoFMap.cell2dof_py(cellNo, i)
                if dof >= 0:
                    pos = np.dot(DoFMap.nodes[i, :], simplex)
                    try:
                        posOld = recorderdDofs[dof]
                        assert np.allclose(pos, posOld)
                    except KeyError:
                        recorderdDofs[dof] = pos
        return recorderdDofs

    def sortVertices(self):
        idx = np.argsort(self.vertices_as_array.view('d,d,d'), order=['f2', 'f1', 'f0'], axis=0).flat[:self.vertices.shape[0]]
        self.reorderVertices(idx)


def stitchSubdomains(subdomains, overlapManagers, returnR=False, ncs=None):
    """
    Stitch subdomains together.
    Works for 2D.
    """
    vertices = uninitialized((0, subdomains[0].dim), dtype=INDEX)
    cells = uninitialized((0, subdomains[0].dim+1), dtype=INDEX)
    globalIndices = []
    numPartitions = len(subdomains)
    # FIX: If we have real overlap (overlapping elements, not vertices),
    #      I'm adding vertices twice
    for i in range(numPartitions):
        if ncs:
            subdomainVertices = subdomains[i].vertices[:ncs[i][0], :]
        else:
            subdomainVertices = subdomains[i].vertices
        subdomainNumVertices = subdomainVertices.shape[0]
        # form vector bv
        # if vertex is in previous subdomains, set number of a subdomain, else -1
        bv = -1*np.ones(subdomainNumVertices, dtype=INDEX)
        # loop over all overlaps with subdomains that we already incorporated
        # for j in range(i-1, -1, -1):
        for j in range(i):
            bv[np.array(overlapManagers[i][j].overlap2local, dtype=INDEX)] = j
        # append all new vertices
        k = len(vertices)
        nv = (bv == -1)
        vertices = np.vstack((vertices,
                              np.compress(nv, subdomainVertices, axis=0)))

        # find new indices after discarding of the known vertices
        globalIndicesSubdomain = uninitialized(subdomainNumVertices,
                                               dtype=INDEX)
        globalIndicesSubdomain[nv] = np.arange(k, k+nv.sum())

        for j in np.compress(np.logical_not(nv),
                             np.arange(subdomainNumVertices)):
            otherSubdomain = bv[j]
            # translate to overlap index in domain i
            m = overlapManagers[i].translate_local_overlap(otherSubdomain,
                                                           np.array([j]))
            # translate to local index in domain otherSubdomain
            m = overlapManagers[otherSubdomain].translate_overlap_local(i, m)
            # translate to global index
            globalIndicesSubdomain[j] = globalIndices[otherSubdomain][m]
        globalIndices.append(globalIndicesSubdomain)

        if ncs:
            subdomainCells = subdomains[i].cells[:ncs[i][1], :]
            addCell = np.ones(subdomainCells.shape[0], dtype=np.bool)
        else:
            subdomainCells = subdomains[i].cells
            # translate cells to new indices
            # get subdomain number for every vertex in every cell
            ww = np.take(bv, subdomainCells.T)
            # take cell wise min
            cellMinSubdomain = ww.min(axis=0)
            # take cell wise max
            cellMaxSubdomain = ww.max(axis=0)
            # only take cells that have at least one new vertex,
            # or that have vertices on different subdomains
            # FIX: the last condition is not obvious
            addCell = np.logical_or(cellMinSubdomain == -1,
                                    np.logical_and(cellMinSubdomain > -1,
                                                   cellMinSubdomain < cellMaxSubdomain))
            # addCell = cellMinSubdomain == -1
            # xx = np.logical_and(cellMinSubdomain > -1,
            #                 cellMinSubdomain < cellMaxSubdomain)
            # print(i, xx.sum())
            # print(ww[:, xx])
        s = (addCell.sum(), subdomains[0].dim+1)
        newcells = np.compress(addCell, subdomainCells, axis=0)
        newcells = globalIndicesSubdomain[newcells.ravel()].reshape(s)
        cells = np.vstack((cells, newcells))
    if subdomains[0].dim == 1:
        mesh = mesh1d(vertices, cells)
    elif subdomains[0].dim == 2:
        mesh = mesh2d(vertices, cells)
    elif subdomains[0].dim == 3:
        mesh = mesh3d(vertices, cells)
    if returnR:
        return (mesh, globalIndices)
    else:
        return mesh


def stitchOverlappingMeshes(meshes, overlapManagers):
    dim = meshes[0].dim
    global_vertices = uninitialized((0, dim), dtype=REAL)
    global_cells = uninitialized((0, dim+1), dtype=INDEX)
    global_boundary_vertices = {}
    global_boundary_edges = {}
    numPartitions = len(meshes)
    localCellLookup = {}
    globalCellLookup = []
    for mySubdomainNo in range(numPartitions):
        translate = -np.ones((meshes[mySubdomainNo].num_vertices), dtype=INDEX)
        idx = np.ones((meshes[mySubdomainNo].cells.shape[0]), dtype=np.bool)
        lookup = -np.ones((meshes[mySubdomainNo].num_cells), dtype=INDEX)
        for otherSubdomainNo in range(mySubdomainNo):
            if otherSubdomainNo not in overlapManagers[mySubdomainNo].overlaps:
                continue
            idx[overlapManagers[mySubdomainNo].overlaps[otherSubdomainNo].cells] = False
            for k in range(overlapManagers[mySubdomainNo].overlaps[otherSubdomainNo].cells.shape[0]):
                p = overlapManagers[mySubdomainNo].overlaps[otherSubdomainNo].cells[k]
                q = overlapManagers[otherSubdomainNo].overlaps[mySubdomainNo].cells[k]
                translate[meshes[mySubdomainNo].cells_as_array[p, :]] = meshes[otherSubdomainNo].cells_as_array[q, :]
                lookup[p] = globalCellLookup[otherSubdomainNo][q]
        # get global vertex indices
        numVertices = numVerticesNew = global_vertices.shape[0]
        for k in range(meshes[mySubdomainNo].num_vertices):
            if translate[k] == -1:
                translate[k] = numVerticesNew
                numVerticesNew += 1
        # translate vertex indices in cells to global indices
        for k in range(meshes[mySubdomainNo].num_cells):
            for m in range(dim+1):
                meshes[mySubdomainNo].cells[k, m] = translate[meshes[mySubdomainNo].cells[k, m]]
        global_vertices = np.vstack((global_vertices,
                                     meshes[mySubdomainNo].vertices_as_array[translate >= numVertices, :]))
        num_cells = global_cells.shape[0]
        global_cells = np.vstack((global_cells,
                                  meshes[mySubdomainNo].cells_as_array[idx, :]))

        for vertexNo in range(meshes[mySubdomainNo].boundaryVertices.shape[0]):
            v = translate[meshes[mySubdomainNo].boundaryVertices[vertexNo]]
            try:
                global_boundary_vertices[v].append(meshes[mySubdomainNo].boundaryVertexTags[vertexNo])
            except KeyError:
                global_boundary_vertices[v] = [meshes[mySubdomainNo].boundaryVertexTags[vertexNo]]

        for edgeNo in range(meshes[mySubdomainNo].boundaryEdges.shape[0]):
            e = (translate[meshes[mySubdomainNo].boundaryEdges[edgeNo, 0]],
                 translate[meshes[mySubdomainNo].boundaryEdges[edgeNo, 1]])
            try:
                global_boundary_edges[e].append(meshes[mySubdomainNo].boundaryEdgeTags[edgeNo])
            except KeyError:
                global_boundary_edges[e] = [meshes[mySubdomainNo].boundaryEdgeTags[edgeNo]]

        for k in range(meshes[mySubdomainNo].num_cells):
            if idx[k]:
                localCellLookup[num_cells] = [(mySubdomainNo, k)]
                lookup[k] = num_cells
                num_cells += 1
            else:
                localCellLookup[lookup[k]].append((mySubdomainNo, k))
        globalCellLookup.append(lookup)
    if dim == 1:
        global_mesh = mesh1d(global_vertices, global_cells)
    elif dim == 2:
        global_mesh = mesh2d(global_vertices, global_cells)
    else:
        raise NotImplementedError()
    boundaryVertices = uninitialized((len(global_boundary_vertices)), dtype=INDEX)
    boundaryVertexTags = uninitialized((len(global_boundary_vertices)), dtype=TAG)
    for vertexNo, vertex in enumerate(global_boundary_vertices):
        boundaryVertices[vertexNo] = vertex
        global_boundary_vertices[vertex] = list(set(global_boundary_vertices[vertex]))
        boundaryVertexTags[vertexNo] = max(global_boundary_vertices[vertex])
    global_mesh._boundaryVertices = boundaryVertices
    global_mesh._boundaryVertexTags = boundaryVertexTags

    boundaryEdges = uninitialized((len(global_boundary_edges), 2), dtype=INDEX)
    boundaryEdgeTags = uninitialized((len(global_boundary_edges)), dtype=TAG)
    for edgeNo, edge in enumerate(global_boundary_edges):
        boundaryEdges[edgeNo, :] = edge
        global_boundary_edges[edge] = list(set(global_boundary_edges[edge]))
        # assert len(global_boundary_edges[edge]) == 1, global_boundary_edges[edge]
        boundaryEdgeTags[edgeNo] = max(global_boundary_edges[edge])
    global_mesh._boundaryEdges = boundaryEdges
    global_mesh._boundaryEdgeTags = boundaryEdgeTags
    return global_mesh, localCellLookup


def stitchNonoverlappingMeshes(meshes, interfaceManagers):
    global_vertices = uninitialized((0, meshes[0].dim), dtype=REAL)
    global_cells = uninitialized((0, meshes[0].dim+1), dtype=INDEX)
    numPartitions = len(meshes)
    localCellLookup = {}
    global_boundary_vertices = {}
    global_boundary_edges = {}
    global_boundary_faces = {}
    for mySubdomainNo in range(numPartitions):
        translate = -np.ones((meshes[mySubdomainNo].num_vertices), dtype=INDEX)
        for otherSubdomainNo in range(mySubdomainNo):
            if otherSubdomainNo not in interfaceManagers[mySubdomainNo].interfaces:
                continue
            # idx[interfaceManagers[mySubdomainNo].overlaps[otherSubdomainNo].cells] = False
            for k in range(interfaceManagers[mySubdomainNo].interfaces[otherSubdomainNo].vertices.shape[0]):
                cellNo = interfaceManagers[mySubdomainNo].interfaces[otherSubdomainNo].vertices[k, 0]
                vertexNo = interfaceManagers[mySubdomainNo].interfaces[otherSubdomainNo].vertices[k, 1]
                p = meshes[mySubdomainNo].cells[cellNo, vertexNo]
                cellNo = interfaceManagers[otherSubdomainNo].interfaces[mySubdomainNo].vertices[k, 0]
                vertexNo = interfaceManagers[otherSubdomainNo].interfaces[mySubdomainNo].vertices[k, 1]
                q = meshes[otherSubdomainNo].cells[cellNo, vertexNo]
                translate[p] = q
            if meshes[0].dim >= 2:
                for k in range(interfaceManagers[mySubdomainNo].interfaces[otherSubdomainNo].edges.shape[0]):
                    cellNo = interfaceManagers[mySubdomainNo].interfaces[otherSubdomainNo].edges[k, 0]
                    edgeNo = interfaceManagers[mySubdomainNo].interfaces[otherSubdomainNo].edges[k, 1]
                    order = interfaceManagers[mySubdomainNo].interfaces[otherSubdomainNo].edges[k, 2]
                    if edgeNo == 0:
                        vertexNo1, vertexNo2 = 0, 1
                    elif edgeNo == 1:
                        vertexNo1, vertexNo2 = 1, 2
                    elif edgeNo == 2:
                        vertexNo1, vertexNo2 = 2, 0
                    elif edgeNo == 3:
                        vertexNo1, vertexNo2 = 0, 3
                    elif edgeNo == 4:
                        vertexNo1, vertexNo2 = 1, 3
                    else:
                        vertexNo1, vertexNo2 = 2, 3
                    if order == 1:
                        vertexNo1, vertexNo2 = vertexNo2, vertexNo1
                    p1 = meshes[mySubdomainNo].cells[cellNo, vertexNo1]
                    p2 = meshes[mySubdomainNo].cells[cellNo, vertexNo2]

                    cellNo = interfaceManagers[otherSubdomainNo].interfaces[mySubdomainNo].edges[k, 0]
                    edgeNo = interfaceManagers[otherSubdomainNo].interfaces[mySubdomainNo].edges[k, 1]
                    order = interfaceManagers[otherSubdomainNo].interfaces[mySubdomainNo].edges[k, 2]
                    if edgeNo == 0:
                        vertexNo1, vertexNo2 = 0, 1
                    elif edgeNo == 1:
                        vertexNo1, vertexNo2 = 1, 2
                    elif edgeNo == 2:
                        vertexNo1, vertexNo2 = 2, 0
                    elif edgeNo == 3:
                        vertexNo1, vertexNo2 = 0, 3
                    elif edgeNo == 4:
                        vertexNo1, vertexNo2 = 1, 3
                    else:
                        vertexNo1, vertexNo2 = 2, 3
                    if order == 1:
                        vertexNo1, vertexNo2 = vertexNo2, vertexNo1
                    q1 = meshes[otherSubdomainNo].cells[cellNo, vertexNo1]
                    q2 = meshes[otherSubdomainNo].cells[cellNo, vertexNo2]

                    translate[p1] = q1
                    translate[p2] = q2
            # missing faces here
            if meshes[0].dim >= 3:
                for k in range(interfaceManagers[mySubdomainNo].interfaces[otherSubdomainNo].faces.shape[0]):
                    cellNo = interfaceManagers[mySubdomainNo].interfaces[otherSubdomainNo].faces[k, 0]
                    faceNo = interfaceManagers[mySubdomainNo].interfaces[otherSubdomainNo].faces[k, 1]
                    order = interfaceManagers[mySubdomainNo].interfaces[otherSubdomainNo].faces[k, 2]

                    if faceNo == 0:
                        vertexNo1, vertexNo2, vertexNo3 = 0, 2, 1
                        # edgeNo1, edgeNo2, edgeNo3 = 2, 1, 0
                    elif faceNo == 1:
                        vertexNo1, vertexNo2, vertexNo3 = 0, 1, 3
                        # edgeNo1, edgeNo2, edgeNo3 = 0, 4, 3
                    elif faceNo == 2:
                        vertexNo1, vertexNo2, vertexNo3 = 1, 2, 3
                        # edgeNo1, edgeNo2, edgeNo3 = 1, 5, 4
                    else:
                        vertexNo1, vertexNo2, vertexNo3 = 2, 0, 3
                        # edgeNo1, edgeNo2, edgeNo3 = 2, 3, 5

                    if order == 1:
                        vertexNo1, vertexNo2, vertexNo3 = vertexNo2, vertexNo3, vertexNo1
                        # edgeNo1, edgeNo2, edgeNo3 = edgeNo2, edgeNo3, edgeNo1
                    elif order == 2:
                        vertexNo1, vertexNo2, vertexNo3 = vertexNo3, vertexNo1, vertexNo2
                        # edgeNo1, edgeNo2, edgeNo3 = edgeNo3, edgeNo1, edgeNo2
                    elif order == -1:
                        vertexNo1, vertexNo2, vertexNo3 = vertexNo2, vertexNo1, vertexNo3
                        # edgeNo1, edgeNo2, edgeNo3 = edgeNo1, edgeNo3, edgeNo2
                    elif order == -2:
                        vertexNo1, vertexNo2, vertexNo3 = vertexNo1, vertexNo3, vertexNo2
                        # edgeNo1, edgeNo2, edgeNo3 = edgeNo3, edgeNo2, edgeNo1
                    elif order == -3:
                        vertexNo1, vertexNo2, vertexNo3 = vertexNo3, vertexNo2, vertexNo1
                        # edgeNo1, edgeNo2, edgeNo3 = edgeNo2, edgeNo1, edgeNo3

                    p1 = meshes[mySubdomainNo].cells[cellNo, vertexNo1]
                    p2 = meshes[mySubdomainNo].cells[cellNo, vertexNo2]
                    p3 = meshes[mySubdomainNo].cells[cellNo, vertexNo3]

                    cellNo = interfaceManagers[otherSubdomainNo].interfaces[mySubdomainNo].faces[k, 0]
                    faceNo = interfaceManagers[otherSubdomainNo].interfaces[mySubdomainNo].faces[k, 1]
                    order = interfaceManagers[otherSubdomainNo].interfaces[mySubdomainNo].faces[k, 2]

                    if faceNo == 0:
                        vertexNo1, vertexNo2, vertexNo3 = 0, 2, 1
                        # edgeNo1, edgeNo2, edgeNo3 = 2, 1, 0
                    elif faceNo == 1:
                        vertexNo1, vertexNo2, vertexNo3 = 0, 1, 3
                        # edgeNo1, edgeNo2, edgeNo3 = 0, 4, 3
                    elif faceNo == 2:
                        vertexNo1, vertexNo2, vertexNo3 = 1, 2, 3
                        # edgeNo1, edgeNo2, edgeNo3 = 1, 5, 4
                    else:
                        vertexNo1, vertexNo2, vertexNo3 = 2, 0, 3
                        # edgeNo1, edgeNo2, edgeNo3 = 2, 3, 5

                    if order == 1:
                        vertexNo1, vertexNo2, vertexNo3 = vertexNo2, vertexNo3, vertexNo1
                        # edgeNo1, edgeNo2, edgeNo3 = edgeNo2, edgeNo3, edgeNo1
                    elif order == 2:
                        vertexNo1, vertexNo2, vertexNo3 = vertexNo3, vertexNo1, vertexNo2
                        # edgeNo1, edgeNo2, edgeNo3 = edgeNo3, edgeNo1, edgeNo2
                    elif order == -1:
                        vertexNo1, vertexNo2, vertexNo3 = vertexNo2, vertexNo1, vertexNo3
                        # edgeNo1, edgeNo2, edgeNo3 = edgeNo1, edgeNo3, edgeNo2
                    elif order == -2:
                        vertexNo1, vertexNo2, vertexNo3 = vertexNo1, vertexNo3, vertexNo2
                        # edgeNo1, edgeNo2, edgeNo3 = edgeNo3, edgeNo2, edgeNo1
                    elif order == -3:
                        vertexNo1, vertexNo2, vertexNo3 = vertexNo3, vertexNo2, vertexNo1
                        # edgeNo1, edgeNo2, edgeNo3 = edgeNo2, edgeNo1, edgeNo3

                    q1 = meshes[otherSubdomainNo].cells[cellNo, vertexNo1]
                    q2 = meshes[otherSubdomainNo].cells[cellNo, vertexNo2]
                    q3 = meshes[otherSubdomainNo].cells[cellNo, vertexNo3]

                    translate[p1] = q1
                    translate[p2] = q2
                    translate[p3] = q3

        numVertices = numVerticesNew = global_vertices.shape[0]
        for k in range(meshes[mySubdomainNo].num_vertices):
            if translate[k] == -1:
                translate[k] = numVerticesNew
                numVerticesNew += 1
        for k in range(meshes[mySubdomainNo].num_cells):
            for m in range(meshes[mySubdomainNo].dim+1):
                meshes[mySubdomainNo].cells[k, m] = translate[meshes[mySubdomainNo].cells[k, m]]
        global_vertices = np.vstack((global_vertices,
                                     meshes[mySubdomainNo].vertices_as_array[translate >= numVertices, :]))
        num_cells = global_cells.shape[0]
        global_cells = np.vstack((global_cells,
                                  meshes[mySubdomainNo].cells))

        # add boundary vertices to global mesh
        for boundaryVertexNo in range(meshes[mySubdomainNo].boundaryVertices.shape[0]):
            vertexNo = meshes[mySubdomainNo].boundaryVertices[boundaryVertexNo]
            v = translate[vertexNo]
            try:
                global_boundary_vertices[v].append(meshes[mySubdomainNo].boundaryVertexTags[boundaryVertexNo])
            except KeyError:
                global_boundary_vertices[v] = [meshes[mySubdomainNo].boundaryVertexTags[boundaryVertexNo]]

        # add boundary edges to global mesh
        for edgeNo in range(meshes[mySubdomainNo].boundaryEdges.shape[0]):
            e = (translate[meshes[mySubdomainNo].boundaryEdges[edgeNo, 0]],
                 translate[meshes[mySubdomainNo].boundaryEdges[edgeNo, 1]])
            try:
                global_boundary_edges[e].append(meshes[mySubdomainNo].boundaryEdgeTags[edgeNo])
            except KeyError:
                global_boundary_edges[e] = [meshes[mySubdomainNo].boundaryEdgeTags[edgeNo]]

        # add boundary faces to global mesh
        for faceNo in range(meshes[mySubdomainNo].boundaryFaces.shape[0]):
            e = (translate[meshes[mySubdomainNo].boundaryFaces[faceNo, 0]],
                 translate[meshes[mySubdomainNo].boundaryFaces[faceNo, 1]],
                 translate[meshes[mySubdomainNo].boundaryFaces[faceNo, 2]])
            try:
                global_boundary_faces[e].append(meshes[mySubdomainNo].boundaryFaceTags[faceNo])
            except KeyError:
                global_boundary_faces[e] = [meshes[mySubdomainNo].boundaryFaceTags[faceNo]]

        for k in range(meshes[mySubdomainNo].num_cells):
            localCellLookup[num_cells] = [(mySubdomainNo, k)]
            num_cells += 1
    if meshes[0].dim == 1:
        global_mesh = mesh1d(global_vertices, global_cells)
    elif meshes[0].dim == 2:
        global_mesh = mesh2d(global_vertices, global_cells)
    elif meshes[0].dim == 3:
        global_mesh = mesh3d(global_vertices, global_cells)

    boundaryVertices = uninitialized((len(global_boundary_vertices)), dtype=INDEX)
    boundaryVertexTags = uninitialized((len(global_boundary_vertices)), dtype=TAG)
    for vertexNo, vertex in enumerate(global_boundary_vertices):
        boundaryVertices[vertexNo] = vertex
        global_boundary_vertices[vertex] = list(set(global_boundary_vertices[vertex]))
        boundaryVertexTags[vertexNo] = max(global_boundary_vertices[vertex])
    global_mesh._boundaryVertices = boundaryVertices
    global_mesh._boundaryVertexTags = boundaryVertexTags

    if meshes[0].dim >= 2:
        boundaryEdges = uninitialized((len(global_boundary_edges), 2), dtype=INDEX)
        boundaryEdgeTags = uninitialized((len(global_boundary_edges)), dtype=TAG)
        for edgeNo, edge in enumerate(global_boundary_edges):
            boundaryEdges[edgeNo, :] = edge
            global_boundary_edges[edge] = list(set(global_boundary_edges[edge]))
            assert len(global_boundary_edges[edge]) == 1, global_boundary_edges[edge]
            boundaryEdgeTags[edgeNo] = global_boundary_edges[edge][0]
        global_mesh._boundaryEdges = boundaryEdges
        global_mesh._boundaryEdgeTags = boundaryEdgeTags

    if meshes[0].dim >= 3:
        boundaryFaces = uninitialized((len(global_boundary_faces), 3), dtype=INDEX)
        boundaryFaceTags = uninitialized((len(global_boundary_faces)), dtype=TAG)
        for faceNo, face in enumerate(global_boundary_faces):
            boundaryFaces[faceNo, :] = face
            global_boundary_faces[face] = list(set(global_boundary_faces[face]))
            assert len(global_boundary_faces[face]) == 1, global_boundary_faces[face]
            boundaryFaceTags[faceNo] = global_boundary_faces[face][0]
        global_mesh._boundaryFaces = boundaryFaces
        global_mesh._boundaryFaceTags = boundaryFaceTags

    return global_mesh, localCellLookup


def stitchSolutions(global_mesh, DoFMaps, localCellLookup, solutions, tag=0):
    from . DoFMaps import getAvailableDoFMaps, str2DoFMap
    for element in getAvailableDoFMaps():
        DoFMap = str2DoFMap(element)
        if isinstance(DoFMaps[0], DoFMap):
            dm_global = DoFMap(global_mesh, tag=tag)
            break
    else:
        raise NotImplementedError(DoFMaps[0])
    x = dm_global.empty(dtype=solutions[0].dtype)
    for cellNo in range(global_mesh.num_cells):
        for k in range(dm_global.dofs_per_element):
            dofGlobal = dm_global.cell2dof_py(cellNo, k)
            if dofGlobal >= 0:
                for subdomainNo, localCellNo in localCellLookup[cellNo]:
                    dofLocal = DoFMaps[subdomainNo].cell2dof_py(localCellNo, k)
                    if dofLocal >= 0:
                        x[dofGlobal] = solutions[subdomainNo][dofLocal]
    return x, dm_global


def getMappingToGlobalDoFMap(mesh, meshOverlaps, DoFMap, comm=None, collectRank=0, tag=0):
    meshes = comm.gather(mesh, root=collectRank)
    overlapManagers = comm.gather(meshOverlaps, root=collectRank)
    DoFMaps = comm.gather(DoFMap, root=collectRank)
    if comm.rank == collectRank:
        from . meshOverlaps import interfaceManager, overlapManager
        if isinstance(overlapManagers[0], overlapManager):
            mesh_global, localCellLookup = stitchOverlappingMeshes(meshes, overlapManagers)
        elif isinstance(overlapManagers[0], interfaceManager):
            mesh_global, localCellLookup = stitchNonoverlappingMeshes(meshes, overlapManagers)
        else:
            raise NotImplementedError()
        from . DoFMaps import getAvailableDoFMaps, str2DoFMap
        for element in getAvailableDoFMaps():
            DoFMap = str2DoFMap(element)
            if isinstance(DoFMaps[0], DoFMap):
                dm_global = DoFMap(mesh_global, tag=tag)
                break
        else:
            raise NotImplementedError()
        mappings = [uninitialized((dm.num_dofs), dtype=INDEX) for dm in DoFMaps]
        for cellNo in range(mesh_global.num_cells):
            for k in range(dm_global.dofs_per_element):
                dofGlobal = dm_global.cell2dof_py(cellNo, k)
                if dofGlobal >= 0:
                    for subdomainNo, localCellNo in localCellLookup[cellNo]:
                        dofLocal = DoFMaps[subdomainNo].cell2dof_py(localCellNo, k)
                        if dofLocal >= 0:
                            mappings[subdomainNo][dofLocal] = dofGlobal
        return mesh_global, dm_global, mappings
    else:
        return None, None, None


def accumulate2global(mesh, meshOverlaps, DoFMap, vec,
                      comm=None, collectRank=0, tag=0):
    """
    Send subdomain meshes and solutions to root node, stitch together
    meshes and solution. Assumes that solution is already accumulated.
    """
    if comm is not None and comm.size > 1:
        meshes = comm.gather(mesh, root=collectRank)
        overlapManagers = comm.gather(meshOverlaps, root=collectRank)
        if isinstance(vec, list):
            assert isinstance(DoFMap, list) and len(vec) == len(DoFMap)
            DoFMaps = []
            vecs = []
            for i in range(len(DoFMap)):
                DoFMaps.append(comm.gather(DoFMap[i], root=collectRank))
                vecs.append(comm.gather(vec[i], root=collectRank))
        else:
            DoFMaps = [comm.gather(DoFMap, root=collectRank)]
            vecs = [comm.gather(vec, root=collectRank)]
        if comm.rank == collectRank:
            from . meshOverlaps import interfaceManager, overlapManager
            if isinstance(overlapManagers[0], overlapManager):
                mesh_global, localCellLookup = stitchOverlappingMeshes(meshes, overlapManagers)
            elif isinstance(overlapManagers[0], interfaceManager):
                mesh_global, localCellLookup = stitchNonoverlappingMeshes(meshes, overlapManagers)
            else:
                raise NotImplementedError()
            if vec is not None:
                global_vecs = []
                global_dms = []
                for dms, vectors in zip(DoFMaps, vecs):
                    x, dm_global = stitchSolutions(mesh_global, dms, localCellLookup, vectors, tag)
                    global_vecs.append(x)
                    global_dms.append(dm_global)
                if len(global_vecs) == 1:
                    x = global_vecs[0]
                    dm_global = global_dms[0]
                else:
                    x = global_vecs
                    dm_global = global_dms
            else:
                x, dm_global = None, None
            return mesh_global, x, dm_global
        else:
            return None, None, None
    else:
        if len(vec) == 1:
            vec = vec[0]
            DoFMap = DoFMap[0]
        return mesh, vec, DoFMap


def getGlobalPartitioning(mesh, meshOverlaps, comm, collectRank=0):
    meshes = comm.gather(mesh, root=collectRank)
    overlapManagers = comm.gather(meshOverlaps, root=collectRank)
    if comm.rank == collectRank:
        from . meshOverlaps import interfaceManager, overlapManager
        if isinstance(overlapManagers[0], overlapManager):
            mesh_global, localCellLookup = stitchOverlappingMeshes(meshes, overlapManagers)
        elif isinstance(overlapManagers[0], interfaceManager):
            mesh_global, localCellLookup = stitchNonoverlappingMeshes(meshes, overlapManagers)
        else:
            raise NotImplementedError()
        return mesh_global, localCellLookup
    else:
        return None, None


def getSubSolution(new_mesh, dm, x, selectedCells):
    from . DoFMaps import getAvailableDoFMaps, str2DoFMap
    for element in getAvailableDoFMaps():
        DoFMap = str2DoFMap(element)
        if isinstance(dm, DoFMap):
            dmSub = DoFMap(new_mesh, tag=-1)
            break
    else:
        raise NotImplementedError()
    y = np.zeros((dmSub.num_dofs), dtype=REAL)
    for cellSub, cellGlobal in enumerate(selectedCells):
        for k in range(dmSub.dofs_per_element):
            dofSub = dmSub.cell2dof_py(cellSub, k)
            dofGlobal = dm.cell2dof_py(cellGlobal, k)
            if dofSub >= 0 and dofGlobal >= 0:
                y[dofSub] = x[dofGlobal]
    return dmSub, y


def getSubMeshSolution(mesh, DoFMap, solution, selectedCells):
    from . meshCy import getSubmesh
    new_mesh = getSubmesh(mesh, selectedCells)
    dmSub, y = getSubSolution(new_mesh, DoFMap, solution, selectedCells)
    return new_mesh, y, dmSub


def getRestrictionProlongationSubmesh(mesh, selectedCells, dm, dm_trunc):
    from PyNucleus_base.linear_operators import CSR_LinearOperator
    indptr = np.arange(dm_trunc.num_dofs+1, dtype=INDEX)
    indices = np.zeros((dm_trunc.num_dofs), dtype=INDEX)
    data = np.ones((dm_trunc.num_dofs), dtype=REAL)
    for cell_trunc in range(selectedCells.shape[0]):
        cell = selectedCells[cell_trunc]
        for i in range(dm.dofs_per_element):
            dof = dm.cell2dof_py(cell, i)
            dof_trunc = dm_trunc.cell2dof_py(cell_trunc, i)
            if dof >= 0 and dof_trunc >= 0:
                indices[dof_trunc] = dof
    R = CSR_LinearOperator(indices, indptr, data)
    R.num_columns = dm.num_dofs
    P = R.transpose()
    return R, P


def plotFunctions(mesh, dm, funs, labels=None, fig=None):
    from . functions import function
    if dm.num_dofs > 50000 or mesh.dim >= 3:
        return
    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt.gcf()
    if labels is None:
        labels = ['']*len(funs)
    else:
        assert len(funs) == len(labels)
    for f, l in zip(funs, labels):
        if isinstance(f, function):
            f = dm.interpolate(f)
        mesh.plotFunction(f, DoFMap=dm, label=l)
    fig.legend()


class plotManager:
    def __init__(self, mesh, dm, useSubPlots=False, defaults={}, interfaces=None):
        self.mesh = mesh
        self.dm = dm
        self.plots = []
        self.useSubPlots = useSubPlots
        if self.mesh.dim == 2:
            self.useSubPlots = True
        self.defaults = defaults
        self.interfaces = interfaces
        self.comm = interfaces.comm if self.interfaces is not None else None
        self.prepared = False

    def add(self, x, **kwargs):
        assert not self.prepared
        self.plots.append([x, kwargs])

    def preparePlots(self, tag=PHYSICAL):
        from . functions import function
        solutions = []
        for k in range(len(self.plots)):
            if isinstance(self.plots[k][0], function):
                self.plots[k][0] = self.dm.interpolate(self.plots[k][0])
            solutions.append(self.plots[k][0])
        (global_mesh,
         global_solutions,
         global_dm) = accumulate2global(self.mesh, self.interfaces, [self.dm]*len(solutions),
                                        solutions, comm=self.comm, tag=tag)
        if self.comm is None or self.comm.rank == 0:
            self.mesh = global_mesh
            if isinstance(global_solutions, list):
                for k in range(len(self.plots)):
                    self.plots[k][0] = global_solutions[k]
                self.dm = global_dm[0]
            else:
                self.plots[0][0] = global_solutions
                self.dm = global_dm
        self.prepared = True

    def plot(self, legendOutside=False):
        import matplotlib.pyplot as plt
        from . DoFMaps import fe_vector

        assert self.comm is None or self.comm.rank == 0

        if not self.prepared:
            self.preparePlots()

        needLegend = False
        if not self.useSubPlots:
            for x, k in self.plots:
                if 'label' in k:
                    needLegend = True
                if isinstance(x, fe_vector):
                    assert self.dm == x.dm
                    x.plot(**k)
                else:
                    self.mesh.plotFunction(x, DoFMap=self.dm, **k)
            if needLegend:
                if legendOutside:
                    plt.gca().legend(loc='lower left',
                                     bbox_to_anchor=(-0.1, 1.2),
                                     borderaxespad=0)
                else:
                    plt.gca().legend()
        else:
            numPlots = len(self.plots)
            plotsPerDirX = int(np.ceil(np.sqrt(numPlots)))
            plotsPerDirY = int(np.ceil(numPlots/plotsPerDirX))
            for k in range(len(self.plots)):
                ax = plt.gcf().add_subplot(plotsPerDirX, plotsPerDirY, k+1)
                plt.sca(ax)
                if k >= numPlots:
                    plt.gcf().delaxes(ax)
                else:
                    kwargs = self.defaults.copy()
                    kwargs.update(self.plots[k][1])
                    label = kwargs.pop('label', '')
                    vmin = kwargs.pop('vmin', None)
                    vmax = kwargs.pop('vmax', None)
                    x = self.plots[k][0]
                    if isinstance(x, fe_vector):
                        assert self.dm == x.dm
                        x.plot(**kwargs)
                    else:
                        self.mesh.plotFunction(x, DoFMap=self.dm, **kwargs)
                    ax.set_ylim([vmin, vmax])
                    ax.set_title(label)


def snapMeshes(mesh1, mesh2):
    from scipy.spatial import KDTree
    from PyNucleus_base import uninitialized

    tree = KDTree(mesh1.vertices)
    vertexCount = mesh1.num_vertices
    vertexTranslation = -np.ones((mesh2.num_vertices), dtype=INDEX)

    eps = 1e-9
    vertices2 = mesh2.vertices_as_array
    verticesToAdd = []
    for vertexNo in range(mesh2.num_vertices):
        neighbors = tree.query_ball_point(vertices2[vertexNo, :], eps)
        if len(neighbors) == 0:
            verticesToAdd.append(vertexNo)
            vertexTranslation[vertexNo] = vertexCount
            vertexCount += 1
        elif len(neighbors) == 1:
            vertexTranslation[vertexNo] = neighbors[0]
        else:
            raise NotImplementedError()
    vertices = np.vstack((mesh1.vertices_as_array,
                          mesh2.vertices_as_array[verticesToAdd, :]))
    translatedCells = uninitialized((mesh2.num_cells, mesh2.manifold_dim+1), dtype=INDEX)
    for cellNo in range(mesh2.num_cells):
        for vertexNo in range(mesh2.manifold_dim+1):
            translatedCells[cellNo, vertexNo] = vertexTranslation[mesh2.cells[cellNo, vertexNo]]
    cells = np.vstack((mesh1.cells_as_array,
                       translatedCells))
    mesh = mesh2d(vertices, cells)
    if mesh1.transformer is None:
        mesh.setMeshTransformation(mesh2.transformer)
    elif mesh2.transformer is None:
        mesh.setMeshTransformation(mesh1.transformer)
    else:
        raise NotImplementedError()
    return mesh
