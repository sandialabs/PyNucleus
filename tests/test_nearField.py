###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


# Compare dense matrix and near field assembly with cluster pairs covering all matrix blocks

from mpi4py import MPI
import numpy as np
from PyNucleus.base.myTypes import REAL, INDEX
from PyNucleus.base import uninitialized
from PyNucleus.base.tupleDict import arrayIndexSet
from PyNucleus.fem import simpleInterval, uniformSquare, P0_DoFMap, P1_DoFMap, constant
from PyNucleus.nl import H2Matrix, nonlocalBuilder, getFractionalKernel
from PyNucleus.nl.nonlocalLaplacian import nearFieldClusterPair
from PyNucleus.nl.clusterMethodCy import (getDoFBoxesAndCells,
                                          tree_node,
                                          getRefinementParams)

from PyNucleus.nl.fractionalOrders import (constFractionalOrder,
                                           variableConstFractionalOrder,
                                           leftRightFractionalOrder,
                                           layersFractionalOrder,
                                           lambdaFractionalOrder)
from PyNucleus.base import driver
from PyNucleus.nl.nonlocalProblems import nonlocalMeshFactory, HOMOGENEOUS_DIRICHLET
import pytest

epsRelDense = 3e-2
epsAbsDense = {(1, np.inf): 1e-5,
               (2, np.inf): 5e-3,
               (1, 1.0): 7e-3,
               (2, 1.0): 5e-3}
epsRelH2 = 1e-1
epsAbsH2 = {(1, np.inf): 5e-5,
            (2, np.inf): 5e-3,
            (1, 1.0): 7e-3,
            (2, 1.0): 5e-3}


class test:
    __test__ = False
    params = {'target_order': 3}

    @classmethod
    def setup_class(self):

        kernel = getFractionalKernel(self.dim, self.s, self.horizon, normalized=self.normalized, phi=self.phi)

        if self.dim == 1:
            self.mesh, nI = nonlocalMeshFactory.build('interval', kernel, self.boundaryCondition)
            domainIndicator = nI['domain']
            boundaryIndicator = nI['boundary']
            interactionIndicator = nI['interaction']
            self.tag = nI['tag']
            self.zeroExterior = nI['zeroExterior']
            # noRef = 6
        elif self.dim == 2:
            self.mesh, nI = nonlocalMeshFactory.build('square', kernel, self.boundaryCondition)
            # noRef = 2
        domainIndicator = nI['domain']
        boundaryIndicator = nI['boundary']
        interactionIndicator = nI['interaction']
        self.tag = nI['tag']
        self.zeroExterior = nI['zeroExterior']

        if self.element == 0:
            DoFMap = P0_DoFMap
        elif self.element == 1:
            DoFMap = P1_DoFMap
        else:
            raise NotImplementedError()

        self.dm = DoFMap(self.mesh, self.tag)
        while self.dm.num_dofs < 230:
            self.mesh = self.mesh.refine()
            self.dm = DoFMap(self.mesh, self.tag)
        self.mesh.sortVertices()
        self.dm = DoFMap(self.mesh, self.tag)
        print(self.dm)

        self.builder = nonlocalBuilder(self.mesh, self.dm, kernel, params=self.params, zeroExterior=self.zeroExterior)

        if isinstance(self.s, variableConstFractionalOrder) and self.phi is None:
            s = constFractionalOrder(self.s.value)
            kernel = getFractionalKernel(self.dim, s, self.horizon, normalized=True)
            self.constBuilder = nonlocalBuilder(self.mesh, self.dm, kernel, params=self.params, zeroExterior=self.zeroExterior)
            self.baseA = self.constBuilder.getDense()
            self.baseLabel = 'dense_const'
        else:
            self.baseA = self.builder.getDense()
            self.baseLabel = 'dense_var'

    def getPnear(self, maxLevels):
        boxes, cells = getDoFBoxesAndCells(self.mesh, self.dm)
        centers = uninitialized((self.dm.num_dofs, self.mesh.dim), dtype=REAL)
        for i in range(self.dm.num_dofs):
            centers[i, :] = boxes[i, :, :].mean(axis=1)
        blocks, jumps = self.builder.getKernelBlocksAndJumps()
        dofs = arrayIndexSet(np.arange(self.dm.num_dofs, dtype=INDEX))
        root = tree_node(None, dofs, boxes)
        if len(blocks) > 1:
            for key in blocks:
                subDofs = arrayIndexSet()
                subDofs.fromSet(blocks[key])
                if len(subDofs) > 0:
                    root.children.append(tree_node(root, subDofs, boxes, mixed_node=key == np.inf))
            root._dofs = None
            assert self.dm.num_dofs == sum([len(c.dofs) for c in root.children])
            assert len(root.children) > 1
        if maxLevels > 0:
            refParams = getRefinementParams(self.mesh, self.builder.kernel,
                                            {'maxLevels': maxLevels,
                                             'maxLevelsMixed': maxLevels})
            for n in root.leaves():
                n.refine(boxes, centers, refParams)
        root.set_id()
        # enter cells in leaf nodes
        for n in root.leaves():
            myCells = set()
            for dof in n.dofs.toSet():
                for jj in range(cells.indptr[dof], cells.indptr[dof+1]):
                    myCells.add(cells.indices[jj])
            n._cells = arrayIndexSet()
            n._cells.fromSet(myCells)

            diam = 0
            for i in range(self.dim):
                diam += (n.box[i, 1]-n.box[i, 0])**2
            diam = np.sqrt(diam)
            if 2*diam > self.horizon.value:
                print('Clusters of size {} to large for horizon {}'.format(diam, self.horizon.value))
                return [], {}
        Pnear = []
        r = list(root.leaves())
        for c in r:
            for d in r:
                assert c.isLeaf
                assert d.isLeaf
                Pnear.append(nearFieldClusterPair(c, d))
        for cP in Pnear:
            cP.set_cells_py()
        return Pnear, jumps

    def constH2(self):
        A_h2 = self.constBuilder.getH2()
        assert isinstance(A_h2, H2Matrix)
        self.compare("{}-h2_const".format(self.baseLabel), self.baseA, A_h2)

    def constCluster(self, maxLevels):
        if isinstance(self.s, variableConstFractionalOrder):
            s = constFractionalOrder(self.s.value)
            Pnear, _ = self.getPnear(maxLevels)
            if len(Pnear) > 0:
                A_fix_near = self.builder.assembleClusters(Pnear)
                self.compare("{}-cluster_const({})".format(self.baseLabel, maxLevels), self.baseA, A_fix_near)
        else:
            pytest.skip('Only works for variableConstFractionalOrder')

    def testConstCluster(self, levels=[0, 1, 2, 3, 4]):
        print()
        print(self.s)
        if isinstance(levels, int):
            levels = [levels]
        for maxLevels in levels:
            self.constCluster(maxLevels)

    def testConstH2(self):
        if isinstance(self.s, variableConstFractionalOrder) and self.dim == 1:
            print()
            print(self.s)
            self.constH2()
        elif self.dim == 2:
            pytest.skip('Does not work in 2d, since mesh to small to get H2 matrix')
        else:
            pytest.skip('Only works for variableConstFractionalOrder in 1D')

    def varDense(self):
        A_var = self.builder.getDense()
        self.compare("{}-dense_var".format(self.baseLabel), self.baseA, A_var)

    def varCluster(self, maxLevels):
        Pnear, jumps = self.getPnear(maxLevels)
        if len(Pnear) > 0:
            print('Jumps: {}'.format(len(jumps)))
            A_var_near = self.builder.assembleClusters(Pnear, jumps=jumps)
            self.compare("{}-cluster_var({})".format(self.baseLabel, maxLevels), self.baseA, A_var_near)

    def testVarDense(self):
        if isinstance(self.s, variableConstFractionalOrder):
            print()
            print(self.s)
            self.varDense()
        else:
            pytest.skip('Only makes sense for variableConstFractionalOrder')

    def testVarCluster(self, levels=[0, 1, 2, 3, 4, 5]):
        print()
        print(self.s)
        if isinstance(levels, int):
            levels = [levels]
        for maxLevels in levels:
            self.varCluster(maxLevels)

    def compare(self, label, A1, A2):
        if isinstance(A1, H2Matrix) or isinstance(A2, H2Matrix):
            epsAbs = epsAbsH2
            epsRel = epsRelH2
        else:
            epsAbs = epsAbsDense
            epsRel = epsRelDense
        A1 = A1.toarray()
        A2 = A2.toarray()
        value = np.absolute(A1-A2).max()
        valueRel = np.absolute((A1-A2)/A1)[np.absolute(A1) > 0].max()
        print('{}: abs: {} rel: {}'.format(label, value, valueRel))
        if value > epsAbs[(self.dim, self.horizon.value)] or valueRel > epsRel:
            print(A1.diagonal())
            print(A2.diagonal())
            print(A1.diagonal()-A2.diagonal())
            print()
            try:
                import matplotlib
                import matplotlib.pyplot as plt
            except ImportError:
                return
            if self.mesh.dim == 1:
                x = self.dm.getDoFCoordinates()
                indMax = np.absolute(A1.diagonal()-A2.diagonal()).argmax()
                print(indMax, x[indMax])
                X, Y = np.meshgrid(x, x)
                plt.figure()
                plt.pcolormesh(X, Y, A1)
                plt.colorbar()
                plt.title('A1')

                plt.figure()
                plt.pcolormesh(X, Y, A2)
                plt.colorbar()
                plt.title('A2')

                plt.figure()
                plt.pcolormesh(X, Y, np.absolute(np.around(A1-A2, 9)), norm=matplotlib.colors.LogNorm())
                plt.colorbar()
                plt.title('|A1-A2|')

                plt.figure()
                # plt.pcolormesh(X, Y, np.absolute(np.around((A1-A2)/A1, 9)), norm=matplotlib.colors.LogNorm())
                plt.pcolormesh(X, Y, np.log10(np.absolute((A1-A2)/A1)))
                plt.colorbar()
                plt.title('log |(A1-A2)/A1|')

                plt.show()
            else:
                plt.figure()
                err = self.dm.zeros()
                err.assign(np.absolute((A1.diagonal()-A2.diagonal())/(A1.diagonal())))
                err.plot(flat=True)
                plt.title('diagonal error')

                plt.figure()
                plt.pcolormesh(A1)
                plt.colorbar()
                plt.title('A1')

                plt.figure()
                plt.pcolormesh(A2)
                plt.colorbar()
                plt.title('A2')

                plt.figure()
                plt.pcolormesh(np.absolute(A1-A2), norm=matplotlib.colors.LogNorm())
                plt.colorbar()
                plt.title('|A1-A2|')

                plt.figure()
                # plt.pcolormesh(np.absolute(np.around((A1-A2)/A1, 9)), norm=matplotlib.colors.LogNorm())
                plt.pcolormesh(np.log10(np.absolute((A1-A2)/A1)))
                plt.colorbar()
                plt.title('log |(A1-A2)/A1|')

                plt.show()
        assert value < epsAbs[(self.dim, self.horizon.value)] and valueRel < epsRel


class test1D(test):
    dim = 1
    element = 1
    horizon = constant(np.inf)
    normalized = True
    phi = None
    boundaryCondition = HOMOGENEOUS_DIRICHLET


class test2D(test):
    dim = 2
    element = 1
    horizon = constant(np.inf)
    normalized = True
    phi = None
    boundaryCondition = HOMOGENEOUS_DIRICHLET


class const1D_025(test1D):
    __test__ = True
    s = variableConstFractionalOrder(0.25)


class const1D_075(test1D):
    __test__ = True
    s = variableConstFractionalOrder(0.75)


class const1D_025_finiteHorizon(test1D):
    __test__ = True
    s = variableConstFractionalOrder(0.25)
    horizon = constant(1.0)


class const1D_075_finiteHorizon(test1D):
    __test__ = True
    s = variableConstFractionalOrder(0.75)
    horizon = constant(1.0)


class leftRight1D(test1D):
    __test__ = True
    s = leftRightFractionalOrder(0.25, 0.75)


class leftRight1DfiniteHorizon(test1D):
    __test__ = True
    s = leftRightFractionalOrder(0.25, 0.75)
    horizon = constant(1.0)


class const2D_025(test2D):
    __test__ = True
    s = variableConstFractionalOrder(0.25)


class const2D_075(test2D):
    __test__ = True
    s = variableConstFractionalOrder(0.75)


class const2D_025_finiteHorizon(test2D):
    __test__ = True
    s = variableConstFractionalOrder(0.25)
    horizon = constant(1.0)


class const2D_075_finiteHorizon(test2D):
    __test__ = True
    s = variableConstFractionalOrder(0.75)
    horizon = constant(1.0)


class leftRight2DinfiniteHorizon(test2D):
    __test__ = True
    s = leftRightFractionalOrder(0.25, 0.75)


class leftRight2DfiniteHorizon(test2D):
    __test__ = True
    s = leftRightFractionalOrder(0.25, 0.75)
    horizon = constant(1.0)


class layers2D(test2D):
    __test__ = True
    t = np.linspace(0.2, 0.8, 4, dtype=REAL)
    s = np.empty((t.shape[0], t.shape[0]), dtype=REAL)
    for i in range(t.shape[0]):
        for j in range(t.shape[0]):
            s[i, j] = 0.5*(t[i]+t[j])
    s = layersFractionalOrder(2, np.linspace(-1., 1., s.shape[0]+1, dtype=REAL), s)



if __name__ == '__main__':
    d = driver(MPI.COMM_WORLD)
    d.add('dim', 1, acceptedValues=[2])
    d.add('doVar', False)
    d.add('doUnSym', False)
    d.add('target_order', 3)
    d.add('element', 1)
    d.add('levels', -1)
    params = d.process()
    if params['levels'] == -1:
        params['levels'] = [0, 1, 2]

    if params['dim'] == 1:
        tests = [const1D_025(), const1D_075()]
    elif params['dim'] == 2:
        tests = [const2D_025(), const2D_075()]
    for t in tests:
        t.setup_class()
        if params['doVar']:
            t.testVarDense()
        t.testConstCluster(params['levels'])
        if params['doVar']:
            t.testVarCluster(params['levels'])

# for s in [variableConstFractionalOrder(params['dim'], 0.25),
#           variableConstFractionalOrder(params['dim'], 0.75)]:
#     t = test(s)
#     t.setup()
#     if params['doVar']:
#         t.varDense()
#     for maxLevels in [0, 1, 2]:
#         t.constCluster(maxLevels)
#         if params['doVar']:
#             t.varCluster(maxLevels)

# if params['doVar']:
#     def sFun(x, y):
#         if ((abs(x[0]-0.25) < 0.125 or abs(x[0]+0.25) < 0.125) and
#             (abs(y[0]-0.25) < 0.125 or abs(y[0]+0.25) < 0.125)):
#             return 0.4
#         elif ((abs(x[0]-0.25) < 0.125 or abs(x[0]+0.25) < 0.125) and
#               not (abs(y[0]-0.25) < 0.125 or abs(y[0]+0.25) < 0.125)):
#             return 0.2
#         elif (not (abs(x[0]-0.25) < 0.125 or abs(x[0]+0.25) < 0.125) and
#               (abs(y[0]-0.25) < 0.125 or abs(y[0]+0.25) < 0.125)):
#             return 0.2
#         elif (not (abs(x[0]-0.25) < 0.125 or abs(x[0]+0.25) < 0.125) and
#               not (abs(y[0]-0.25) < 0.125 or abs(y[0]+0.25) < 0.125)):
#             return 0.75
#         else:
#             raise NotImplementedError()

#     sSpecial = lambdaFractionalOrder(params['dim'], 0.2, 0.75, True, sFun)

#     for s in [leftRightFractionalOrder(params['dim'], 0.25, 0.75),
#               leftRightFractionalOrder(params['dim'], 0.75, 0.25),
#               sSpecial]:
#         t = test(s)
#         t.setup()
#         for maxLevels in [0, 1, 2]:
#             t.varCluster(maxLevels)

#         if params['doUnSym']:
#             s.symmetric = False
#             A_unsym = assembleFractionalLaplacian(mesh, dm, s, params=params, zeroExterior=zeroExterior, interior=interior)

#             print()
#             print(sFun)
#             compare("var-unsym:  ", A_var.toarray(), A_unsym.toarray())

#             for maxLevels in [0, 1, 2]:
#                 Pnear = getPnear(mesh, dm, maxLevels)
#                 A_unsym_near = assembleNearField(Pnear, mesh, dm, sFun, params=params, zeroExterior=zeroExterior, interior=interior)

#                 print()
#                 print(sFun)
#                 compare("maxLevels:       {}\n".format(maxLevels) +
#                      "var-unsym_near: ", A_var.toarray(), A_unsym_near.toarray())

# if params['doUnSym']:
#     for sll, srr, slr, srl in [(0.25, 0.75, 0.25, 0.75)]:
#         sFun = leftRightFractionalOrder(sll, srr, slr, srl)
#         A_var = assembleNonlocalOperator(mesh, dm, sFun, params=params, zeroExterior=zeroExterior, interior=interior)
#         for maxLevels in [0, 1, 2]:
#             Pnear = getPnear(mesh, dm, maxLevels)
#             A_var_near = assembleNearField(Pnear, mesh, dm, sFun, params=params, zeroExterior=zeroExterior, interior=interior)

#             print()
#             print(sFun)
#             compare("maxLevels:         {}\n".format(maxLevels) +
#                  "unsym-unsym_near: ", A_var.toarray(), A_var_near.toarray())
