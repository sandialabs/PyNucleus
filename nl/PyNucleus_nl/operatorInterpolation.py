###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
from PyNucleus_base.myTypes import INDEX, REAL


class admissibleSet:
    def __init__(self, ranges):
        if not isinstance(ranges, np.ndarray):
            ranges = np.array(ranges)
        if ranges.ndim == 1:
            ranges = ranges[np.newaxis, :]
        assert ranges.shape[1] == 2
        self.ranges = ranges

    def getNumParams(self):
        return self.ranges.shape[0]

    def getNumActiveParams(self):
        k = 0
        for i in range(self.numParams):
            if self.ranges[i, 0] < self.ranges[i, 1]:
                k += 1
        return k

    def getActiveParams(self):
        return np.array([i for i in range(self.numParams)
                         if self.ranges[i, 0] < self.ranges[i, 1]],
                        dtype=INDEX)

    def getInactiveParams(self):
        return np.array([i for i in range(self.numParams)
                         if self.ranges[i, 0] >= self.ranges[i, 1]],
                        dtype=INDEX)

    def getLowerBounds(self):
        return self.ranges[:, 0].copy()

    def getUpperBounds(self):
        return self.ranges[:, 1].copy()

    def getBounds(self):
        from scipy.optimize import Bounds
        return Bounds(self.ranges[:, 0], self.ranges[:, 1])

    def getActiveLowerBounds(self):
        idx = self.getActiveParams()
        return self.ranges[idx, 0]

    def getActiveUpperBounds(self):
        idx = self.getActiveParams()
        return self.ranges[idx, 1]

    def getActiveBounds(self):
        from scipy.optimize import Bounds
        return Bounds(self.getActiveLowerBounds(),
                      self.getActiveUpperBounds())

    numParams = property(fget=getNumParams)
    numActiveParams = property(fget=getNumActiveParams)
    bounds = property(fget=getBounds)

    def isAdmissible(self, z):
        if not isinstance(z, (list, tuple, np.ndarray)):
            z = np.array([z])
        assert len(z) == self.getNumParams()
        for i in range(self.numParams):
            if not (self.ranges[i, 0] <= z[i] <= self.ranges[i, 1]):
                return False
        return True

    def __add__(self, other):
        return admissibleSetPair(self, other)

    def augment(self, zActive):
        if zActive is None:
            zActive = np.array([], dtype=REAL)
        if not isinstance(zActive, (list, tuple, np.ndarray)):
            zActive = np.array([zActive])
        elif isinstance(zActive, (list, tuple)):
            zActive = np.array(zActive)
        assert zActive.shape[0] == self.numActiveParams, (zActive, zActive.shape[0], self.numActiveParams)
        z = self.getLowerBounds()
        z[self.getActiveParams()] = zActive
        return z

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.ranges)


class admissibleSetPair(admissibleSet):
    def __init__(self, set1, set2):
        self.set1 = set1
        self.set2 = set2
        super(admissibleSetPair, self).__init__(np.vstack((set1.ranges, set2.ranges)))

    def project1(self, z):
        assert z.shape[0] == self.numParams, (z.shape[0], self.numParams)
        return z[:self.set1.numParams]

    def project2(self, z):
        assert z.shape[0] == self.numParams, (z.shape[0], self.numParams)
        return z[-self.set2.numParams:]

    def inject1(self, z1):
        assert z1.shape[0] == self.set1.numParams, (z1.shape[0], self.set1.numParams)
        z = np.zeros((self.numParams), dtype=REAL)
        z[:self.set1.numParams] = z1
        return z

    def inject2(self, z2):
        assert z2.shape[0] == self.set2.numParams, (z2.shape[0], self.set2.numParams)
        z = np.zeros((self.numParams), dtype=REAL)
        z[self.set1.numParams:] = z2
        return z


def getChebyIntervalsAndNodes(s_left, s_right, delta, r, eta, M_max=20, M_min=3, variableOrder=False, doSplitM=False, fixedXi=-1):
    # find best Chebyshev integration order

    assert delta > 0.
    assert s_left > 0.
    assert s_right < 1.

    def regularityLifting(s):
        # solution regularity given rhs regularity r
        return min(r+s, 1/2)

    def getSigmaMaxFixedOrder(smin, eta, M=1):
        """
        For given smin and prescribed contraction factor using (M+1)-th order
        Chebyshev interpolation, get the largest interval [smin, smax].
        """
        s1 = smin
        s2 = min(1, smin + regularityLifting(smin))
        epsHat = lambda t: s1+s2-2*t
        if delta > 1:
            C_delta = 4*(np.exp(-1.) + delta**(epsHat(smin)+1))
        else:
            C_delta = 4*np.exp(-1.)
        sigma = (eta/C_delta)**(1/(M+1))
        smax = smin + 2*sigma/(1+4*sigma) * min(1-smin, regularityLifting(smin))
        assert (s1+s2)/2 - 1/2 < smax
        assert smax < (s1+s2)/2
        return smax

    def getSigmaMaxVariableOrder(smin, xi):
        s1 = smin
        s2 = min(1, smin + regularityLifting(smin))
        smax = (s1+s2)/2 - xi * min(1-smin, regularityLifting(smin))
        assert (s1+s2)/2 - 1/2 < smax
        assert smax < (s1+s2)/2

        epsHat = lambda t: s1+s2-2*t
        if delta > 1:
            C_delta = 4*(np.exp(-1.) + delta**(epsHat(smin)+1))
        else:
            C_delta = 4*np.exp(-1.)
        sigma = (smax-smin)/2/epsHat(smax)
        assert sigma < 1.
        assert sigma >= 0.
        M = int(np.ceil(np.log(eta/C_delta)/np.log(sigma) - 1))
        return smax, M

    def getIntervalsFixedOrder(s_left, s_right, eta, M, M2=None):
        """
        Get intervals for [s_left, s_right] using (M+1)-th order Chebyshev
        interpolation with error bounded by err.
        """
        if M2 is None:
            M2 = M
        s = s_left
        if s >= 1/2:
            M = M2
        intervals = []
        Mvals = []
        k = 0
        while (s < s_right) and (k < 1000):
            s_new = getSigmaMaxFixedOrder(s, eta, M=M)
            s_new = min(s_new, s_right)
            intervals.append((s, s_new))
            Mvals.append(M)
            s = s_new
            if s >= 1/2:
                M = M2
            k += 1
        Mvals = np.array(Mvals)
        return intervals, Mvals

    def getIntervalsVariableOrder(s_left, s_right, eta, xi):
        """
        Get intervals for [s_left, s_right] using (M+1)-th order Chebyshev
        interpolation with error bounded by err.
        """
        s = s_left
        intervals = []
        Mvals = []
        k = 0
        while (s < s_right) and (k < 1000):
            s_new, M = getSigmaMaxVariableOrder(s, xi)
            M = max(M, M_min)
            M = min(M, M_max)
            s_new = min(s_new, s_right)
            intervals.append((s, s_new))
            Mvals.append(M)
            s = s_new
            k += 1
        Mvals = np.array(Mvals)
        return intervals, Mvals

    def getChebyNodes(n, a, b):
        """
        Get n-th order Chebyshev nodes for interval [a, b].
        """
        eta = np.cos((2.0*np.arange(n, 0, -1)-1.0)/(2*n)*np.pi)
        return 0.5*(a+b) + 0.5*(b-a)*eta

    def costFixedOrder(M, M2=None):
        _, Mvals = getIntervalsFixedOrder(s_left, s_right, eta, M, M2)
        return (Mvals+1).sum()

    def costVariableOrder(xi):
        _, Mvals = getIntervalsVariableOrder(s_left, s_right, eta, xi)
        return (Mvals+1).sum()

    if variableOrder:
        if fixedXi <= 0:
            xi_vals = np.linspace(0.1, 0.5, 300)[1:-1]
        else:
            assert 0.1 < fixedXi
            assert fixedXi < 0.5
            xi_vals = np.array([fixedXi])
        cost_vals = np.array([costVariableOrder(xi) for xi in xi_vals])
        xi_opt = xi_vals[cost_vals.argmin()]
        intervals, Mvals = getIntervalsVariableOrder(s_left, s_right, eta, xi_opt)
    else:
        Mvals = np.arange(M_min, M_max+1)
        if doSplitM:
            cost_vals = np.empty((Mvals.shape[0], Mvals.shape[0]))
            for i in range(Mvals.shape[0]):
                M = Mvals[i]
                for j in range(Mvals.shape[0]):
                    M2 = Mvals[j]
                    cost_vals[i, j] = costFixedOrder(M, M2)
            idx, idx2 = np.unravel_index(cost_vals.argmin(), cost_vals.shape)
            Mopt = Mvals[idx]
            M2opt = Mvals[idx2]
            intervals, Mvals = getIntervalsFixedOrder(s_left, s_right, eta, Mopt, M2opt)
        else:
            cost_vals = np.array([costFixedOrder(M) for M in Mvals])
            Mopt = Mvals[cost_vals.argmin()]
            intervals, Mvals = getIntervalsFixedOrder(s_left, s_right, eta, Mopt)

    nodes = []
    for k in range(len(intervals)):
        a, b = intervals[k]
        M = Mvals[k]
        intervalNodes = getChebyNodes(M+1, a, b)
        nodes.append(intervalNodes)
    return intervals, nodes
