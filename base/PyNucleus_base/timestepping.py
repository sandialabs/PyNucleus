###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from . solvers import iterative_solver
from . myTypes import REAL
from . blas import uninitialized
import numpy as np


class Stepper:
    """
    Solve

    f(t, u, u_t) = A(t, u_t) + B(t, u) - g(t) = 0

    mass:                  t, u  -> A(t, u)
    explicit:              t, u  -> B(t, u)
    forcing:               t     -> g(t)
    implicitSolverBuilder: t, alpha, beta -> solver for alpha*A(t, u)+beta*B(t, u) = g

    """

    def __init__(self, dm, mass, solverBuilder, forcing, explicit=None, dt=None, solverIsTimeDependent=False, explicitIslinearAndTimeIndependent=False):
        self.dm = dm
        self.mass = mass
        self.solverBuilder = solverBuilder
        self.forcing = forcing
        self.explicit = explicit
        self.dt = dt
        self.solverIsTimeDependent = solverIsTimeDependent
        self.explicitIslinearAndTimeIndependent = explicitIslinearAndTimeIndependent

    def step(self, t, dt, u):
        raise NotImplementedError()

    def __call__(self, t, dt, u, forcingVector=None):
        return self.step(t, dt, u, forcingVector)

    def setRHS(self, t, dt, rhs):
        raise NotImplementedError()

    def residual(self, t, dt, ut, ut_plus_dt, residual, alpha=1., beta=1., forcingVector=None):
        raise NotImplementedError()

    def apply_jacobian(self, t, dt, ut, ut_plus_dt, residual, alpha=1., beta=1.):
        raise NotImplementedError()


class CrankNicolson(Stepper):
    def __init__(self, dm, mass, solverBuilder, forcing, explicit=None, theta=0.5, dt=None, solverIsTimeDependent=False, explicitIslinearAndTimeIndependent=False):
        assert theta > 0 and theta <= 1.
        self.theta = theta
        super(CrankNicolson, self).__init__(dm, mass, solverBuilder, forcing, explicit, dt, solverIsTimeDependent, explicitIslinearAndTimeIndependent)
        if self.theta < 1.:
            assert explicit is not None
        self.solver = None
        self.rhs = self.dm.zeros()
        self.rhs2 = self.dm.zeros()

    def getSolver(self, t, dt):
        if not self.solverIsTimeDependent:
            assert self.dt is not None
            assert self.dt == dt
            if self.solver is None:
                self.solver = self.solverBuilder(0., 1./self.dt, self.theta)
            return self.solver
        else:
            solver = self.solverBuilder(t+dt, 1./dt, self.theta)

    def setRHS(self, t, dt, rhs):
        self.forcing(t+dt, rhs)
        if self.theta < 1.:
            rhs *= self.theta
            self.forcing(t, self.rhs2)
            self.rhs2 *= (1.-self.theta)
            rhs += self.rhs2

    def step(self, t, dt, u, forcingVector=None):
        if dt is None:
            dt = self.dt
        assert dt is not None
        if not self.solverIsTimeDependent:
            assert dt == self.dt
        if forcingVector is not None:
            assert forcingVector.shape[0] == self.rhs.shape[0]
            self.rhs.assign(forcingVector)
        else:
            self.setRHS(t, dt, self.rhs)
        if self.theta < 1.:
            self.explicit(t, u, self.rhs2)
            self.rhs2 *= -(1.-self.theta)
            self.rhs += self.rhs2
        self.mass(t, u, self.rhs2)
        self.rhs2 *= 1./dt
        self.rhs += self.rhs2
        solver = self.getSolver(t, dt)
        if isinstance(solver, iterative_solver):
            solver.setInitialGuess(u)
        solver(self.rhs, u)
        return t+dt

    def residual(self, t, dt, ut, ut_plus_dt, residual, alpha=1., beta=1., forcingVector=None):
        if abs(alpha/dt) > 0:
            self.mass(t, ut, self.rhs)
            self.mass(t+dt, ut_plus_dt, self.rhs2)
            self.rhs *= -alpha/dt
            self.rhs2 *= alpha/dt
            residual.assign(self.rhs)
            residual += self.rhs2
        else:
            residual.assign(0.)
        if self.explicitIslinearAndTimeIndependent:
            self.rhs = beta*(1-self.theta)*ut + beta*self.theta*ut_plus_dt
            self.explicit(t, self.rhs, self.rhs2)
            residual += self.rhs2
        else:
            if self.theta < 1:
                self.explicit(t, ut, self.rhs)
                self.rhs *= beta*(1-self.theta)
                residual += self.rhs
            self.explicit(t+dt, ut_plus_dt, self.rhs2)
            self.rhs2 *= beta*self.theta
            residual += self.rhs2
        if forcingVector is not None:
            assert forcingVector.shape[0] == self.rhs.shape[0]
            self.rhs.assign(forcingVector)
        else:
            self.setRHS(t, dt, self.rhs)
        residual -= self.rhs

    def apply_jacobian(self, t, dt, ut, ut_plus_dt, residual, alpha=1., beta=1.):
        if abs(alpha/dt) > 0:
            self.mass(t, ut, self.rhs)
            self.mass(t+dt, ut_plus_dt, self.rhs2)
            self.rhs *= -alpha/dt
            self.rhs2 *= alpha/dt
            residual.assign(self.rhs)
            residual += self.rhs2
        else:
            residual.assign(0.)
        if self.explicitIslinearAndTimeIndependent:
            self.rhs = beta*(1-self.theta)*ut + beta*self.theta*ut_plus_dt
            self.explicit(t, self.rhs, self.rhs2)
            residual += self.rhs2
        else:
            if self.theta < 1:
                self.explicit(t, ut, self.rhs)
                self.rhs *= beta*(1-self.theta)
                residual += self.rhs
            self.explicit(t+dt, ut_plus_dt, self.rhs2)
            self.rhs2 *= beta*self.theta
            residual += self.rhs2


class ExplicitEuler(Stepper):
    def __init__(self, dm, mass, solverBuilder, forcing, explicit, solverIsTimeDependent=False):
        assert explicit is not None
        super(ExplicitEuler, self).__init__(dm, mass, solverBuilder, forcing, explicit, 0., solverIsTimeDependent)
        self.rhs = self.dm.zeros()
        self.rhs2 = self.dm.zeros()

    def step(self, t, dt, u):
        self.forcing(t+dt, self.rhs)
        self.rhs *= dt
        self.explicit(t, u, self.rhs2)
        self.rhs2 *= -dt
        self.rhs += self.rhs2
        if not self.solverIsTimeDependent:
            solver = self.solver
        else:
            solver = self.solverBuilder(t+dt, 0.)
        if isinstance(solver, iterative_solver):
            solver.setInitialGuess(u)
        solver(self.rhs, u)
        return t+dt


class ImplicitEuler(CrankNicolson):
    def __init__(self, dm, mass, solverBuilder, forcing, explicit=None, dt=None, solverIsTimeDependent=False):
        super(ImplicitEuler, self).__init__(dm, mass, solverBuilder, forcing, explicit,
                                            dt=dt,
                                            theta=1.,
                                            solverIsTimeDependent=solverIsTimeDependent)


class L1Scheme(Stepper):
    """
    L1 scheme for the Caputo fractional time derivative.
    """
    def __init__(self, alpha, maxTimeSteps, dm, mass, solverBuilder, forcing, explicit=None, dt=None, solverIsTimeDependent=False):
        from scipy.special import gamma
        super(L1Scheme, self).__init__(dm, mass, solverBuilder, forcing, explicit, dt, solverIsTimeDependent)
        assert not self.solverIsTimeDependent
        assert self.dt is not None
        assert 0 < alpha < 1.
        self.alpha = alpha
        self.maxTimeSteps = maxTimeSteps
        self.memory = self.dm.zeros(self.maxTimeSteps+1)
        self.b = (np.arange(1, self.maxTimeSteps+2)**(1-self.alpha) - np.arange(self.maxTimeSteps+1)**(1-self.alpha)) / gamma(2-self.alpha)
        self.solver = self.solverBuilder(0., self.dt**self.alpha / self.b[0])
        self.rhs = self.dm.zeros()
        self.rhs2 = self.dm.zeros()
        self.k = 1

    def step(self, t, dt, u):
        if dt is None:
            dt = self.dt
        assert dt is not None
        assert self.k <= self.maxTimeSteps
        if not self.solverIsTimeDependent:
            assert dt == self.dt
        self.forcing(t+dt, self.rhs)
        self.rhs *= dt**self.alpha/self.b[0]
        self.mass(t, u, self.rhs2)
        if self.k == 1:
            self.mass(0, u, self.memory[0])
        coeff = uninitialized((self.k), dtype=REAL)
        coeff[0] = self.b[self.k-1]/self.b[0]
        for j in range(1, self.k):
            coeff[self.k-j] = (self.b[j-1]-self.b[j])/self.b[0]
        self.rhs2.assign(np.dot(coeff, self.memory.toarray()[:self.k, :]))
        self.rhs += self.rhs2

        solver = self.solver
        if isinstance(solver, iterative_solver):
            solver.setInitialGuess(u)
        solver(self.rhs, u)
        self.mass(t+dt, u, self.memory[self.k])
        self.k += 1
        return t+dt


class fastL1Scheme(Stepper):
    """
    Fast L1 scheme for the Caputo fractional time derivative.
    """
    def __init__(self, alpha, maxTimeSteps, dm, mass, solverBuilder, forcing, explicit=None, dt=None, solverIsTimeDependent=False, eps=1e-4):
        from scipy.special import gamma
        super(fastL1Scheme, self).__init__(dm, mass, solverBuilder, forcing, explicit, dt, solverIsTimeDependent)
        assert not self.solverIsTimeDependent
        assert self.dt is not None
        assert 0 < alpha < 1.
        self.alpha = alpha
        self.maxTimeSteps = maxTimeSteps
        self.eps = eps
        self.s, self.w = self.getWeights()
        self.Nexp = self.w.shape[0]-1
        self.memory = self.dm.zeros(self.Nexp+1)
        self.solver = self.solverBuilder(0., self.dt**self.alpha * gamma(2-self.alpha))
        self.rhs = self.dm.zeros()
        self.rhs2 = self.dm.zeros()
        self.uold = self.dm.zeros()
        self.k = 1

    def getWeights(self):
        from scipy.special import roots_sh_jacobi, roots_sh_legendre
        from scipy.special import gamma
        M = int(np.ceil(np.log2(self.maxTimeSteps*self.dt)))
        N = int(np.ceil(np.log2(1/self.dt) + np.log2(np.log(1/self.eps))))
        no = int(np.ceil(np.log(1/self.eps))/2)
        ns = int(np.ceil(np.log(1/self.eps))/2)
        nl = int(np.ceil(np.log(1/self.dt) + np.log(1/self.eps))/2)
        s, w = [np.array([0.])], [np.array([1.])]
        so, wo = roots_sh_jacobi(no, self.alpha+1, self.alpha+1)
        so *= 2**M
        wo *= (2**M)**(self.alpha+1)
        s.append(so)
        w.append(wo)
        ss0, ws0 = roots_sh_legendre(ns)
        for j in range(M, 0):
            ss = (2**(j+1)-2**j) * ss0 + 2**j
            ws = ws0 * (2**(j+1)-2**j) * ss**self.alpha
            s.append(ss)
            w.append(ws)
        sl0, wl0 = roots_sh_legendre(nl)
        for j in range(max(M, 0), N+1):
            sl = (2**(j+1)-2**j)*sl0 + 2**j
            wl = wl0 * (2**(j+1)-2**j) * sl**self.alpha
            s.append(sl)
            w.append(wl)
        s = np.concatenate(s)
        w = self.alpha * (1-self.alpha) * self.dt**self.alpha * np.concatenate(w)/gamma(1+self.alpha)
        return s, w

    def step(self, t, dt, u):
        from scipy.special import gamma
        if dt is None:
            dt = self.dt
        assert dt is not None
        assert self.k <= self.maxTimeSteps
        if not self.solverIsTimeDependent:
            assert dt == self.dt

        self.mass(t, u, self.rhs2)
        if self.k == 1:
            self.memory[0].assign(self.rhs2)
        else:
            expDtS = np.exp(-dt*self.s)
            self.memory.scale(expDtS)
            temp = expDtS/(self.s**2*dt)
            temp[0] = 0.
            self.memory.scaledUpdate(self.rhs2, temp * (expDtS - 1 + self.s*dt))
            self.memory.scaledUpdate(self.uold, temp * (1 - expDtS - expDtS*self.s*dt))
            del expDtS, temp
        self.uold.assign(self.rhs2)

        self.forcing(t+dt, self.rhs)
        self.rhs *= dt**self.alpha * gamma(2-self.alpha)

        self.rhs2 *= self.alpha
        self.rhs += self.rhs2

        self.w[0] = (1-self.alpha) * (dt/(t+dt))**self.alpha
        self.rhs2.assign(np.dot(self.w, self.memory.toarray()))
        self.rhs += self.rhs2

        solver = self.solver
        if isinstance(solver, iterative_solver):
            solver.setInitialGuess(u)
        solver(self.rhs, u)

        self.k += 1
        return t+dt


class IMEX:
    def __init__(self,
                 dm,
                 implicit, implicitSolve, explicit,
                 c, bExpl, bImpl, AExpl, AImpl,
                 numSystemVectors,
                 mass=None, massSolve=None, forcing=None):
        """
        IMEX time stepping for
        u_t + I(t, u(t)) + E(t, u(t)) + f(t)

        implicit: implicit part (t, u1, u2) that sets u2 with implicit function evaluation at (t, u1)
        implicitSolve: solver taking (t, rhs, u, u_new, dt) to solve (Mass+dt*Stiffness)u_new = rhs
        explicit: explicit part (t, u1, u2) to set u2 with explicit function evaluation at (t, u1)

        Defining the IMEX scheme
        c: time sub-steps for both implicit and explicit method
        bExpl: weights for explicit method
        bImpl: weights for implicit method
        AExpl: weights for explicit method
        AImpl: weights for implicit method
        size: size of the vectors of the solution u

        Optional arguments:
        mass: mass matrix (for the time derivative)
        massSolve: method to solve mass problem
        forcing: forcing function (t, f(t))
        """
        self.implicit = implicit
        self.implicitSolve = implicitSolve
        self.explicit = explicit
        self.forcing = forcing
        self.c = c
        self.bExpl = bExpl
        self.bImpl = bImpl
        self.AExpl = AExpl
        self.AImpl = AImpl
        self.s = AExpl.shape[0]
        self.mass = mass
        self.massSolve = massSolve
        self.U = [dm.zeros(numSystemVectors) for _ in range(self.s)]
        self.UExpl = [dm.zeros(numSystemVectors) for _ in range(self.s)]
        self.UImpl = [dm.zeros(numSystemVectors) for _ in range(self.s)]
        self.rhs = dm.zeros(numSystemVectors)
        if self.forcing:
            self.force = [dm.zeros(numSystemVectors) for _ in range(self.s)]

    def step(self, u, t, dt, unew):
        if self.forcing:
            for k in range(self.s):
                if self.AImpl[:, k].sum() != 0. or self.bImpl[k] != 0.:
                    self.forcing(t+self.c[k]*dt, self.force[k])
        for k in range(self.s):
            if np.absolute(self.AExpl[k, :]).max() == 0.:
                for j in range(u.numVectors):
                    self.U[k][j].assign(u[j])
            else:
                if self.mass:
                    self.mass(u, self.rhs)
                else:
                    for j in range(self.rhs.numVectors):
                        self.rhs[j].assign(u[j])
                for j in range(k):
                    if self.AExpl[k, j] != 0:
                        self.rhs += dt*self.AExpl[k, j]*self.UExpl[j]
                    if self.AImpl[k, j] != 0:
                        self.rhs += dt*self.AImpl[k, j]*self.UImpl[j]
                if self.forcing:
                    for j in range(k+1):
                        if self.AImpl[k, j] != 0:
                            self.rhs += dt*self.AImpl[k, j]*self.force[j]

                self.implicitSolve(t + self.c[k]*dt, self.rhs, u, self.U[k], self.AImpl[k, k]*dt)

            if self.AExpl[:, k].sum() != 0. or self.bExpl[k] != 0.:
                self.explicit(t + self.c[k]*dt, self.U[k], self.UExpl[k])
            if self.AImpl[:, k].sum() != 0. or self.bImpl[k] != 0.:
                self.implicit(t + self.c[k]*dt, self.U[k], self.UImpl[k])
        if self.mass:
            self.mass(u, self.rhs)
            for k in range(self.s):
                if self.bExpl[k] != 0.:
                    self.rhs += dt*self.bExpl[k]*self.UExpl[k]
                if self.bImpl[k] != 0.:
                    self.rhs += dt*self.bImpl[k]*self.UImpl[k]
            if self.forcing:
                for k in range(self.s):
                    if self.bImpl[k] != 0.:
                        self.rhs += dt*self.bImpl[k]*self.force[k]
            self.massSolve(self.rhs, u, unew)
        else:
            for i in range(u.numVectors):
                unew[i][:] = u[i][:]
            for k in range(self.s):
                unew += dt*self.bExpl[k]*self.UExpl[k]
                unew += dt*self.bImpl[k]*self.UImpl[k]
            if self.forcing:
                for k in range(self.s):
                    unew += dt*self.bImpl[k]*self.force[k]


class EulerIMEX(IMEX):
    gamma = 1.

    def __init__(self, dm, implicit, implicitSolve, explicit, numSystemVectors,
                 mass=None, massSolve=None, forcing=None):
        AExpl = np.array([[0, 0],
                          [1, 0]])
        AImpl = np.array([[0, 0],
                          [0, 1]])
        bExpl = np.array([1, 0])
        bImpl = np.array([0, 1])
        c = np.array([0, 1])
        IMEX.__init__(self, dm,
                      implicit, implicitSolve, explicit,
                      c, bExpl, bImpl, AExpl, AImpl,
                      numSystemVectors,
                      mass, massSolve, forcing)


class ARS3(IMEX):
    gamma = (3+np.sqrt(3))/6

    def __init__(self, dm, implicit, implicitSolve, explicit, numSystemVectors,
                 mass=None, massSolve=None, forcing=None):
        gamma = self.gamma
        AExpl = np.array([[0, 0, 0],
                          [gamma, 0, 0],
                          [gamma-1, 2*(1-gamma), 0]])
        AImpl = np.array([[0, 0, 0],
                          [0, gamma, 0],
                          [0, 1-2*gamma, gamma]])
        bExpl = np.array([0, 1/2, 1/2])
        bImpl = np.array([0, 1/2, 1/2])
        c = np.array([0,gamma, 1-gamma])
        IMEX.__init__(self, dm,
                      implicit, implicitSolve, explicit,
                      c, bExpl, bImpl, AExpl, AImpl,
                      numSystemVectors,
                      mass, massSolve, forcing)


class koto(IMEX):
    gamma = 1.

    def __init__(self, dm, implicit, implicitSolve, explicit, numSystemVectors,
                 mass=None, massSolve=None, forcing=None):
        AImpl = np.array([[0, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, -0.5, 1, 0],
                          [0, -1, 1, 1]])
        AExpl = np.array([[0, 0, 0, 0],
                          [1, 0, 0, 0],
                          [0.5, 0, 0, 0],
                          [0, 0, 1, 0]])
        bExpl = np.array([0, 0, 1, 0])
        bImpl = np.array([0, -1, 1, 1])
        c = np.array([0, 1, 0.5, 1])
        IMEX.__init__(self, dm,
                      implicit, implicitSolve, explicit,
                      c, bExpl, bImpl, AExpl, AImpl,
                      numSystemVectors,
                      mass, massSolve, forcing)
