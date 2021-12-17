###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from . solvers import iterative_solver
import numpy as np


class Stepper:
    """
    Solve

    A(t, u_t) + B(t, u) = f(t)

    mass:                  t, u  -> A(t, u)
    explicit:              t, u  -> B(t, u)
    forcing:               t     -> f(t)
    implicitSolverBuilder: t, dt -> solver for A(t, u)+dt*B(t, u)

    """

    def __init__(self, dm, mass, solverBuilder, forcing, explicit=None, dt=None, solverIsTimeDependent=False):
        self.dm = dm
        self.mass = mass
        self.solverBuilder = solverBuilder
        self.forcing = forcing
        self.explicit = explicit
        self.dt = dt
        self.solverIsTimeDependent = solverIsTimeDependent

    def step(self, t, dt, u):
        raise NotImplementedError()

    def __call__(self, t, dt, u):
        return self.step(t, dt, u)


class CrankNicolson(Stepper):
    def __init__(self, dm, mass, solverBuilder, forcing, explicit=None, theta=0.5, dt=None, solverIsTimeDependent=False):
        assert theta > 0 and theta <= 1.
        self.theta = theta
        super(CrankNicolson, self).__init__(dm, mass, solverBuilder, forcing, explicit, dt, solverIsTimeDependent)
        if self.theta < 1.:
            assert explicit is not None
        if not self.solverIsTimeDependent:
            assert self.dt is not None
            self.solver = self.solverBuilder(0., self.theta*self.dt)
        self.rhs = self.dm.zeros()
        self.rhs2 = self.dm.zeros()

    def step(self, t, dt, u):
        if dt is None:
            dt = self.dt
        assert dt is not None
        if not self.solverIsTimeDependent:
            assert dt == self.dt
        self.forcing(t+dt, self.rhs)
        if self.theta < 1.:
            self.rhs *= self.theta
            self.forcing(t, self.rhs2)
            self.rhs2 *= (1.-self.theta)
            self.rhs += self.rhs2
        self.rhs *= dt
        if self.theta < 1.:
            self.explicit(t, u, self.rhs2)
            self.rhs2 *= -(1.-self.theta)*dt
            self.rhs += self.rhs2
        self.mass(t, u, self.rhs2)
        self.rhs += self.rhs2
        if not self.solverIsTimeDependent:
            solver = self.solver
        else:
            solver = self.solverBuilder(t+dt, self.theta*dt)
        if isinstance(solver, iterative_solver):
            solver.setInitialGuess(u)
        solver(self.rhs, u)
        return t+dt


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
                                            theta=1.,
                                            solverIsTimeDependent=solverIsTimeDependent)


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
