###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from . solvers import iterative_solver
from . myTypes import REAL
from . blas import uninitialized
from . factory import factory
import numpy as np


class Stepper:
    """
    Solve

    f(t, u, u_t) = A(t, u_t) + B(t, u) - g(t) = 0.

    We assume that A is linear in its second argument. Let

    residual     : t, alpha, beta, gamma, ut, u -> alpha*A(t, ut)+beta*B(t, u)-gamma*g(t)
    solverBuilder: t, alpha, beta               -> solver for alpha*A(t, u)+beta*B(t, u) = g

    """

    def __init__(self, dm, residual, solverBuilder, dt=None, solverIsTimeDependent=False, explicitIslinearAndTimeIndependent=False):
        self.dm = dm
        self.residualNew = residual
        self.solverBuilder = solverBuilder
        self.dt = dt
        self.solverIsTimeDependent = solverIsTimeDependent
        self.explicitIslinearAndTimeIndependent = explicitIslinearAndTimeIndependent
        self._solver = None

    def getSolver(self, t, coeff_A, coeff_B):
        if not self.solverIsTimeDependent:
            if self._solver is None:
                self._solver = self.solverBuilder(0., coeff_A, coeff_B)
            return self._solver
        else:
            return self.solverBuilder(t, coeff_A, coeff_B)

    def step(self, t, dt, u):
        raise NotImplementedError()

    def __call__(self, t, dt, u, forcingVector=None):
        return self.step(t, dt, u, forcingVector)

    def setRHS(self, t, dt, rhs):
        raise NotImplementedError()

    def residualNew(self, t, u, ut, residual, coeff_A=1., coeff_B=1., coeff_g=1., coeff_residual=0., forcingVector=None):
        raise NotImplementedError()

    def residual(self, t, dt, ut, ut_plus_dt, residual, alpha=1., beta=1., forcingVector=None):
        raise NotImplementedError()

    def apply_jacobian(self, t, dt, ut, ut_plus_dt, residual, alpha=1., beta=1.):
        raise NotImplementedError()


class CrankNicolson(Stepper):
    """
    1/dt*A(t+dt, u_{k+1}) + theta*B(t+dt, u_{k+1}) = (1-theta)*g(t_{k}) + theta*g(t_{k+1}) + 1/dt*A(t_{k}, u_{k}) - (1-theta)*B(t_{k}, u_{k})
    """
    def __init__(self, dm, residual, solverBuilder, theta=0.5, dt=None,
                 solverIsTimeDependent=False, explicitIslinearAndTimeIndependent=False):
        assert theta >= 0 and theta <= 1.
        self.theta = theta
        super(CrankNicolson, self).__init__(dm, residual, solverBuilder, dt, solverIsTimeDependent, explicitIslinearAndTimeIndependent)
        self.rhs = self.dm.zeros()
        self.rhs2 = self.dm.zeros()

    def setRHS(self, t, dt, rhs):
        self.residualNew(t=t,
                         u=None,
                         ut=None,
                         residual=rhs,
                         coeff_A=0.,
                         coeff_B=0.,
                         coeff_g=-(1-self.theta))
        self.residualNew(t=t+dt,
                         u=None,
                         ut=None,
                         residual=rhs,
                         coeff_A=0.,
                         coeff_B=0.,
                         coeff_g=-self.theta,
                         coeff_residual=1.)

    def step(self, t, dt, u, forcingVector=None):
        if dt is None:
            dt = self.dt
        assert dt is not None
        if not self.solverIsTimeDependent:
            assert dt == self.dt

        # 1/dt * A(t, u) - (1-theta) * B(t, u) + (1-theta) * g(t)
        self.residualNew(t, u, u, self.rhs, coeff_A=1./dt, coeff_B=-(1-self.theta), coeff_g=-(1-self.theta), forcingVector=forcingVector)
        # theta * g(t+dt)
        self.residualNew(t+dt, u, u, self.rhs, coeff_A=0., coeff_B=0., coeff_g=-self.theta, coeff_residual=1., forcingVector=forcingVector)

        # solver for 1/dt * A(t+dt, u) + theta*B(t+dt, u)
        solver = self.getSolver(t=t+dt, coeff_A=1/dt, coeff_B=self.theta)
        if isinstance(solver, iterative_solver):
            solver.setInitialGuess(u)
        solver(self.rhs, u)
        return t+dt

    def residual(self, t, dt, ut, ut_plus_dt, residual, alpha=1., beta=1., forcingVector=None):
        # alpha/dt*[A(t+dt, ut_plus_dt) - A(t, ut)]
        # + beta*[(1-theta)*B(t, ut) + theta*B(t+dt, ut_plus_dt)]
        # - (1-theta)*g(t) - theta*g(t+dt)

        self.residualNew(t, ut, ut, self.rhs, alpha=alpha/dt, beta=-beta*(1-self.theta), gamma=-(1-self.theta))
        self.residualNew(t+dt, ut_plus_dt, ut_plus_dt, alpha=alpha/dt, beta=beta*self.theta, gamma=self.theta, delta=-1.)

        # if abs(alpha/dt) > 0:
        #     self.mass(t, ut, self.rhs)
        #     self.mass(t+dt, ut_plus_dt, self.rhs2)
        #     self.rhs *= -alpha/dt
        #     self.rhs2 *= alpha/dt
        #     residual.assign(self.rhs)
        #     residual += self.rhs2
        # else:
        #     residual.assign(0.)
        # if self.explicitIslinearAndTimeIndependent:
        #     self.rhs = beta*(1-self.theta)*ut + beta*self.theta*ut_plus_dt
        #     self.explicit(t, self.rhs, self.rhs2)
        #     residual += self.rhs2
        # else:
        #     if self.theta < 1:
        #         self.explicit(t, ut, self.rhs)
        #         self.rhs *= beta*(1-self.theta)
        #         residual += self.rhs
        #     self.explicit(t+dt, ut_plus_dt, self.rhs2)
        #     self.rhs2 *= beta*self.theta
        #     residual += self.rhs2
        # if forcingVector is not None:
        #     assert forcingVector.shape[0] == self.rhs.shape[0]
        #     self.rhs.assign(forcingVector)
        # else:
        #     self.setRHS(t, dt, self.rhs)
        # residual -= self.rhs

    def apply_jacobian(self, t, dt, ut, ut_plus_dt, residual, alpha=1., beta=1.):
        # alpha/dt*[A(t+dt, ut_plus_dt) - A(t, ut)]
        # + beta*[(1-theta)*B(t, ut) + theta*B(t+dt, ut_plus_dt)]

        self.residualNew(t, ut, ut, residual, alpha=alpha/dt, beta=-beta*(1-self.theta), gamma=0.)
        self.residualNew(t, ut_plus_dt, ut_plus_dt, residual, alpha=alpha/dt, beta=beta*self.theta, gamma=0., delta=-1.)

        # if abs(alpha/dt) > 0:
        #     self.mass(t, ut, self.rhs)
        #     self.mass(t+dt, ut_plus_dt, self.rhs2)
        #     self.rhs *= -alpha/dt
        #     self.rhs2 *= alpha/dt
        #     residual.assign(self.rhs)
        #     residual += self.rhs2
        # else:
        #     residual.assign(0.)
        # if self.explicitIslinearAndTimeIndependent:
        #     self.rhs = beta*(1-self.theta)*ut + beta*self.theta*ut_plus_dt
        #     self.explicit(t, self.rhs, self.rhs2)
        #     residual += self.rhs2
        # else:
        #     if self.theta < 1:
        #         self.explicit(t, ut, self.rhs)
        #         self.rhs *= beta*(1-self.theta)
        #         residual += self.rhs
        #     self.explicit(t+dt, ut_plus_dt, self.rhs2)
        #     self.rhs2 *= beta*self.theta
        #     residual += self.rhs2


class ExplicitEuler(CrankNicolson):
    """
    1/dt*A(t+dt, u_{k+1}) = g(t_{k}) + 1/dt*A(t_{k}, u_{k}) - B(t_{k}, u_{k})
    """
    def __init__(self, dm, residual, solverBuilder, dt=None, solverIsTimeDependent=False, explicitIslinearAndTimeIndependent=False):
        super(ExplicitEuler, self).__init__(dm, residual, solverBuilder,
                                            dt=dt,
                                            theta=0.,
                                            solverIsTimeDependent=solverIsTimeDependent,
                                            explicitIslinearAndTimeIndependent=explicitIslinearAndTimeIndependent)


class ImplicitEuler(CrankNicolson):
    """
    1/dt*A(t+dt, u_{k+1}) + B(t+dt, u_{k+1}) = g(t_{k+1}) + 1/dt*A(t_{k}, u_{k})
    """
    def __init__(self, dm, residual, solverBuilder, dt=None, solverIsTimeDependent=False, explicitIslinearAndTimeIndependent=False):
        super(ImplicitEuler, self).__init__(dm, residual, solverBuilder,
                                            dt=dt,
                                            theta=1.,
                                            solverIsTimeDependent=solverIsTimeDependent,
                                            explicitIslinearAndTimeIndependent=explicitIslinearAndTimeIndependent)


class L1Scheme(Stepper):
    """
    L1 scheme for the Caputo fractional time derivative.
    """
    def __init__(self, alpha, maxTimeSteps, dm, residual, solverBuilder, dt=None, solverIsTimeDependent=False):
        from scipy.special import gamma
        super(L1Scheme, self).__init__(dm, residual, solverBuilder, dt, solverIsTimeDependent)
        assert not self.solverIsTimeDependent
        assert self.dt is not None
        assert 0 < alpha < 1.
        self.alpha = alpha
        self.maxTimeSteps = maxTimeSteps
        self.memory = self.dm.zeros(self.maxTimeSteps+1)
        self.b = (np.arange(1, self.maxTimeSteps+2)**(1-self.alpha) - np.arange(self.maxTimeSteps+1)**(1-self.alpha)) / gamma(2-self.alpha)
        self.solver = self.solverBuilder(0., 1., self.dt**self.alpha / self.b[0])
        self.rhs = self.dm.zeros()
        self.rhs2 = self.dm.zeros()
        self.k = 1

    def step(self, t, dt, u, forcingVector=None):
        if dt is None:
            dt = self.dt
        assert dt is not None
        assert self.k <= self.maxTimeSteps
        if not self.solverIsTimeDependent:
            assert dt == self.dt
        self.residualNew(t=t+dt,
                         u=None,
                         ut=None,
                         residual=self.rhs,
                         coeff_A=0.,
                         coeff_B=0.,
                         coeff_g=-dt**self.alpha/self.b[0])
        # self.forcing(t+dt, self.rhs)
        # self.rhs *= dt**self.alpha/self.b[0]
        # self.mass(t, u, self.rhs2)
        self.residualNew(t=t,
                         u=None,
                         ut=u,
                         residual=self.rhs2,
                         coeff_A=1.,
                         coeff_B=0.,
                         coeff_g=0.)
        if self.k == 1:
            self.memory[0].assign(self.rhs2)
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
        self.residualNew(t=t+dt,
                         u=None,
                         ut=u,
                         residual=self.memory[self.k],
                         coeff_A=1.,
                         coeff_B=0.,
                         coeff_g=0.)
        # self.mass(t+dt, u, self.memory[self.k])
        self.k += 1
        return t+dt


class fastL1Scheme(Stepper):
    """
    Fast L1 scheme for the Caputo fractional time derivative.
    """
    def __init__(self, alpha, maxTimeSteps, dm, residual, solverBuilder, dt=None, solverIsTimeDependent=False, eps=1e-4):
        from scipy.special import gamma
        super(fastL1Scheme, self).__init__(dm, residual, solverBuilder, dt, solverIsTimeDependent)
        assert not self.solverIsTimeDependent
        assert self.dt is not None
        assert 0 < alpha < 1.
        self.alpha = alpha
        self.maxTimeSteps = maxTimeSteps
        self.eps = eps
        self.s, self.w = self.getWeights()
        self.Nexp = self.w.shape[0]-1
        self.memory = self.dm.zeros(self.Nexp+1)
        self.solver = self.solverBuilder(0., 1., self.dt**self.alpha * gamma(2-self.alpha))
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

    def step(self, t, dt, u, forcingVector=None):
        from scipy.special import gamma
        if dt is None:
            dt = self.dt
        assert dt is not None
        assert self.k <= self.maxTimeSteps
        if not self.solverIsTimeDependent:
            assert dt == self.dt

        self.residualNew(t=t,
                         u=None,
                         ut=u,
                         residual=self.rhs2,
                         coeff_A=1.,
                         coeff_B=0.,
                         coeff_g=0.)
        # self.mass(t, u, self.rhs2)
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

        self.residualNew(t=t+dt,
                         u=None,
                         ut=None,
                         residual=self.rhs,
                         coeff_A=0.,
                         coeff_B=0.,
                         coeff_g=-dt**self.alpha * gamma(2-self.alpha))
        # self.forcing(t+dt, self.rhs)
        # self.rhs *= dt**self.alpha * gamma(2-self.alpha)

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


class IMEX(Stepper):
    def __init__(self,
                 dm,
                 residual,
                 solverBuilder,
                 c, bExpl, bImpl, AExpl, AImpl,
                 numSystemVectors=-1,
                 dt=None, solverIsTimeDependent=False, explicitIslinearAndTimeIndependent=False):
        """
        IMEX time stepping for

        f(t, u, u_t) = A(t, u_t) + I(t, u) + E(t, u) - g(t) = 0.

        We assume that A is linear in its second argument. Let

        mass:                  t, u           -> A(t, u)
        implicit:              t, u           -> I(t, u)
        explicit:              t, u           -> E(t, u)
        forcing:               t              -> g(t)
        implicitSolverBuilder: t, alpha, beta -> solver for alpha*A(t, u)+beta*I(t, u) = g

        t_k = t + c[k]*dt

        To take the step from u to u_new:

        Solve for U[k]:

        A(t_k, U[k]) + dt * A_impl[k, k] * I(t_k, U[k])
        = A(t, u) - dt * sum_j=0^{k-1} A_impl[k, j] * I(t_j, U[j])
                  - dt * sum_j=0^{k-1} A_expl[k, j] * E(t_j, U[j])
                  + dt * sum_j=0^{k-1} A_impl[k, j] * g(t_j)

        Finally, solve for u_new

        A(t+dt, u_new) = A(t, u) - dt * sum_k=0^s b_impl[k] * I(t_k, U[k])
                                 - dt * sum_k=0^s b_expl[k] * E(t_k, U[k])
                                 + dt * sum_k=0^s b_impl[k] * g(t_k)

        To save evaluations, set
        force[k]  := g(t_k)
        U_impl[k] := I(t_k, U[k])
        U_expl[k] := E(t_k, U[k])

        implicit: implicit part (t, u, r) that sets r with implicit function evaluation I at (t, u)
        explicit: explicit part (t, u, r) to set r with explicit function evaluation at (t, u)

        Defining the IMEX scheme
        c: time sub-steps for both implicit and explicit method
        bExpl: weights for explicit method
        bImpl: weights for implicit method
        AExpl: weights for explicit method
        AImpl: weights for implicit method
        numSystemVectors: size of the vectors of the solution u
        """
        super().__init__(dm, residual, solverBuilder, dt, solverIsTimeDependent, explicitIslinearAndTimeIndependent)
        self.c = c
        self.bExpl = bExpl
        self.bImpl = bImpl
        self.AExpl = AExpl
        self.AImpl = AImpl
        self.s = AExpl.shape[0]

        self._massSolve = None
        self._implicitSolve = None

        self.U = [dm.zeros(numSystemVectors) for _ in range(self.s)]
        self.UExpl = [dm.zeros(numSystemVectors) for _ in range(self.s)]
        self.UImpl = [dm.zeros(numSystemVectors) for _ in range(self.s)]
        self.rhs = dm.zeros(numSystemVectors)
        self.force = [dm.zeros(numSystemVectors) for _ in range(self.s)]

    def residualNew(self, t, u, ut, residual, coeff_A=1., coeff_I=1., coeff_E=1., coeff_g=1., coeff_residual=0., forcingVector=None):
        raise NotImplementedError()

    def getMassSolver(self, t):
        if self.solverIsTimeDependent:
            return self.solverBuilder(t, 1., 0.)
        else:
            if self._massSolve is None:
                self._massSolve = self.solverBuilder(t, 1., 0.)
            return self._massSolve

    def getImplicitSolver(self, t, alpha, beta):
        if self.solverIsTimeDependent:
            return self.solverBuilder(t, alpha, beta)
        else:
            if self._implicitSolve is None:
                self._implicitSolve = self.solverBuilder(t, alpha, beta)
            return self._implicitSolve

    def _stepOfPicard(self, t, dt, ut, unew, forcingVector=None):
        # Solve
        # uold := unew
        # 1/dt*(A(t+dt, unew) - A(t, ut)) + I(t+dt, unew) + E(t+dt, uold) - g(t+dt) = 0
        # for unew
        u = unew.copy()
        if dt is None:
            dt = self.dt
        assert dt is not None
        # evaluate forcing for all steps, assign to self.force
        for k in range(self.s):
            if self.AImpl[:, k].sum() != 0. or self.bImpl[k] != 0.:
                if forcingVector is None:
                    self.residualNew(t=t+self.c[k]*dt,
                                     u=None,
                                     ut=None,
                                     residual=self.force[k],
                                     coeff_A=0.,
                                     coeff_I=0.,
                                     coeff_E=0.,
                                     coeff_g=-1.)
                else:
                    self.residualNew(t=t+self.c[k]*dt,
                                     u=None,
                                     ut=None,
                                     residual=self.force[k],
                                     forcingVector=forcingVector[k],
                                     coeff_A=0.,
                                     coeff_I=0.,
                                     coeff_E=0.,
                                     coeff_g=-1.)
        # loop over steps
        for k in range(self.s):
            if np.absolute(self.AExpl[k, :]).max() == 0.:
                self.U[k].assign(u)
            else:
                # rhs := M*u
                # self.mass(t, u, self.rhs)
                self.residualNew(t + dt,
                                 u=None,
                                 ut=ut,
                                 residual=self.rhs,
                                 coeff_A=1.,
                                 coeff_I=0.,
                                 coeff_E=0.,
                                 coeff_g=0.)
                # rhs += dt * sum_j=0^{k-1} A_expl[k, j] * U_expl[j]
                # rhs += dt * sum_j=0^{k-1} A_impl[k, j] * U_impl[j]
                for j in range(k):
                    if self.AExpl[k, j] != 0:
                        self.rhs -= dt*self.AExpl[k, j]*self.UExpl[j]
                    if self.AImpl[k, j] != 0:
                        self.rhs -= dt*self.AImpl[k, j]*self.UImpl[j]
                # rhs += dt * sum_j=0^k A_impl[k, j] * force[j]
                for j in range(k+1):
                    if self.AImpl[k, j] != 0:
                        self.rhs += dt*self.AImpl[k, j]*self.force[j]

                # U[k] := implicitSolve of rhs
                implicit_solver = self.getImplicitSolver(t + self.c[k]*dt, 1., self.AImpl[k, k]*dt)
                if isinstance(implicit_solver, iterative_solver):
                    implicit_solver.setInitialGuess(u)
                implicit_solver(self.rhs, self.U[k])

            # evaluate explicit and implicit parts on U[k]
            if self.AExpl[:, k].sum() != 0. or self.bExpl[k] != 0.:
                self.residualNew(t + self.c[k]*dt,
                                 u=self.U[k],
                                 ut=None,
                                 residual=self.UExpl[k],
                                 coeff_A=0.,
                                 coeff_I=0.,
                                 coeff_E=1.,
                                 coeff_g=0.)
            if self.AImpl[:, k].sum() != 0. or self.bImpl[k] != 0.:
                self.residualNew(t + self.c[k]*dt,
                                 u=self.U[k],
                                 ut=None,
                                 residual=self.UImpl[k],
                                 coeff_A=0.,
                                 coeff_I=1.,
                                 coeff_E=0.,
                                 coeff_g=0.)
        # rhs := M*u
        self.residualNew(t + dt,
                         u=None,
                         ut=ut,
                         residual=self.rhs,
                         coeff_A=1.,
                         coeff_I=0.,
                         coeff_E=0.,
                         coeff_g=0.)
        # self.mass(t, u, self.rhs)
        # rhs += dt * sum_k b_expl[k] * U_expl[k]
        # rhs += dt * sum_k b_impl[k] * U_impl[k]
        for k in range(self.s):
            if self.bExpl[k] != 0.:
                self.rhs -= dt*self.bExpl[k]*self.UExpl[k]
            if self.bImpl[k] != 0.:
                self.rhs -= dt*self.bImpl[k]*self.UImpl[k]
        # rhs += sum_k b_impl[k] * force[k]
        for k in range(self.s):
            if self.bImpl[k] != 0.:
                self.rhs += dt*self.bImpl[k]*self.force[k]
        # unew := mass^{-1} rhs
        mass_solver = self.getMassSolver(t)
        if isinstance(mass_solver, iterative_solver):
            mass_solver.setInitialGuess(u)
        mass_solver(self.rhs, unew)
        return t+dt

    def picardStep(self, t, dt, unew, forcingVector=None, tol=1e-3):
        # Solve
        # uold := unew
        # 1/dt*(A(t+dt, unew) - A(t, uold)) + I(t+dt, unew) + E(t+dt, unew) - g(t+dt) = 0
        # for unew
        # by lagging the explicit term and applying Picard iteration.
        previous_timestep_u = unew.copy()
        picardIts = 0
        l2Norm = np.inf
        while l2Norm > tol:
            previous_picard_it = unew.copy()
            t_new = self._stepOfPicard(t, dt, previous_timestep_u, unew)
            diff = unew-previous_picard_it
            l2Norm = np.sqrt(sum([diff[k].norm()**2 for k in range(diff.numVectors)]))
            picardIts += 1
        return t_new, picardIts

    def step(self, t, dt, unew, forcingVector=None):
        return self._stepOfPicard(t, dt, unew, unew, forcingVector)


class EulerIMEX(IMEX):
    gamma = 1.

    def __init__(self, dm, residual, numSystemVectors=-1,
                 solverBuilder=None,
                 dt=None, solverIsTimeDependent=False, explicitIslinearAndTimeIndependent=False):
        AExpl = np.array([[0, 0],
                          [1, 0]])
        AImpl = np.array([[0, 0],
                          [0, 1]])
        bExpl = np.array([1, 0])
        bImpl = np.array([0, 1])
        c = np.array([0, 1])
        IMEX.__init__(self,
                      dm=dm,
                      residual=residual,
                      c=c,
                      bExpl=bExpl,
                      bImpl=bImpl,
                      AExpl=AExpl,
                      AImpl=AImpl,
                      numSystemVectors=numSystemVectors,
                      solverBuilder=solverBuilder,
                      dt=dt, solverIsTimeDependent=solverIsTimeDependent,
                      explicitIslinearAndTimeIndependent=explicitIslinearAndTimeIndependent)


class ARS3(IMEX):
    gamma = (3+np.sqrt(3))/6

    def __init__(self, dm, residual, solverBuilder, numSystemVectors=-1,
                 dt=None, solverIsTimeDependent=False, explicitIslinearAndTimeIndependent=False):
        gamma = self.gamma
        AExpl = np.array([[0, 0, 0],
                          [gamma, 0, 0],
                          [gamma-1, 2*(1-gamma), 0]])
        AImpl = np.array([[0, 0, 0],
                          [0, gamma, 0],
                          [0, 1-2*gamma, gamma]])
        bExpl = np.array([0, 1/2, 1/2])
        bImpl = np.array([0, 1/2, 1/2])
        c = np.array([0, gamma, 1-gamma])
        IMEX.__init__(self,
                      dm=dm,
                      residual=residual,
                      c=c,
                      bExpl=bExpl,
                      bImpl=bImpl,
                      AExpl=AExpl,
                      AImpl=AImpl,
                      numSystemVectors=numSystemVectors,
                      solverBuilder=solverBuilder,
                      dt=dt, solverIsTimeDependent=solverIsTimeDependent,
                      explicitIslinearAndTimeIndependent=explicitIslinearAndTimeIndependent)


class koto(IMEX):
    gamma = 1.

    def __init__(self, dm, residual, solverBuilder, numSystemVectors=-1,
                 dt=None, solverIsTimeDependent=False, explicitIslinearAndTimeIndependent=False):
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
        IMEX.__init__(self,
                      dm=dm,
                      residual=residual,
                      c=c,
                      bExpl=bExpl,
                      bImpl=bImpl,
                      AExpl=AExpl,
                      AImpl=AImpl,
                      numSystemVectors=numSystemVectors,
                      solverBuilder=solverBuilder,
                      dt=dt, solverIsTimeDependent=solverIsTimeDependent,
                      explicitIslinearAndTimeIndependent=explicitIslinearAndTimeIndependent)


timestepperFactory = factory()
timestepperFactory.register('Crank-Nicolson', CrankNicolson)
timestepperFactory.register('Implicit Euler', ImplicitEuler)
timestepperFactory.register('Explicit Euler', ExplicitEuler)
timestepperFactory.register('L1', L1Scheme, aliases=['L1'])
timestepperFactory.register('fast L1', fastL1Scheme, aliases=['fastL1'])
timestepperFactory.register('Euler IMEX', EulerIMEX, aliases=['euler_imex'])
timestepperFactory.register('ARS3 IMEX', ARS3, aliases=['ars3'])
timestepperFactory.register('Koto IMEX', koto, aliases=['koto'])
