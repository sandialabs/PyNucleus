PYTHON ?= python3


example1:
	$(PYTHON) example1.py --export example1_stepMesh.py --finalTarget mesh
	$(PYTHON) example1.py --export example1_stepDoFMap.py --finalTarget dofmap
	$(PYTHON) example1.py --export example1_stepFunctions.py --finalTarget functions
	$(PYTHON) example1.py --export example1_stepSolvers.py --finalTarget solvers
	$(PYTHON) example1.py --export example1_stepErrors.py --finalTarget innerNorm

	$(PYTHON) example2.py --export example2_stepKernelFracInf.py --finalTarget kernelFracInf
	$(PYTHON) example2.py --export example2_stepMeshFracInf.py --finalTarget meshFracInf
	$(PYTHON) example2.py --export example2_stepSolveFracInf.py --finalTarget solveFracInf
	$(PYTHON) example2.py --export example2_stepFiniteHorizon.py --finalTarget finiteHorizon
