PYTHON ?= python3
VIRTUAL_ENV ?=
ifeq ($(VIRTUAL_ENV),)
  FLAGS ?= --no-use-pep517 -e
  PIP_FLAGS ?= --user
else
  PYTHON ?= python
  FLAGS ?= -e
  PIP_FLAGS ?=
endif
PIP_INSTALL_FLAGS ?=
TEST_RESULTS ?= index.html
MODULE_INSTALL_DIR ?= ${PWD}/modules/install
MODULES_DIR ?= ${PWD}/modules/modules
DATE = $(shell date +%y%m%d)
SHA = $(shell git describe --always --dirty --abbrev=40)
VERSION = ${DATE}-${SHA}


install :
	$(PYTHON) -m pip install $(PIP_INSTALL_FLAGS) packageTools/. && \
	$(PYTHON) -m pip install $(PIP_INSTALL_FLAGS) base/. && \
	$(PYTHON) -m pip install $(PIP_INSTALL_FLAGS) metisCy/.  && \
	$(PYTHON) -m pip install $(PIP_INSTALL_FLAGS) fem/.  && \
	$(PYTHON) -m pip install $(PIP_INSTALL_FLAGS) multilevelSolver/.  && \
	$(PYTHON) -m pip install $(PIP_INSTALL_FLAGS) nl/.  && \
	$(PYTHON) -m pip install $(PIP_INSTALL_FLAGS) .

module_lmod:
	rm -rf $(MODULE_INSTALL_DIR)/PyNucleus/$(VERSION)
	mkdir -p $(MODULE_INSTALL_DIR)/PyNucleus/$(VERSION)
	mkdir -p $(MODULES_DIR)/PyNucleus/
	$(PYTHON) makeModule.py $(MODULES_DIR)/PyNucleus/$(VERSION).lua $(MODULE_INSTALL_DIR)/PyNucleus/$(VERSION)
	PYTHON="PYTHONPATH=$(MODULE_INSTALL_DIR)/PyNucleus/$(VERSION):${PYTHONPATH} $(PYTHON)" PIP_INSTALL_FLAGS="--target=$(MODULE_INSTALL_DIR)/PyNucleus/$(VERSION)" PIP_FLAGS="" make prereq
	PYTHON="PYTHONPATH=$(MODULE_INSTALL_DIR)/PyNucleus/$(VERSION):${PYTHONPATH} $(PYTHON)" PIP_INSTALL_FLAGS="--target=$(MODULE_INSTALL_DIR)/PyNucleus/$(VERSION)" PIP_FLAGS="" make install


clean :
	$(PYTHON) -m pip uninstall PyNucleus_packageTools PyNucleus_base PyNucleus_metisCy PyNucleus_fem PyNucleus_multilevelSolver PyNucleus_nl

dev : dev_packageTools dev_base dev_metisCy dev_fem dev_multilevelSolver dev_nl dev_package

dev_packageTools :
	@ echo "Entering directory \`packageTools/'"
	cd packageTools; $(PYTHON) -m pip install $(FLAGS) .
dev_base_build :
	@ echo "Entering directory \`base'"
	cd base; $(PYTHON) -m pip install $(FLAGS) .
dev_base :
	make dev_base_build
	$(PYTHON) -c "import PyNucleus_base" || (make clean_base && make dev_base_build)
dev_metisCy_build :
	@ echo "Entering directory \`metisCy'"
	cd metisCy; $(PYTHON) -m pip install $(FLAGS) .
dev_metisCy :
	make dev_metisCy_build
	$(PYTHON) -c "import PyNucleus_metisCy" || (make clean_metisCy && make dev_metisCy_build)
dev_fem_build :
	@ echo "Entering directory \`fem'"
	cd fem; $(PYTHON) -m pip install $(FLAGS) .
dev_fem :
	make dev_fem_build
	$(PYTHON) -c "import PyNucleus_fem" || (make clean_fem && make dev_fem_build)
dev_multilevelSolver_build :
	@ echo "Entering directory \`multilevelSolver'"
	cd multilevelSolver; $(PYTHON) -m pip install $(FLAGS) .
dev_multilevelSolver :
	make dev_multilevelSolver_build
	$(PYTHON) -c "import PyNucleus_multilevelSolver" || (make clean_multilevelSolver && make dev_multilevelSolver_build)
dev_nl_build :
	@ echo "Entering directory \`nl'"
	cd nl; $(PYTHON) -m pip install $(FLAGS) .
dev_nl :
	make dev_nl_build
	$(PYTHON) -c "import PyNucleus_nl" || (make clean_nl && make dev_nl_build)
dev_package :
	$(PYTHON) -m pip install $(FLAGS) .



clean_dev: clean_package clean_packageTools clean_base clean_metisCy clean_fem clean_multilevelSolver clean_nl
clean_packageTools :
	$(PYTHON) -m pip uninstall PyNucleus_packageTools -y
clean_base :
	$(PYTHON) -m pip uninstall PyNucleus_base -y
	cd base/PyNucleus_base && \
           rm -f *.so *.c *.pyc && \
	   rm -rf __pycache__
	cd base && rm -rf build __pycache__ *.egg-info
clean_metisCy :
	$(PYTHON) -m pip uninstall PyNucleus_metisCy -y
	cd metisCy/PyNucleus_metisCy && \
	   rm -f *.so *.c *.pyc && \
	   rm -rf __pycache__
	cd metisCy && rm -rf build __pycache__ *.egg-info
clean_fem :
	$(PYTHON) -m pip uninstall PyNucleus_fem -y
	cd fem/PyNucleus_fem && \
           rm -f *.so *.c *.pyc && \
	   rm -rf __pycache__
	cd fem && rm -rf build __pycache__ *.egg-info
clean_multilevelSolver :
	$(PYTHON) -m pip uninstall PyNucleus_multilevelSolver -y
	cd multilevelSolver/PyNucleus_multilevelSolver && \
           rm -f *.so *.c *.pyc && \
	   rm -rf __pycache__
	cd multilevelSolver && rm -rf build __pycache__ *.egg-info
clean_nl :
	$(PYTHON) -m pip uninstall PyNucleus_nl -y
	cd nl/PyNucleus_nl && \
	   rm -rf *.so *.c *.pyc *.html __pycache__ kernelsCy.cpp adaptiveQuad.cpp
	cd nl && rm -rf build __pycache__ *.egg-info
clean_package :
	$(PYTHON) -m pip uninstall PyNucleus -y

.PHONY: docs
docs :
	$(PYTHON) -m sphinx -b html docs docs/build
	find docs/build/_downloads -name "*.ipynb" | xargs -I {} cp {} examples

clean_docs :
	cd docs; rm -rf build

createVirtualEnv:
	$(PYTHON) -m virtualenv --python=$(PYTHON) venv --system-site-packages


list-tests:
	$(PYTHON) -m pytest --collect-only tests/

.PHONY: tests
tests:
	$(PYTHON) -m pytest -rA --html=$(TEST_RESULTS) --self-contained-html tests/

prereq:
	$(PYTHON) -m pip install $(PIP_FLAGS) $(PIP_INSTALL_FLAGS) wheel Cython cython numpy scipy matplotlib pyyaml h5py pybind11 MeshPy tabulate modepy mpi4py pyamg meshio
	$(PYTHON) -m pip install $(PIP_FLAGS) $(PIP_INSTALL_FLAGS) scikit-sparse

prereq-extra:
	$(PYTHON) -m pip install $(PIP_FLAGS) pytest pytest-html pytest-xdist Sphinx sphinxcontrib-programoutput sphinx-gallery sphinx-rtd-theme flake8 flake8-junit-report cython-lint

flake8:
	$(PYTHON) -m flake8 --output-file=flake8.txt --exit-zero drivers examples packageTools base metisCy fem multilevelSolver nl tests
	flake8_junit flake8.txt flake8.xml
	rm flake8.txt

cython-lint:
	- cython-lint --max-line-length=160 --ignore="E741" drivers examples packageTools base metisCy fem multilevelSolver nl tests > cython-lint.txt 2>&1
	flake8_junit cython-lint.txt cython-lint.xml
	rm cython-lint.txt
	sed 's/name="flake8"/name="cython-lint"/g' cython-lint.xml > cython-lint2.xml
	mv cython-lint2.xml cython-lint.xml
