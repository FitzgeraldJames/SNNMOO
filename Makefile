PYTHONPATH=./
PYTHON=python3

MODULES=snnmoo

type:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mypy --config-file=mypy.ini $(MODULES)

.PHONY: analysis
analysis:
	(cd analysis; PYTHONPATH=../$(PYTHONPATH) $(PYTHON) -m jupyter notebook )

