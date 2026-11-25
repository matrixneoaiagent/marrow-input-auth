PYTHON ?= python3

.PHONY: demo test generate clean

demo:
	$(PYTHON) -m marrow demo

test:
	$(PYTHON) -m pytest -q

generate:
	$(PYTHON) -m marrow generate --force

clean:
	rm -rf .pytest_cache __pycache__ marrow/__pycache__ tests/__pycache__
