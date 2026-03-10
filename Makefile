PYTHON = python3
UV = uv
SRC = src
DEBUGGER = pdb

install:
	$(UV) sync

run:
	$(UV) run $(PYTHON) -m $(SRC)

debug:
	$(UV) run -m $(DEBUGGER) -m $(SRC)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	flake8 . --exclude=llm_sdk,.venv
	mypy . --warn-return-any --warn-unused-ignores \
			--ignore-missing-imports --disallow-untyped-defs \
			--check-untyped-defs
	
