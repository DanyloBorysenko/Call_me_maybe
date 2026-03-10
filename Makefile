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
	
