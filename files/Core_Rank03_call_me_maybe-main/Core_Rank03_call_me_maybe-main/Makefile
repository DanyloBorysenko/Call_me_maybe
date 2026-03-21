PYTHON = python3
PIP = pip
ENV = uv
MAIN = call_me_maybe.py
ENV_NAME = venv
# CONFIG = config.txt
WORK_DIR = srcs
DEBUGGER = pdb
ALL_CACHE = $(shell find . -path "./.venv" -prune -o -name "__pycache__" -print -o -name ".mypy_cache" -print)
VENV_PATH = $(shell find . -type d -name ".$(ENV_NAME)" -print)
UV_CACHE_DIR = "/goinfre/cache/uv_cache"
HF_HOME = "/goinfre/cache/hf_cache"

install:
	@if [ -d ".$(ENV_NAME)" ]; then \
		echo "venv already exists"; \
	else \
		echo "Creating virtual environment..."; \
		$(ENV) $(ENV_NAME); \
		if [ -f "$(pyproject.toml)"]; then \
			echo "Installing dependencies..."; \
			$(ENV) $(PIP) install -r pyproject.toml; \
		else \
			echo "pyproject.toml file missing"; \
			echo "skipping package installation"; \
		fi; \
	fi

prep:
	mkdir -p $(UV_CACHE_DIR)
	mkdir -p $(HF_HOME)
	export UV_CACHE_DIR=$(UV_CACHE_DIR)
	export HF_HOME=$(HF_HOME)
	echo $(UV_CACHE_DIR)
	echo $(HF_HOME)

run:
	@if [ ! -d ".$(ENV_NAME)" ]; then \
		make install;\
	fi
	$(ENV) run $(PYTHON) -m src

visual:
	@if [ ! -d ".$(ENV_NAME)" ]; then \
		make install;\
	fi
	streamlit run visualizer.py

debug:
	@if [ ! -d ".$(ENV_NAME)" ]; then \
		make install;\
	fi
#	pdb debugger controls: n(ext), c(ont(inue)), exit
	$(ENV) run $(PYTHON) -m $(DEBUGGER) $(MAIN)

clean:
	@if [ -n "$(ALL_CACHE)" ]; then \
		echo "Cleaning workspace..."; \
		for cache in $(ALL_CACHE); do \
			echo "Removing $$cache"; \
			rm -rf "$$cache"; \
		done; \
		echo "Workspace cleaned"; \
	else \
		echo "Nothing to clean."; \
	fi

fclean: clean
	@if [ -n "$(VENV_PATH)" ]; then \
		echo "Removing virtual environment..."; \
		rm -rf "$(VENV_PATH)";\
		echo "Virtual environment removed"; \
	else \
		echo "There is no virtual environment";\
	fi

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores \
			--ignore-missing-imports --disallow-untyped-defs \
			--check-untyped-defs

.PHONY: install run debug clean fclean lint lint-strict