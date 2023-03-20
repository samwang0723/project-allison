VENV_NAME = venv
PYTHON = $(VENV_NAME)/bin/python
PIP = $(VENV_NAME)/bin/pip

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: venv
venv: $(VENV_NAME)/bin/activate ## Create virtual environment

$(VENV_NAME)/bin/activate: requirements.txt
	test -d $(VENV_NAME) || python3 -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade wheel
	$(PIP) install -r requirements.txt
	touch $(VENV_NAME)/bin/activate

.PHONY: build
build: venv ## Build package
	$(PYTHON) setup.py build

.PHONY: create_dirs
create_dirs: ## Create cache directories for config and data
	mkdir -p $(HOME)/.jarvis/data
	mkdir -p $(HOME)/.jarvis/voice_records
	mkdir -p $(HOME)/.jarvis/auth
	mkdir -p $(HOME)/.jarvis/exe
	cp jarvis/voice_output.py $(HOME)/.jarvis/exe/voice_output.py

.PHONY: install
install: venv create_dirs ## Install packages under virtual environment and create cache directories
	if ! brew list portaudio >/dev/null 2>&1; then \
		brew install portaudio; \
		sudo ln -s /opt/homebrew/lib/libspeexdsp.dylib /usr/local/lib/libspeexdsp.dylib; \
	fi
	$(PYTHON) setup.py install

.PHONY: clean
clean: ## Clean up cache directories and virtual environment, build
	rm -rf $(VENV_NAME) build dist *.egg-info
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +
	rm -rf $(HOME)/.jarvis

.PHONY: run
run: venv ## Run jarvis
	$(PYTHON) jarvis/__main__.py