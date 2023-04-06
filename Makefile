VENV_NAME = venv
PYTHON = $(VENV_NAME)/bin/python
PIP = $(VENV_NAME)/bin/pip
VERSION_FILE = VERSION
CHANGELOG_FILE = CHANGELOG.md
GIT_COMMIT = $(shell git rev-parse HEAD)
CURRENT_VERSION = $(shell cat $(VERSION_FILE))
MAJOR = $(shell echo $(CURRENT_VERSION) | cut -d. -f1)
MINOR = $(shell echo $(CURRENT_VERSION) | cut -d. -f2)
PATCH = $(shell echo $(CURRENT_VERSION) | cut -d. -f3)
NEXT_PATCH = $(shell echo $$(($(PATCH) + 1)))
NEXT_MINOR = $(shell echo $$(($(MINOR) + 1)))
NEW_VERSION ?= $(shell if [ $(NEXT_PATCH) -eq 10 ]; then \
	if [ $(NEXT_MINOR) -eq 10 ]; then \
		echo "$$(($(MAJOR) + 1)).0.0"; \
	else \
		echo "$(MAJOR).$(NEXT_MINOR).0"; \
	fi; \
else \
	echo "$(MAJOR).$(MINOR).$(NEXT_PATCH)"; \
fi)

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: venv
# Check if Graphviz is installed
ifeq (,$(shell which dot))
	brew install graphviz
endif
venv: $(VENV_NAME)/bin/activate ## Create virtual environment

$(VENV_NAME)/bin/activate: requirements.txt
	test -d $(VENV_NAME) || python3 -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
	$(PIP) install --upgrade wheel --trusted-host pypi.org --trusted-host files.pythonhosted.org
	$(PIP) install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
	touch $(VENV_NAME)/bin/activate

.PHONY: build
build: venv ## Build package
	$(PYTHON) setup.py build

.PHONY: create_dirs
create_dirs: ## Create cache directories for config and data
	mkdir -p $(HOME)/.project_allison/data
	mkdir -p $(HOME)/.project_allison/auth
	mkdir -p $(HOME)/.project_allison/templates
	mkdir -p $(HOME)/.project_allison/static
	mkdir -p $(HOME)/.project_allison/static/tmp
	cp -r project_allison/data/* $(HOME)/.project_allison/data
	cp -r project_allison/auth/* $(HOME)/.project_allison/auth
	cp -r project_allison/templates/* $(HOME)/.project_allison/templates
	cp -r project_allison/static/* $(HOME)/.project_allison/static
	cp .env $(HOME)/.project_allison/.env

.PHONY: install
install: venv create_dirs ## Install packages under virtual environment and create cache directories
	$(PYTHON) setup.py install

.PHONY: clean
clean: ## Clean up cache directories and virtual environment, build
	rm -rf $(VENV_NAME) build dist *.egg-info
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +

.PHONY: run
run: venv ## Run project_allison
	$(PYTHON) project_allison/__main__.py

# Targets
.PHONY: changelog tag release

changelog:
	@echo "Generating changelog for version $(NEW_VERSION)..."
	git-chglog -o CHANGELOG.md

tag:
	@echo "Tagging version $(NEW_VERSION)..."
	@echo "$(NEW_VERSION)" > $(VERSION_FILE)
	@git add $(VERSION_FILE) $(CHANGELOG_FILE)
	@git commit -m "Release version $(NEW_VERSION)"
	@git tag -a "v$(NEW_VERSION)" -m "Version $(NEW_VERSION)"
	@echo "Version $(NEW_VERSION) tagged."

release: changelog tag
	@echo "Releasing version $(NEW_VERSION)..."
	@git push origin "v$(NEW_VERSION)"
	@echo "Version $(NEW_VERSION) released."
