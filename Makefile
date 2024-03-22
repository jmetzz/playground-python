.PHONY: help
.EXPORT_ALL_VARIABLES:
SHELL := /bin/bash -euo pipefail

BLACK ?= \033[0;30m
RED ?= \033[0;31m
GREEN ?= \033[0;32m
YELLOW ?= \033[0;33m
BLUE ?= \033[0;34m
PURPLE ?= \033[0;35m
CYAN ?= \033[0;36m
GRAY ?= \033[0;37m
COFF ?= \033[0m


##################
#    Targets     #
##################

initialize:
ifneq ($(wildcard .git),)
	@printf "$(CYAN)>>> Repository already initialized.$(COFF)\n"
else
	@printf "$(CYAN)>>> Initializing git repository.$(COFF)\n"
	git init
endif


deps: initialize ## Install dependencies, including dev & test dependencies
	@printf "$(CYAN)>>> Creating environment for project...$(COFF)\n"
	poetry install --no-root --without profiling --sync
	poetry run pre-commit install
	poetry run pre-commit install --hook-type pre-push


test: ## Run unit tests
	@printf "$(CYAN)Running test suite$(COFF)\n"
	export PYTHONPATH="./src" && poetry run pytest --cov=src


testchanges:  ## Run unit tests only for changed files and new files
	@printf "$(CYAN)Running tests for changed and new files$(COFF)\n"
	$(eval CHANGED_FILES := $(shell git diff --name-only | grep -E '^src/.*\.py$$' | sed 's/src\//tests\//g; s/\.py$$/_test.py/'))
	$(eval NEW_TEST_FILES := $(shell git ls-files --others --exclude-standard | grep -E '^tests/.*_test\.py$$'))
	@if [ -n "$(CHANGED_FILES)" ]; then \
		echo $(CHANGED_FILES) | xargs -n 1 poetry run pytest; \
	fi
	@if [ -n "$(NEW_TEST_FILES)" ]; then \
		echo $(NEW_TEST_FILES) | xargs -n 1 poetry run pytest; \
	fi


check: ## Run static code checkers and linters
	@printf "$(CYAN)Running static code analysis and license generation$(COFF)\n"
	poetry run ruff check src tests
	@printf "All $(GREEN)done$(COFF)\n"



lint: ## Runs ruff formatter
	@printf "$(CYAN)Auto-formatting with ruff$(COFF)\n"
	poetry run ruff check src tests --fix
	poetry run ruff format src tests notebooks

license: ## Generated the licenses.md file based on the project's dependencies
	@printf " >>> Generating $(CYAN)licenses.md$(COFF) file\n"
	poetry run pip-licenses --with-authors -f markdown --output-file ./licenses.md


clean: ## Removed the build, dist directories, pycache, pyo or pyc and swap files
	@printf "$(CYAN)Cleaning EVERYTHING!$(COFF)\n"
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type f -name '*.py[co]' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -type f -name '.DS_Store' -delete
	@printf "$(GREEN)>>> Removed$(COFF) pycache, .pyc, .pyo, .DS_Store files and files with ~\n"


all: clean lint test ## Runs clean, lint, and test target commands
	@printf "$(GREEN)>>> Done$(COFF) ~\n"


bastion: ## Connect to the dev db with a port FWD (Broadcasts on local 12.0.0.1:5432)
	@printf "$(GREEN)Postgres will be listening on 127.0.0.1:5432$(COFF)\n"
	gcloud compute ssh test-connectivity-vm --project "$(GCP_PROJECT)"  --zone "$(GCP_ZONE)" --ssh-flag="-L 127.0.0.1:$(DB_PORT):$(TARGET_HOST_NAME):$(DB_PORT) -Nv"


to_s3: ## Upload Data to S3
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif


from_s3: ## Download Data from S3
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################
.DEFAULT_GOAL := help
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(CYAN)%-30s$(COFF) %s\n", $$1, $$2}'
