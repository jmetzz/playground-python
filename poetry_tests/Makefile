.PHONY: run docker-run check clean deps interpreter lint
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

# variables
PYTHON_VERSION = 3.9.2
PYTHON_INTERPRETER = python3
PROFILE = default
PROJECT_NAME = poetry_tests
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

#################################################################################
# COMMANDS                                                                      #
#################################################################################

ifeq (,$(shell which pyenv))
HAS_PYENV=False
else
HAS_PYENV=True
endif


install_pyenv:
ifeq (,$(shell which pyenv))
	@printf "$(CYAN)>>> Installing pyenv ...$(COFF)\n"
	curl https://pyenv.run | bash
else
	@printf "$(CYAN)>>> pyenv is already installed$(COFF)\n"
endif


## Install the necessary python version
interpreter: install_pyenv
	@printf "$(CYAN)>>> Intalling required python version, if not available yet...$(COFF)\n"
	@pyenv install -s $(PYTHON_VERSION)
	@pyenv local $(PYTHON_VERSION)
	pip3 install --user -U pip poetry
	@printf "$(CYAN)>>> Done$(COFF)\n"


## Install Dev & Test Dependencies
deps:
	@printf "$(CYAN)>>> Creating environment for project...$(COFF)\n"
	poetry install

## Run cli.py to check
run:
	@printf "$(CYAN)Running test suite$(COFF)\n"
	poetry run python cli.py check
	poetry run python cli.py check -n 3

## Run inside docker container
docker-run:
	@printf "$(CYAN)Build docker image and running the container$(COFF)\n"
	docker build --build-arg PIP_CONF="$(cat ~/.config/pip/pip.conf)" -t poetry_test -f Dockerfile .
	docker run poetry_test

## Run static code checkers and linters
check:
	@printf "$(CYAN)Running static code analysis and license generation$(COFF)\n"
	poetry run isort --check --color src
	poetry run black --check *.py src
	poetry run flake8 src
	poetry run pip-licenses --with-authors -f markdown --output-file ./licenses.md

## Runs black formatter
lint:
	@printf "$(CYAN)Auto-formatting with black$(COFF)\n"
	poetry run isort src
	poetry run black *.py src

## Remove temporary files created during build and all compiled Python files
clean:
	@printf "$(CYAN)Cleaning EVERYTHING!$(COFF)\n"
	@find . -type f -name "*.py[co]" -delete
	@find . -type f -name ".DS_Store" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -name '*~' -delete
	@printf "$(CYAN)>>> Done$(COFF)\n"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help
# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

