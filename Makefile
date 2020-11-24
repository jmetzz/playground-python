BLACK ?= \033[0;30m
RED ?= \033[0;31m
GREEN ?= \033[0;32m
YELLOW ?= \033[0;33m
BLUE ?= \033[0;34m
PURPLE ?= \033[0;35m
CYAN ?= \033[0;36m
GRAY ?= \033[0;37m
COFF ?= \033[0m

.PHONY: docker dockerd docker-psql deps check test clean lint bastion
.EXPORT_ALL_VARIABLES:

GCP_PROJECT=
GCP_ZONE=europe-west3-c
TARGET_HOST_NAME=
DB_PORT=5432


###################
# Docker commands #
###################


## Bring up the stack
docker:
	@docker-compose down --remove-orphans
	@docker-compose up

## Bring up the stack (daemonized)
dockerd:
	@docker-compose down --remove-orphans
	@docker-compose up -d

## Jump into postgres shell, requires running postgres container
docker-psql:
	@printf "$(CYAN)entering psql shell (requires running db container)$(COFF)\n"
	@docker-compose exec -u postgres db psql <db-name> <db-user>

##################
# Local commands #
##################


## Install project dependencies
deps:
	@printf "$(CYAN)Updating git submodules and python deps$(COFF)\n"
	pip3 install -U pip poetry docker-compose
	poetry install

## Run static code checkers and linters
check:
	@printf "$(CYAN)Running static code analysis, tests and license generation$(COFF)\n"
	poetry run flake8
	poetry run black --check .
	poetry run mypy . --ignore-missing-imports
	poetry run pip-licenses --with-authors -f markdown --output-file licenses.md

## Run unit tests
test:
	@printf "$(CYAN)Running test suite$(COFF)\n"
	poetry run pytest

## Remove temporary files created during build and all compiled Python files
clean:
	@printf "$(CYAN)Cleaning EVERYTHING!$(COFF)\n"
	docker-compose down --remove-orphans
	docker volume prune -f

## Runs black formatter
lint:
	@printf "$(CYAN)Auto-formatting with black$(COFF)\n"
	poetry run black .

## Connect to the dev db with a port FWD (Broadcasts on local 12.0.0.1:5432)
bastion:
	@printf "$(GREEN)Postgres will be listening on 127.0.0.1:5432$(COFF)\n"
	gcloud compute ssh test-connectivity-vm --project "$(GCP_PROJECT)"  --zone "$(GCP_ZONE)" --ssh-flag="-L 127.0.0.1:$(DB_PORT):$(TARGET_HOST_NAME):$(DB_PORT) -Nv"

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
