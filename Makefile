SHELL = /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
.DEFAULT_GOAL := help

include .env
export $(shell sed 's/=.*//' .env)
export PYTHONPATH
export PIPENV_VENV_IN_PROJECT=1

PYTHON := python3
PIP := $(PYTHON) -m pip
PIPENV := $(PYTHON) -m pipenv
PYLINT := $(PIPENV) run pylint
BLACK := $(PIPENV) run black
MYPY := $(PIPENV) run mypy
ISORT := $(PIPENV) run isort

APP_NAME = nlp-sentiiment-analysis:0.0.1
APP_DIR = nlp_sentiiment_analysis
TEST_SRC = $(APP_DIR)/tests

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

### Local commands ###

venv:
	$(PIP) install -U pipenv
	$(PIPENV) shell

install-packages:
	pipenv install --dev

lint:
	$(PYLINT) $(APP_DIR) --exit-zero

format:
	$(BLACK) $(APP_DIR)
	$(ISORT) $(APP_DIR)

check-typing:
	$(MYPY) $(APP_DIR)

lint-and-format: ## Lint, format and static-check
	$(PYLINT) -E $(APP_DIR) --exit-zero
	$(MYPY) $(APP_DIR)
	$(BLACK) $(APP_DIR)
	$(ISORT) $(APP_DIR)

create-local-database-linux:
	sudo -u postgres psql -c 'create database $(DATABASE_NAME);'
	sudo -u postgres psql -c 'grant all privileges on database $(DATABASE_NAME) to $(DATABASE_USERNAME);'

create-local-database-mac:
	sudo mkdir -p /etc/paths.d && \
  	echo /Applications/Postgres.app/Contents/Versions/latest/bin \
  	| sudo tee /etc/paths.d/postgresapp

	sudo psql -U postgres -c 'create database $(DATABASE_NAME);'
	sudo psql -U postgres -c 'grant all privileges on database $(DATABASE_NAME) to $(DATABASE_USERNAME);'

drop-local-database:
	sudo psql -U postgres -c 'drop database $(DATABASE_NAME);'

run-local:
	$(PYTHON) $(APP_DIR)/manage.py migrate && python3 $(APP_DIR)/manage.py runserver

migrate:
	$(PYTHON) $(APP_DIR)/manage.py migrate --check --no-input

checkmigrations:
	$(PYTHON) $(APP_DIR)/manage.py makemigrations --check --no-input --dry-run

.PHONY: help venv install-packages lint format check-typing lint-and-format
	create-local-database-linux create-local-database-mac drop-local-database
	run-local migrate checkmigrations
