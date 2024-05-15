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

POSTGRES_COMMAND := /Applications/Postgres.app/Contents/Versions/latest/bin

APP_NAME = nlp-sentiment-analysis:0.0.1
APP_DIR = nlp_sentiment_analysis
TEST_SRC = $(APP_DIR)/tests

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

### Local commands ###

venv:
	$(PIP) install -U pipenv
	$(PIPENV) shell

install-packages:
	pipenv install --dev

create-local-database-linux:
	sudo -u postgres psql -c 'create database $(DATABASE_NAME);'
	sudo -u postgres psql -c 'grant all privileges on database $(DATABASE_NAME) to $(DATABASE_USERNAME);'

create-local-database-mac:
	sudo mkdir -p /etc/paths.d && \
  	echo $(POSTGRES_COMMAND) \
  	| sudo tee /etc/paths.d/postgresapp

	sudo $(POSTGRES_COMMAND)/psql -U postgres -c 'create database $(DATABASE_NAME);'
	sudo $(POSTGRES_COMMAND)/psql -U postgres -c 'grant all privileges on database $(DATABASE_NAME) to $(DATABASE_USERNAME);'

drop-local-database-linux:
	sudo psql -U postgres -c 'drop database $(DATABASE_NAME);'

drop-local-database-mac:
	sudo $(POSTGRES_COMMAND)/psql -U postgres -c 'drop database $(DATABASE_NAME);'

run-local:
	$(PYTHON) $(APP_DIR)/manage.py migrate && python3 $(APP_DIR)/manage.py runserver

migrate:
	$(PYTHON) $(APP_DIR)/manage.py migrate --check --no-input

test:
	$(PYTHON) $(APP_DIR)/manage.py test text_analysis

### Docker commands ###
up:
	docker compose up -d --build

down:
	docker compose down

test-docker:
	docker compose exec web python manage.py test text_analysis

copy-env:
	docker compose exec cp .env.example .env

.PHONY: help venv install-packages create-local-database-linux
	create-local-database-mac drop-local-database run-local migrate test up down
	test-docker copy-env
