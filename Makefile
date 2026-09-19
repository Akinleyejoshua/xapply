# XApply - installation and startup flow
# Usage: make install && make login && make run
.DEFAULT_GOAL := help
PYTHON_VERSION ?= 3.12
VENV ?= .venv
UV := $(shell command -v uv 2>/dev/null)

ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
else
	BIN := $(VENV)/bin
endif
PY  := $(BIN)/python
PIP := $(BIN)/pip

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(VENV)/pyvenv.cfg:
ifdef UV
	uv venv --python $(PYTHON_VERSION) $(VENV)
else
	python3 -m venv $(VENV)
endif

venv: $(VENV)/pyvenv.cfg  ## Create the virtual environment (uv if available, else venv)

deps: venv  ## Install Python dependencies
ifdef UV
	uv pip install --python $(PY) -r requirements.txt
else
	$(PIP) install --upgrade pip && $(PIP) install -r requirements.txt
endif

browsers: deps  ## Download Playwright Chromium (PDF rendering + fallback browser)
	$(PY) -m playwright install chromium

env:  ## Create .env from .env.example if it does not exist
	@test -f .env || cp .env.example .env
	@grep -Eq '^GEMINI_API_KEY=.+' .env || echo ">> Add your GEMINI_API_KEY to .env before running"

dirs:
	@mkdir -p output_resumes logs

install: deps browsers env dirs check  ## Full setup: venv, deps, browsers, .env, smoke test
	@echo ""
	@echo "Setup complete. Next steps:"
	@echo "  1. Edit profile.json with your real profile"
	@echo "  2. make serve      web dashboard: scan boards, apply, review (no login needed)"
	@echo "     or make scan    preview postings in the terminal"
	@echo "     or make run     assisted mode in the terminal"
	@echo "  3. make login      only if you want the LinkedIn source"

login: dirs  ## Open the persistent browser so you can log in to LinkedIn once (LinkedIn only)
	$(PY) main.py login

scan: dirs  ## Preview what the configured sources would find (applies to nothing)
	$(PY) main.py discover

scan-save: dirs  ## Scan and write the URLs to jobs.txt
	$(PY) main.py discover --save jobs.txt

models: ## List the models the current provider offers
	$(PY) main.py models

settings: ## Show the settings in force and where each came from
	$(PY) main.py settings

settings-reset: ## Forget the settings saved from the dashboard, back to .env
	$(PY) main.py settings --reset

companies: ## Show the company board tokens and count their open roles
	$(PY) main.py companies --probe

run: dirs  ## Run the pipeline in assisted mode (you click Submit)
	$(PY) main.py run

run-auto: dirs  ## Run the pipeline with AUTO_SUBMIT (bot clicks Submit)
	$(PY) main.py run --auto-submit

analyze: dirs  ## Dry-run one posting: make analyze URL=https://www.linkedin.com/jobs/view/123/
	$(PY) main.py analyze --url "$(URL)"

PORT ?= 8000
serve: dirs  ## Start the web dashboard. Override the port with: make serve PORT=8001
	$(PY) main.py serve --port $(PORT)

ui: serve  ## Alias for `make serve`

db-init:  ## Create the SQLite schema
	$(PY) main.py init-db

check: deps  ## Byte-compile and import-check every module
	$(PY) -m compileall -q .
	$(PY) -c "import config, models, database, llm, ai_agent, resume_builder, browser_bot, appliers, job_search, discovery, pipeline, reports, api, main; print('imports OK')"

test: deps  ## Run the offline test-suite
	$(PY) -m pytest -q tests

clean:  ## Remove caches (keeps DB, resumes and browser session)
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache

reset: clean  ## DANGER: remove venv, DB, generated resumes, logs and browser session
	rm -rf $(VENV) applications.db output_resumes/*.pdf logs/*.log .browser_profile settings.local.json

.PHONY: help venv deps browsers env dirs install login scan scan-save models settings settings-reset companies run run-auto \
	analyze serve ui db-init check test clean reset
