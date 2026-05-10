.PHONY: setup download train run pipeline test lint docker-build docker-up docker-down clean

## ── Local development ────────────────────────────────────────────────────────

setup:            ## Install Python dependencies
	pip install -r requirements.txt

download:         ## Download the UCI dataset
	python scripts/download_data.py

train: download   ## Bootstrap: download → initial training
	python scripts/initial_train.py

run: train        ## Start API server (with background scheduler)
	python src/serving/api.py

pipeline:         ## Manually trigger one pipeline cycle
	python scripts/run_pipeline.py

test:             ## Run the test suite
	pytest tests/ -v

lint:             ## Basic import / syntax check
	python -m py_compile config.py \
		src/ingestion/data_generator.py \
		src/ingestion/ingestor.py \
		src/preprocessing/preprocessor.py \
		src/features/feature_engineer.py \
		src/training/trainer.py \
		src/inference/predictor.py \
		src/drift/drift_detector.py \
		src/serving/api.py \
		src/pipeline/scheduler.py
	@echo "All files compile OK."

## ── Docker ───────────────────────────────────────────────────────────────────

docker-build:     ## Build the Docker image
	docker-compose build

docker-up:        ## Start the stack (builds if needed)
	docker-compose up

docker-up-d:      ## Start the stack in the background
	docker-compose up -d

docker-down:      ## Stop and remove containers
	docker-compose down

docker-logs:      ## Tail container logs
	docker-compose logs -f

## ── Housekeeping ─────────────────────────────────────────────────────────────

clean:            ## Remove Python bytecode and __pycache__
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean."
