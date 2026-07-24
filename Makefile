.PHONY: help install dev test lint format clean docker-build docker-run stop

help:
	@echo "Leaf Disease Detection - Development Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          - Install dependencies"
	@echo "  make dev-install      - Install with development tools"
	@echo ""
	@echo "Development:"
	@echo "  make dev              - Run development server"
	@echo "  make train            - Run model training"
	@echo "  make test             - Run tests with coverage"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - Run linting checks"
	@echo "  make format           - Format code with black"
	@echo "  make format-check     - Check code formatting"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container"
	@echo "  make docker-compose   - Run with docker-compose"
	@echo "  make stop             - Stop Docker containers"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            - Clean build artifacts"
	@echo "  make logs             - View application logs"
	@echo "  make git-log          - View git history"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

dev-install: install
	pip install black flake8 pytest pytest-cov pytest-mock

dev:
	python app.py

train:
	python src/train.py --data-dir dataset --epochs 20 --batch-size 32

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ app.py --max-line-length=120

format:
	black src/ app.py tests/

format-check:
	black --check src/ app.py tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf build/ dist/ *.egg-info

docker-build:
	docker build -t leaf-disease-detection:latest .

docker-run: docker-build
	docker run -p 5000:5000 \
		-v $$(pwd)/models:/app/models \
		-v $$(pwd)/logs:/app/logs \
		-v $$(pwd)/static/uploads:/app/static/uploads \
		-e FLASK_ENV=production \
		leaf-disease-detection:latest

docker-compose:
	docker-compose up -d

stop:
	docker-compose down

logs:
	tail -f logs/app_*.log

git-log:
	git log --oneline -15

.DEFAULT_GOAL := help
