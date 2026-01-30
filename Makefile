# Enterprise Agentic RAG Platform - Makefile
# Common commands for development and deployment

.PHONY: help start stop restart logs health test clean setup dev prod

# Default target
help:
	@echo "Enterprise Agentic RAG Platform"
	@echo "================================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup        - Install dependencies and setup environment"
	@echo "  make env          - Copy .env.example to .env"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make start        - Start all services"
	@echo "  make stop         - Stop all services"
	@echo "  make restart      - Restart all services"
	@echo "  make logs         - View service logs"
	@echo "  make logs-f       - Follow service logs"
	@echo "  make status       - Show service status"
	@echo ""
	@echo "Development Commands:"
	@echo "  make dev          - Start with dev tools (mongo-express, redis-commander)"
	@echo "  make dev-backend  - Run backend locally (not in Docker)"
	@echo "  make shell        - Open shell in backend container"
	@echo ""
	@echo "Testing Commands:"
	@echo "  make test         - Run all tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make lint         - Run linting"
	@echo ""
	@echo "Database Commands:"
	@echo "  make db-shell     - Open MongoDB shell"
	@echo "  make db-seed      - Seed database with sample data"
	@echo "  make db-reset     - Reset database (WARNING: deletes all data)"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make health       - Check all service health"
	@echo "  make clean        - Clean up containers and volumes"
	@echo "  make pull-model   - Pull Ollama model"

# ============================================
# Setup
# ============================================

setup:
	@echo "Setting up Enterprise Agentic RAG Platform..."
	@cp -n .env.example .env 2>/dev/null || true
	@pip install -e ".[dev]" 2>/dev/null || pip install -r backend/requirements.txt
	@echo "Setup complete! Run 'make start' to start services."

env:
	@cp -n .env.example .env 2>/dev/null || echo ".env already exists"
	@echo "Environment file ready. Edit .env as needed."

# ============================================
# Docker Commands
# ============================================

start:
	@echo "Starting services..."
	docker compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	@make health

stop:
	@echo "Stopping services..."
	docker compose down

restart:
	@echo "Restarting services..."
	docker compose restart

logs:
	docker compose logs --tail=100

logs-f:
	docker compose logs -f

status:
	docker compose ps

# Start with dev tools
dev:
	@echo "Starting services with dev tools..."
	docker compose --profile dev up -d
	@echo ""
	@echo "Dev tools available at:"
	@echo "  - Mongo Express: http://localhost:8081"
	@echo "  - Redis Commander: http://localhost:8082"

# ============================================
# Development
# ============================================

dev-backend:
	@echo "Starting backend locally..."
	cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

shell:
	docker compose exec backend /bin/bash

# ============================================
# Testing
# ============================================

test:
	@echo "Running tests..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=backend --cov-report=html --cov-report=term

lint:
	@echo "Running linting..."
	ruff check backend/
	mypy backend/ --ignore-missing-imports

# ============================================
# Database
# ============================================

db-shell:
	docker compose exec mongodb mongosh -u raguser -p ragpassword enterprise_rag

db-seed:
	@echo "Seeding database..."
	docker compose exec mongodb mongosh -u raguser -p ragpassword enterprise_rag /docker-entrypoint-initdb.d/mongo-init.js

db-reset:
	@echo "WARNING: This will delete all data!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	docker compose down -v
	docker compose up -d mongodb
	@sleep 5
	@make db-seed

# ============================================
# Utility
# ============================================

health:
	@echo "Checking service health..."
	@echo ""
	@echo "Backend API:"
	@curl -s http://localhost:8000/api/v1/health | python -m json.tool 2>/dev/null || echo "  Backend not responding"
	@echo ""
	@echo "MongoDB:"
	@docker compose exec -T mongodb mongosh --eval "db.adminCommand('ping')" --quiet 2>/dev/null | head -1 || echo "  MongoDB not responding"
	@echo ""
	@echo "Redis:"
	@docker compose exec -T redis redis-cli ping 2>/dev/null || echo "  Redis not responding"
	@echo ""
	@echo "Ollama:"
	@curl -s http://localhost:11434/api/tags | python -m json.tool 2>/dev/null | head -5 || echo "  Ollama not responding"

clean:
	@echo "Cleaning up..."
	docker compose down -v --remove-orphans
	docker system prune -f
	rm -rf __pycache__ .pytest_cache .coverage htmlcov .mypy_cache

pull-model:
	@echo "Pulling Ollama model (llama3.2:3b)..."
	docker compose exec ollama ollama pull llama3.2:3b
	@echo "Model pulled successfully!"

# ============================================
# Build
# ============================================

build:
	@echo "Building Docker images..."
	docker compose build

build-no-cache:
	@echo "Building Docker images (no cache)..."
	docker compose build --no-cache

# ============================================
# Production
# ============================================

prod:
	@echo "Starting in production mode..."
	docker compose -f docker-compose.yml up -d
	@echo "Production services started."
