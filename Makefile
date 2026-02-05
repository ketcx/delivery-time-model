.PHONY: format lint typecheck test all clean

# Format code with black
format:
	@echo "🎨 Formatting code with black..."
	black src/ tests/ --line-length 100

# Lint code with ruff
lint:
	@echo "🔍 Linting code with ruff..."
	ruff check src/ tests/

# Type check with mypy
typecheck:
	@echo "📋 Type checking with mypy..."
	mypy src/ --ignore-missing-imports

# Run tests with pytest
test:
	@echo "🧪 Running tests with pytest..."
	pytest tests/ -v

# Run all quality checks
all: format lint typecheck test
	@echo "✅ All quality checks passed!"

# Clean up cache and temporary files
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✨ Cleanup complete!"
