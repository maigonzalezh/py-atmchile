# Contributing

## Environment requirements

- Python >= 3.9
- [uv](https://github.com/astral-sh/uv)

```bash
# Install uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone <repository-url>
cd atmchile
uv sync --all-extras
```

## Running tests

```bash
# All tests
uv run pytest

# With verbose output
uv run pytest -v

# Single test file
uv run pytest tests/test_climate_data.py

# Single test
uv run pytest tests/test_climate_data.py::test_init_with_default_path

# With coverage (minimum threshold: 80%)
uv run test-cov
```

Coverage reports are written to `htmlcov/index.html`.

## Linting and formatting

```bash
# Check for lint errors
uv run ruff check .

# Auto-fix lint errors + format
uv run ruff check . --fix
uv run ruff format .
```

CI runs both checks on every push and pull request — make sure they pass locally before opening a PR.

## Pull request workflow

1. Fork the repository and create a feature branch from `main`.
2. Make your changes and add or update tests as needed.
3. Ensure `uv run ruff check . && uv run pytest` pass with no errors.
4. Open a pull request against `main` with a clear description of what changed and why.

## Updating dependencies

```bash
uv lock --upgrade   # upgrade lock file
uv sync --all-extras  # sync environment
```
