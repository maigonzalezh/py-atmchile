# Contributing

## Environment requirements

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv)

```bash
# Install uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone https://github.com/maigonzalezh/py-atmchile.git
cd atmchile
uv sync --group dev
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

# With coverage
uv run test-cov
```

Coverage reports are written to `htmlcov/index.html`.

### Integration tests

Integration tests hit the live SINCA and DMC endpoints and are excluded from
the default run. Run them manually when you have network access:

```bash
# Run all integration tests
uv run pytest -m integration -v

# Run a single integration test
uv run pytest tests/test_integration.py::test_integration_air_quality_get_data -v
```

Tests skip automatically if the endpoint is unreachable.

## Linting and formatting

```bash
# Check for lint errors
uv run ruff check .

# Auto-fix lint errors + format
uv run ruff check . --fix
uv run ruff format .
```

CI runs both checks on every push and pull request — make sure they pass locally before opening a PR.

## Definition of Ready — gate a producción

Todo cambio que llega a producción pasa por una revisión humana obligatoria. No es
opcional y está aplicado por configuración del repositorio:

- **`main`**: no se permite push directo. Todo cambio entra por un pull request con
  el CI en verde, revisado por un humano en la UI del PR antes de mergear. La regla
  aplica también a administradores.
- **Publicación (tag `v*`)**: el push del tag dispara `publish.yml`, pero los jobs de
  publicación quedan **pausados** en los environments `pypi`/`testpypi` hasta que un
  humano aprueba el deployment manualmente desde la pestaña Actions.

## Pull request workflow

1. Fork the repository and create a feature branch from `main`.
2. Make your changes and add or update tests as needed.
3. Ensure `uv run ruff check . && uv run pytest` pass with no errors.
4. Open a pull request against `main` with a clear description of what changed and why.

## Updating dependencies

```bash
uv lock --upgrade   # upgrade lock file
uv sync --group dev  # sync environment
```

## Release workflow

The package version is derived automatically from git tags via `hatch-vcs` —
there is no `version` field in `pyproject.toml` to keep in sync.

1. On a `release/vX.Y.Z` branch off `main`, rename `[Unreleased]` in
   `CHANGELOG.md` to `[X.Y.Z] - YYYY-MM-DD` and open a PR against `main`.
2. After the PR is merged, tag the merge commit and push:
   ```bash
   git checkout main && git pull
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
3. The `publish.yml` workflow picks up the tag and builds the distribution
   (version computed from the tag). The publish jobs then **pause** on the
   `testpypi` and `pypi` environments — approve the deployment manually from the
   Actions tab to release to TestPyPI and PyPI and create the GitHub Release.
