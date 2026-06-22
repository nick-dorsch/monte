#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  deploy/publish.sh
  deploy/publish.sh --yes

Runs the release checks, builds the package, validates the artifacts, tests a local
wheel install, asks for confirmation, then uploads to PyPI.

Options:
  --yes, -y   Skip the interactive confirmation prompt
  --help, -h  Show this help

Authentication is handled by twine. For API-token auth, use username __token__
and your PyPI token as the password, or configure ~/.pypirc.
EOF
}

skip_confirm="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y)
      skip_confirm="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

version="$(uv run python - <<'PY'
import tomllib
from pathlib import Path
with Path('pyproject.toml').open('rb') as f:
    print(tomllib.load(f)['project']['version'])
PY
)"

cat <<EOF
Publishing drisk $version to PyPI
Repository root: $repo_root
EOF

printf '\n==> Checking lint\n'
uv run ruff check .

printf '\n==> Checking formatting\n'
uv run ruff format . --check

printf '\n==> Running tests\n'
uv run pytest

printf '\n==> Cleaning previous build artifacts\n'
rm -rf dist/

printf '\n==> Building source distribution and wheel\n'
uv run --with build python -m build

printf '\n==> Validating distribution metadata\n'
uv run --with twine twine check dist/*

printf '\n==> Testing local wheel install in isolated environment\n'
mapfile -t wheels < <(find dist -maxdepth 1 -type f -name '*.whl' | sort)
if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "Error: expected exactly one wheel in dist/, found ${#wheels[@]}." >&2
  exit 1
fi

uv run --isolated --with "${wheels[0]}" python - <<'PY'
import drisk as dr

print(f"Imported drisk from {dr.__file__}")
print(dr.Normal.elicit(0, 1))
PY

if [[ "$skip_confirm" != "true" ]]; then
  printf '\nReady to upload drisk %s to PyPI.\n' "$version"
  printf 'This is permanent: PyPI files for a published version cannot be replaced.\n'
  read -r -p "Type 'publish drisk $version' to continue: " confirmation
  if [[ "$confirmation" != "publish drisk $version" ]]; then
    echo "Upload cancelled."
    exit 1
  fi
fi

printf '\n==> Uploading to PyPI\n'
uv run --with twine twine upload dist/*

cat <<'EOF'

Publish complete.

Suggested verification:
  uv run --isolated --with drisk python -c "import drisk as dr; print(dr)"
EOF
