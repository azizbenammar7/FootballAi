#!/usr/bin/env bash
# Fails when pipeline, dashboard, scripts, or data/processed differ from the V1
# baseline, except for the two exact exceptions below.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

baseline_ref="technical-test-v1.0"
protected_paths=(pipeline dashboard scripts data/processed)

# Bytecode committed by mistake at the tag and later removed. Only deletion is allowed.
is_removed_bytecode() {
  case "$1" in
    dashboard/__pycache__/app.cpython-313.pyc | \
    dashboard/__pycache__/app.cpython-314.pyc | \
    pipeline/__pycache__/01_track.cpython-314.pyc | \
    pipeline/__pycache__/02_stats.cpython-313.pyc | \
    pipeline/__pycache__/02_stats.cpython-314.pyc | \
    pipeline/__pycache__/03_fatigue.cpython-313.pyc | \
    pipeline/__pycache__/03_fatigue.cpython-314.pyc | \
    scripts/__pycache__/gen_dummy_data.cpython-314.pyc) return 0 ;;
    *) return 1 ;;
  esac
}

# Modern CI helpers. Only additions are allowed, so a tagged file can never match.
is_new_ci_script() {
  [[ "$1" == scripts/ci/* ]]
}

if ! git rev-parse --verify --quiet "${baseline_ref}^{commit}" >/dev/null; then
  echo "ERROR: V1 baseline ${baseline_ref} is not available." >&2
  exit 2
fi

changes="$(mktemp)"
untracked="$(mktemp)"
trap 'rm -f "$changes" "$untracked"' EXIT
git diff -z --no-renames --name-status "$baseline_ref" -- "${protected_paths[@]}" >"$changes"
git ls-files -z --others --exclude-standard -- "${protected_paths[@]}" >"$untracked"

# These index flags hide local edits from git diff.
hidden_edits="$(git ls-files -v -- "${protected_paths[@]}" | grep -E '^([a-z]|S) ' || true)"

violations=0
report_violation() {
  echo "ERROR: V1 baseline changed: $1 $2" >&2
  violations=$((violations + 1))
}

while IFS= read -r -d '' status && IFS= read -r -d '' path; do
  if [[ "$status" == D ]] && is_removed_bytecode "$path"; then continue; fi
  if [[ "$status" == A ]] && is_new_ci_script "$path"; then continue; fi
  report_violation "$status" "$path"
done <"$changes"

while IFS= read -r -d '' path; do
  if is_new_ci_script "$path"; then continue; fi
  report_violation "?" "$path"
done <"$untracked"

if [[ -n "$hidden_edits" ]]; then
  report_violation "hidden" "$hidden_edits"
fi

[[ "$violations" -eq 0 ]]
