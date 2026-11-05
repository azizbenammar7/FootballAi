#!/usr/bin/env bash
# A private, reproducible Tier-A demo. Only the generated demo_fast fixture is used.
set -euo pipefail

portfolio_root="$(mktemp -d "${TMPDIR:-/tmp}/footballai-portfolio.XXXXXX")"
export FOOTBALLAI_V2_RUN_ROOT="$portfolio_root/runs"
export FOOTBALLAI_V2_QUEUE_ROOT="$portfolio_root/queue"
export FOOTBALLAI_DEMO_SKIP_LEGACY_IMPORT=1
export FOOTBALLAI_ENVIRONMENT=local
export FOOTBALLAI_DATABASE_BACKEND=local_manifest
export FOOTBALLAI_OBJECT_STORAGE_BACKEND=local
export FOOTBALLAI_QUEUE_BACKEND=local
export FOOTBALLAI_COPILOT_ENABLED=1
export FOOTBALLAI_COPILOT_AGENT_ENABLED=1
export FOOTBALLAI_COPILOT_PROVIDER=fake
export FOOTBALLAI_COPILOT_REAL_PROVIDER_ENABLED=0
export FOOTBALLAI_COPILOT_AGENT_REAL_MODEL_ENABLED=0
unset ANTHROPIC_API_KEY

echo "Portfolio demo data (generated locally): $portfolio_root"
exec "$(dirname "$0")/run_demo.sh"
