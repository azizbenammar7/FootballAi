# AI2C identity-association evidence

Generated offline by `footballai_v2.execution.ai2c.cli` from cached, checksummed
inputs. No detector, tracker, video, or model was run. See
[`../../docs/v2/AI2C_PLAYER_IDENTITY.md`](../../docs/v2/AI2C_PLAYER_IDENTITY.md)
for the method, results, and limitations.

| File | Content |
|---|---|
| `identity-resolution-evaluation.json` | `footballai.identity-resolution-evaluation/v1`: protocol, development sweep, baselines, held-out results, AI2A second-clip results, legacy full-match scale diagnostic |
| `ai1-bounded-18s-validation/player-identities.json` | `footballai.player-identities/v1` for the AI2A clip (default configuration plus AI2A team evidence and cannot-links). A test reproduces it byte for byte |
| `belgium-japan-wide-90s/player-identities.json` | `footballai.player-identities/v1` for the AI2B clip (default configuration) |

All automatic identities in these files are `provisional`. They are evaluation
output, not verified players.

## Inputs

On `main`:

- `evaluation/ai2a/ai1-bounded-18s-validation/{review-samples,identity-ground-truth,identity-evaluation}.json`
- `data/processed/raw_tracks.parquet` (legacy V1, SHA-256
  `8afd3b5ac305bc276db625b3978a9962405e3fa0ac1766f5ceb7cf605a27879f`)

From unmerged PR #37 (`feat/ai2b-appearance-reid-benchmark` at
`d108538ec94d0ab0326ba64e2206b851bbad304e`), read only, not copied into this
branch:

| File | SHA-256 |
|---|---|
| `evaluation/ai2b/belgium-japan-wide-90s/review-samples.json` | `ce8ca6dd0d85b91b60431396035b4a8f56d73862bc2d9448b4c9f5bbefff441c` |
| `evaluation/ai2b/belgium-japan-wide-90s/identity-ground-truth.json` | `470a3c0c3be448f7ec02845a39a0b56ed5e1c0633cb739693778fc0e5dbdf786` |
| `evaluation/ai2b/belgium-japan-wide-90s/appearance-evaluation.json` | `89e83f33cb900f902beaae730a8ca42c31741c4a88bfbfec5016871273d11f90` (split definition only) |

## Reproduce

```bash
git fetch origin feat/ai2b-appearance-reid-benchmark
mkdir -p /tmp/ai2b
for f in review-samples.json identity-ground-truth.json; do
  git show d108538ec94d0ab0326ba64e2206b851bbad304e:evaluation/ai2b/belgium-japan-wide-90s/$f > /tmp/ai2b/$f
done
sha256sum /tmp/ai2b/*.json
```

```bash
A=evaluation/ai2a/ai1-bounded-18s-validation
PYTHONPATH=v2/src .venv-test/bin/python -m footballai_v2.execution.ai2c.cli benchmark \
  --review-samples /tmp/ai2b/review-samples.json \
  --ground-truth /tmp/ai2b/identity-ground-truth.json \
  --development-identities TEAM_A_01,TEAM_A_03,TEAM_B_01,TEAM_B_03 \
  --evaluation-identities TEAM_A_02,TEAM_A_04,TEAM_B_02,TEAM_B_04 \
  --extra-clip $A/review-samples.json,$A/identity-ground-truth.json,$A/identity-evaluation.json \
  --legacy-raw-tracks data/processed/raw_tracks.parquet \
  --output /tmp/ai2c-benchmark.json
```

Every metric except `runtime` is deterministic. Runtime and peak traced memory
depend on the host.
