# AI3 pitch-calibration evidence

| File | Content |
|---|---|
| `synthetic-benchmark.json` | `footballai.pitch-calibration-benchmark/v1`: synthetic pinhole scenes with known ground truth (development seeds 0–39, evaluation seeds 1000–1199 scored once), invalid-input rejection, the camera-cut/pan temporal scenario, runtime, and the list of unmeasured quantities |

**This is synthetic evidence only.** It shows the estimator and the quality
gates behave as designed. It does not measure calibration accuracy on real
football footage: `real_world_evaluation.status` is `pending` with 0 annotated
real frames. No video, frame, crop or detector was used.

Method, thresholds and interpretation:
[`../../docs/v2/AI3_PITCH_CALIBRATION.md`](../../docs/v2/AI3_PITCH_CALIBRATION.md).

## Reproduce

```bash
PYTHONPATH=v2/src python -m footballai_v2.execution.ai3.cli benchmark \
  --output /tmp/ai3-benchmark.json
```

Scene generation uses `random.Random` and the evaluation is deterministic, so
every section except `runtime` reproduces exactly on the same software.
`runtime` (wall-clock and peak traced memory) varies by machine.
