# AI2A — identity evaluation foundation

AI2A establishes labelled, reproducible evaluation evidence between AI1
tracklets and a future identity resolver. It does **not** solve persistent
player identity.

## Phase boundary

```text
AI1
video -> immutable footballai.detections/v1 -> footballai.tracklets/v1

AI2A (this phase)
tracklets -> manual labels + review crops + team/temporal/appearance evidence
          -> candidate pairs + evaluation metrics

AI2B
tracklets -> appearance embeddings + evaluated candidate associations

AI2C (future)
candidate graph -> persistent identity resolver
```

AI2A contains no pretrained person-ReID framework, pitch calibration,
homography, jersey-number recognition, ball tracking, possession, events, or
tactical analysis. It creates no cloud resource and changes no database.

## Typed identifiers and artifacts

- `detection_id` identifies an ordered observation from an AI1 sampled frame.
- `tracklet_id` identifies one temporary AI1 ByteTrack result and uses the
  canonical form `tracklet_000001`.
- `player_identity_id` is a stable pseudonym such as `TEAM_A_04`. In AI2A it
  is only a manual/ground-truth evaluation label; the type is reserved for a
  future resolver output, but no resolver exists yet.

The versioned evaluation artifacts are:

- `footballai.identity-ground-truth/v1`: source checksums, tracklet extents,
  pseudonymous identity, team/role where known, annotation confidence/status,
  annotator/source, notes, and review state;
- `footballai.tracklet-review-samples/v1`: bounded early/middle/late crop
  selections with source frame/time, bounding box, relative path, and SHA-256;
- `footballai.identity-evaluation/v1`: dataset counts, team evidence, explicit
  cannot-links, candidate graph, measured metrics, and metrics not measured.

No personal names are required or stored. `UNKNOWN` and `uncertain` are first-
class values. One tracklet can have only one annotation. Reusing an identity
across fragments is allowed only when it does not violate an explicit
cannot-link constraint.

## Manual review workflow

The developer CLI never auto-assigns a player identity. `prepare` verifies that
the video, detections, and tracklets checksums agree, then extracts at most
three deterministic JPEG crops per tracklet and creates an HTML/contact-sheet
review bundle:

```bash
PYTHONPATH=v2/src .venv-test/bin/python -m \
  footballai_v2.execution.ai2a.review_cli prepare \
  --video /path/to/bounded-source.mp4 \
  --detections /path/to/detections.parquet \
  --tracklets /path/to/tracklets.parquet \
  --source-id bounded-evaluation-clip \
  --output-dir /tmp/footballai-ai2a-review
```

Review and explicitly assign one tracklet at a time:

```bash
PYTHONPATH=v2/src .venv-test/bin/python -m \
  footballai_v2.execution.ai2a.review_cli list \
  --tracklets /path/to/tracklets.parquet \
  --review-samples /tmp/footballai-ai2a-review/review-samples.json \
  --ground-truth /tmp/footballai-ai2a-review/identity-ground-truth.json

PYTHONPATH=v2/src .venv-test/bin/python -m \
  footballai_v2.execution.ai2a.review_cli assign \
  --tracklets /path/to/tracklets.parquet \
  --ground-truth /tmp/footballai-ai2a-review/identity-ground-truth.json \
  --tracklet tracklet_000014 --identity TEAM_A_04 --team TEAM_A \
  --role PLAYER --status confirmed --confidence 0.9 \
  --annotator evaluation-reviewer
```

Use `--identity UNKNOWN --team UNKNOWN` when evidence is insufficient. The
`render` command reloads saved annotations into the local HTML sheet.

## Evidence and constraints

The team baseline uses the median HSV value from the central upper-body region
of representative crops and nearest manually selected team-colour anchors.
Low-saturation, distant, or ambiguous evidence is rejected as `UNKNOWN`. This
is evaluation evidence, not a verified identity or a trained team classifier.
A clip containing one team may provide only a `TEAM_A` anchor; it must not
invent `TEAM_B` labels for staff or officials.

A temporal cannot-link is emitted when two tracklets share a sampled frame and
their pixel bounding boxes are spatially distinct. The candidate graph omits
cannot-linked pairs and incompatible known-team pairs. Edges may contain only
implemented channels: temporal compatibility, team compatibility, and the
bounded jersey-colour similarity. There is no pitch-trajectory or jersey-
number evidence.

## Metrics

AI2A measures pairwise association precision/recall/F1, candidate-pair recall,
team classification accuracy, and rejection rate. Identity observation
coverage is implemented for future resolver outputs but is reported as not
measured in the AI2A benchmark because there is no resolver output. HOTA,
IDF1, and ID-switch counts are explicitly not measured because this annotation
format does not support their tracking semantics.

## Bounded benchmark

The committed benchmark metadata under
`evaluation/ai2a/ai1-bounded-18s-validation/` references the AI1 validation
clip by checksum. The clip is 18.551867 seconds: adequate for schema, tooling,
constraint, and metric validation, but too short for meaningful long-term
identity claims.

The run reused 1,357 cached detections and 39 cached tracklets. Manual review
covered all 39 tracklets using 108 deterministic crops in about 15 minutes:
33 pseudonymous identities, five unknown identities, and six uncertain
annotations. The clip shows one playing team plus staff/media and has no
defensible `TEAM_B` examples.

Measured baseline results:

- 330 temporal/spatial cannot-links;
- 39 candidate-graph nodes and 411 candidate edges;
- candidate-pair recall 1.0 on one confirmed positive association;
- pairwise precision 0.007299, recall 1.0, F1 0.014493;
- team accuracy 0.435897 and rejection rate 0.025641 across 39 annotations.

The low pairwise precision and poor unknown rejection are evidence that colour
similarity is not a persistent identity solution. The single positive pair and
short duration make all association figures preliminary. No detector was run,
and the normal tests import neither Ultralytics nor model weights.
