# AI2B — appearance identity benchmark

AI2B measures whether compact appearance embeddings improve candidate
same-player association on a bounded, manually labelled football benchmark.
It does **not** create persistent player identities. Appearance similarity is
evidence, not identity.

## Evolution and boundary

```text
AI1:  detections -> temporary tracklets
AI2A: tracklets -> manual ground truth + temporal/team constraints
AI2B: tracklets -> appearance embeddings -> evaluated candidate associations
AI2C: candidate graph -> persistent identity resolver (future)
```

AI2B contains no jersey-number OCR, pitch homography, ball tracking,
possession, event detection, tactical metrics, coach recommendation, cloud
deployment, infrastructure change, database change, or coach-UI identity.
Developer output may only describe an experimental "Candidate same-player
association".

## Bounded benchmark

The local source is licensed project footage from the Belgium–Japan broadcast.
AI2B uses the 90-second source window from 130 to 220 seconds at 1280×720 and
30 fps. Video pixels and AI1 caches remain ignored. The committed evidence in
`evaluation/ai2b/belgium-japan-wide-90s/` references them by checksum.

The benchmark gate passed with:

- 448 AI1 tracklets;
- 48 manually reviewed annotations;
- 8 confirmed pseudonymous identities, split 4 `TEAM_A` / 4 `TEAM_B`;
- 40 confirmed identity-labelled tracklets, 8 `UNKNOWN`, 0 uncertain;
- 87 positive same-player pairs and 693 negative different-player pairs;
- 5 minutes of bounded crop review recorded by the evaluation run.

Only visually supported fragments were labelled. Long tracklets showing a
person/team switch and crops without sufficient identity evidence were not
forced into an identity.

## Cached pipeline

The new source window required one detector execution. YOLOv8m ran once at
5 sampled fps on CPU, producing 8,508 detections across 450 sampled frames in
129.928 seconds. Every later tracking, crop, team, embedding, threshold, and
evaluation operation reused these caches:

- detections SHA-256:
  `12993fa98d4c2c02005fad9c904a2b9fcba5980ac855e41a40b6b2a857f506de`;
- tracklets SHA-256:
  `8c8cf6975a17217e522019371f2693c3061ac8873f6fc70f007da0d7fc78d040`.

Normal tests use synthetic arrays and mocked encoders. They never invoke YOLO
or download weights.

## Team evidence

The lightweight AI2A central-upper-body HSV representation is retained with
one manually selected anchor per team. It is a conservative guard only when
both team predictions conflict at confidence ≥0.8. `UNKNOWN` never forces a
rejection, and team evidence alone never establishes player identity.

On 48 labelled tracklets:

| Class | Precision | Recall | Support |
|---|---:|---:|---:|
| TEAM_A | 0.9545 | 0.9130 | 23 |
| TEAM_B | 0.8182 | 0.8571 | 21 |
| UNKNOWN rejection | — | 0.0000 | 4 |

The zero unknown rejection confirms that this baseline is not a general
person/team classifier.

## Appearance encoder

`AppearanceEncoder` accepts one BGR player crop and returns a fixed-dimensional
L2-normalized vector. Tracklet evidence is the normalized mean of up to three
valid early/middle/late crop embeddings. Crops smaller than 24×48 are rejected.

The first baseline is
`torchvision.mobilenet_v3_small.features`, ImageNet-1K V1:

| Audit item | Result |
|---|---|
| Purpose | Experimental general visual control; not football-ReID-specific |
| Parameters | 2,542,856 |
| Weight size | 9.829 MB |
| Library license | BSD-3-Clause |
| Weight/data caveat | ImageNet-1K dataset terms apply; weights are not committed |
| Runtime | torchvision 0.28.0 / PyTorch 2.13.0 / Python 3.13 |
| Device | CPU in this run; MPS selected naturally when available |
| Weight SHA-256 | `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f` |
| Preprocessing | deterministic RGB resize-256 / center-crop-224 / ImageNet normalization |
| Dimension | 576 |

The model is loaded lazily from an explicit local file. Its checksum is checked
before loading, and the code contains no model download path.

## Appearance cache and performance

`footballai.appearance-embeddings/v1` stores model/checksum/preprocessing
provenance, relative crop references, crop checksums, normalized crop vectors,
normalized tracklet means, embedding checksums, and rejection reasons. Absolute
paths are forbidden.

- deterministic crops inspected: 1,195;
- usable crop embeddings: 687;
- rejected tiny crops: 508;
- tracklet embeddings: 313;
- local JSON diagnostic cache: 17,808,279 bytes;
- crop extraction: 60.005 seconds;
- end-to-end embedding feature build: 20.802 seconds;
- throughput: 33.025 crop embeddings/second on CPU;
- pair scoring and evaluation: 0.045 seconds.

The large vector cache stays local. The smaller committed evaluation artifact
retains its schema, provenance, counts, measured metrics, and review links.

## Split, threshold, and measured comparison

The deterministic split is identity-disjoint: four identities for development
and four different identities for evaluation. The appearance threshold
0.796524 was selected on 182 appearance-eligible development pairs; all reported
A/B/C results below use the same 171 held-out evaluation pairs (41 positive,
130 negative).

| Evidence | Precision | Recall | F1 | Positive / negative support |
|---|---:|---:|---:|---:|
| A — temporal only | 0.2547 | 1.0000 | 0.4059 | 41 / 130 |
| B — temporal + team | 0.2595 | 1.0000 | 0.4121 | 41 / 130 |
| C — temporal + team + appearance | 0.4038 | 0.5122 | 0.4516 | 41 / 130 |

Appearance increased held-out precision and F1, but lost 20 of 41 positive
pairs. Candidate-pair recall before thresholding was 0.7805 (32/41) because
several confirmed fragments had no usable embedding.

Held-out cosine distributions also overlap substantially:

- positives: support 32, median 0.8316, p10 0.7368, p90 0.9021;
- negatives: support 104, median 0.6908, p10 0.6052, p90 0.8655.

These are exploratory bounded results, not production generalization claims.

## Error analysis

Reviewed false positives were dominated by same-colour kits at similar camera
distance and pose: red jersey 6 was confused with red jersey 14. Reviewed false
negatives included pose/camera-scale differences, partial/background-heavy
crops, single-observation fragments, and unavailable embeddings where every
crop was below 24×48. No cause is assigned beyond visible crop evidence.

The evaluation artifact includes bounded review metadata for five false-positive
and five false-negative pairs, linking both tracklets to relative crop references.
No copyrighted pixels are committed.

## Scientific conclusion

Appearance evidence improves discrimination modestly on the held-out identities,
but 0.4038 precision, 0.5122 recall, 0.4516 F1, 0.7805 candidate recall, strong
positive/negative distribution overlap, and zero unknown-team rejection are not
strong enough to justify a persistent identity resolver.

Before AI2C, collect more identity-disjoint clips across matches/cameras, improve
usable distant-player crop coverage, test a maintained person-ReID-specific
encoder under an acceptable license/runtime footprint, and validate thresholds
across clips rather than within one broadcast segment.
