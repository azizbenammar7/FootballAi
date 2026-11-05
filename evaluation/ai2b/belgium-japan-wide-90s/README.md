# Belgium–Japan wide 90-second AI2B benchmark

This bounded benchmark uses a 90.0-second, 1280×720 local clip derived from
licensed project footage (source time 130–220 seconds). The video, deterministic
crop pixels, AI1 Parquet caches, model weights, and 17.8 MB embedding cache are
intentionally not committed.

Committed evidence:

- `identity-ground-truth.json`: 48 reviewed tracklets, 8 identities, both teams;
- `review-samples.json`: 1,195 relative crop references and checksums;
- `appearance-evaluation.json`: model/cache provenance, identity-disjoint split,
  team metrics, similarity distributions, A/B/C comparison, performance, and
  bounded false-positive/false-negative review metadata.

The dataset gate passed with 87 positive and 693 negative labelled pairs. The
results are exploratory. Appearance similarity is not identity, and no
persistent player identity artifact exists.

Scientific recommendation: more cross-clip identity evidence and better
distant-player crop coverage are required before AI2C.
