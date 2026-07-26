# Known V1 limitations

This document records verified limitations of the preserved technical-test baseline. “Implementation behavior” describes what the code currently does. “Scientific limitation” describes why an output cannot be treated as validated measurement. “Missing evidence” identifies a claim the repository cannot substantiate. “V2 requirement” states the future control needed; none is implemented in this milestone.

| # | Classification | Verified V1 behavior or limitation | Future V2 requirement |
|---|---|---|---|
| 1 | Implementation behavior | ByteTrack identifiers are temporary track IDs. They are not persistent real-player identities. | Introduce explicit observation, track, and verified-player identity concepts. |
| 2 | Implementation behavior / scientific limitation | Halftime stitching pairs highly active first-half and second-half tracks by rank. Activity rank does not prove that two tracks belong to the same player. | Use evidence-based identity association with confidence and reviewable provenance. |
| 3 | Implementation behavior | Movement uses bounding-box centers, not reliably detected player foot/contact points. | Define and validate a stable ground-contact observation method. |
| 4 | Scientific limitation | A single global pixel-to-metre scale is used without pitch homography, so image displacement is not calibrated world displacement. | Add frame-aware pitch calibration and quantified error bounds. |
| 5 | Scientific limitation | Camera pan and zoom can be interpreted as player movement. | Compensate camera motion before player kinematics. |
| 6 | Implementation behavior | Active time is the span from the first detection to the last detection, including unobserved gaps between them. | Track observed intervals and distinguish elapsed span from observed active time. |
| 7 | Known implementation defect | The component named “HSR load” is calculated from total distance relative to a population reference, not from high-speed-running distance. | Version the corrected metric and preserve the V1 definition for comparability. |
| 8 | Implementation behavior | The match is split into halves at the video midpoint rather than at detected or supplied football half boundaries. | Store explicit match-period boundaries with their source. |
| 9 | Implementation behavior | A 15-minute block receives the half of its first record; a block crossing the midpoint can therefore contain observations from both halves under one half label. | Segment periods before aggregating blocks or split mixed blocks. |
| 10 | Implementation behavior | Coverage gating uses presence in populated blocks and a derived block fraction; it does not measure actual observed-frame coverage for the risk gate. | Define coverage from expected versus observed samples and gaps. |
| 11 | Missing evidence / data-quality limitation | The committed artifacts contain many fragmented tracks: 177 player-summary records, of which 154 current risk records are `INSUFFICIENT`. | Add track-quality diagnostics and identity-resolution evidence before player-level analysis. |
| 12 | Scientific limitation | Risk outputs are unvalidated heuristics. They must not be interpreted as medical injury predictions or clinical advice. | Rename and validate the decision-support output, document intended use, and require domain review. |
| 13 | Missing evidence | The repository has no reproducible analysis-run manifest binding input identity, code revision, parameters, model version, and outputs for a full-match execution. | Add immutable analysis-run provenance and versioned output references. |
| 14 | Implementation behavior | Checkpoint CSV files preserve partial detections during a process, but the tracking stage deletes an existing checkpoint at startup and cannot resume from it. | Implement explicit resumable states with compatible input/config verification. |
| 15 | Implementation behavior | Synthetic and real processing write the same output filenames under `data/processed`, allowing one mode to replace the other. | Isolate outputs by analysis-run ID and record data origin. |

These limitations remain intentionally unchanged in V2 Milestone 1. The characterization suite makes selected behavior reproducible so later corrections can be introduced as explicit, reviewed schema or algorithm versions.
