# ADR-002: Detector-neutral V2 architecture

- Status: Accepted for future implementation
- Date: 2026-07-26

## Context

The preserved V1 technical-test baseline uses Ultralytics YOLOv8. V2 must not
couple the platform contract, storage model, API, or dashboard to one model
vendor or detection library. Licensing, accuracy, throughput, hardware,
deployment, and reproducibility requirements can differ by operating context.

## Decision

- Preserve the V1 YOLOv8 implementation unchanged.
- Define a detector interface in a later milestone.
- Evaluate RT-DETR and RT-DETRv2 as the preferred permissively licensed V2
  candidate family.
- Permit optional detector adapters for reproducible comparison.
- Select the operational default only after a documented benchmark using
  representative data, agreed metrics, and repeatable configuration.
- Record complete model implementation and weight provenance for every run.

Required future provenance fields are:

```text
component
model_name
model_version
implementation_repository
implementation_commit
weights_source
weights_checksum
code_license
weights_license
training_dataset
dataset_license
configuration_checksum
```

## Consequences

Platform code will depend on an interface and explicit provenance rather than
on model-specific objects. Benchmark adapters may coexist, and changing a
detector will remain visible in attempt provenance. Model code and weight
licensing must be reviewed separately before public or commercial deployment.

## Scope

Milestone 2 adds no detector interface or implementation, RT-DETR code,
PyTorch, model weights, or ML dependency. This decision is not a legal
guarantee. Final code, weights, dataset, and deployment licensing decisions
require review before an operational default is approved.
