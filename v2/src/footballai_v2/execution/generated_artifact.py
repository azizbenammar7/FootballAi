"""Storage-neutral bytes produced by a pipeline adapter."""

from __future__ import annotations

from dataclasses import dataclass

from footballai_v2.contracts.v1 import ArtifactCategory


@dataclass(frozen=True, slots=True)
class GeneratedArtifact:
    artifact_id: str
    name: str
    category: ArtifactCategory
    relative_path: str
    content: bytes
    media_type: str
    schema_version: str
