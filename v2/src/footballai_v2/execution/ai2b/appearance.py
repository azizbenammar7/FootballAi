"""Appearance encoder interface and compact experimental visual baseline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from footballai_v2.execution.ai2b.artifact import AppearanceModelProvenance, normalize_embedding


PREPROCESSING_VERSION = "rgb-center-crop-224/v1"


class AppearanceEncoder(Protocol):
    """Encode one representative BGR player crop into a normalized vector."""

    @property
    def provenance(self) -> AppearanceModelProvenance: ...

    def encode(self, crop: np.ndarray) -> tuple[float, ...]: ...


def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Apply deterministic ImageNet RGB resize, center crop, and normalization."""
    import cv2

    if not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError("appearance crop must be an HxWx3 array")
    height, width = crop.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("appearance crop must have positive dimensions")
    scale = 256.0 / min(height, width)
    resized_width = max(224, int(round(width * scale)))
    resized_height = max(224, int(round(height * scale)))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (resized_width - 224) // 2
    top = (resized_height - 224) // 2
    centered = resized[top : top + 224, left : left + 224].astype(np.float32) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    normalized = (centered - mean) / std
    return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)


class MobileNetV3SmallEncoder:
    """Lazy, offline-only MobileNetV3-Small ImageNet feature extractor.

    This is a general visual embedding control, not a football-specific ReID model.
    The caller must explicitly supply a local, checksummed weights file.
    """

    def __init__(
        self,
        weights_path: Path,
        *,
        model_sha256: str,
        model_version: str = "torchvision-0.28.0/IMAGENET1K_V1",
        device: str = "auto",
    ) -> None:
        self._weights_path = Path(weights_path)
        self._model_sha256 = model_sha256
        self._model_version = model_version
        self._requested_device = device
        self._resolved_device: str | None = None
        self._model = None

    @property
    def provenance(self) -> AppearanceModelProvenance:
        return AppearanceModelProvenance(
            model_name="torchvision.mobilenet_v3_small.features",
            model_version=self._model_version,
            model_sha256=self._model_sha256,
            preprocessing_version=PREPROCESSING_VERSION,
            embedding_dimension=576,
            purpose="experimental general visual embedding control; not football-ReID-specific",
            license="BSD-3-Clause library; ImageNet-1K dataset terms apply to pretrained weights",
            device=self._device(),
        )

    def _device(self) -> str:
        if self._resolved_device is None:
            import torch

            if self._requested_device == "auto":
                self._resolved_device = "mps" if torch.backends.mps.is_available() else "cpu"
            elif self._requested_device in {"cpu", "mps"}:
                if self._requested_device == "mps" and not torch.backends.mps.is_available():
                    raise RuntimeError("MPS was requested but is not available")
                self._resolved_device = self._requested_device
            else:
                raise ValueError("appearance device must be auto, cpu, or mps")
        return self._resolved_device

    def _load_model(self):
        if self._model is not None:
            return self._model
        import torch
        from torchvision.models import mobilenet_v3_small

        path = self._weights_path.resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != self._model_sha256:
            raise ValueError("appearance model checksum mismatch")
        model = mobilenet_v3_small(weights=None)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval().to(self._device())
        self._model = model
        return model

    def encode(self, crop: np.ndarray) -> tuple[float, ...]:
        import torch

        model = self._load_model()
        tensor = torch.from_numpy(preprocess_crop(crop)).unsqueeze(0).to(self._device())
        with torch.inference_mode():
            features = model.features(tensor)
            pooled = model.avgpool(features)
            vector = torch.flatten(pooled, 1)[0].to("cpu").tolist()
        return normalize_embedding(vector)
