"""Pipeline profile metadata and adapters."""

from footballai_v2.execution.adapters.demo_pipeline import DemoPipeline
from footballai_v2.execution.adapters.v1_compat_pipeline import V1CompatPipeline, profile_catalog

__all__ = ["DemoPipeline", "V1CompatPipeline", "profile_catalog"]
