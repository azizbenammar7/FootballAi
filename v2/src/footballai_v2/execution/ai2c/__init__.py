"""AI2C: conservative, reviewable player-identity association over AI1 tracklets.

``detection_id``, ``tracklet_id`` and ``player_identity_id`` stay distinct. A
tracklet is never promoted to a player identity on its own: a tracklet with no
accepted association is ``unresolved`` and carries no player identity.

The resolver defaults to ``review_only``. Automatic association
(``temporal_only``, ``temporal_kinematic``) is experimental, must be selected
explicitly, and only ever yields ``provisional`` identities.
"""

from footballai_v2.execution.ai2c.artifact import (
    PLAYER_IDENTITIES_SCHEMA,
    AssignmentStatus,
    PlayerIdentitiesArtifact,
    player_identities_from_json_bytes,
    player_identities_to_json_bytes,
)
from footballai_v2.execution.ai2c.resolver import (
    SELECTED_TEMPORAL_KINEMATIC,
    ResolverConfig,
    ResolverMode,
    resolve_identities,
)

__all__ = [
    "PLAYER_IDENTITIES_SCHEMA",
    "SELECTED_TEMPORAL_KINEMATIC",
    "AssignmentStatus",
    "PlayerIdentitiesArtifact",
    "ResolverConfig",
    "ResolverMode",
    "player_identities_from_json_bytes",
    "player_identities_to_json_bytes",
    "resolve_identities",
]
