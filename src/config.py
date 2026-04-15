"""Project configuration helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ProjectConfig:
    target_column: str = "is_fraud"
    id_columns: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)
    user_id_column: str | None = "user_id"
    timestamp_column: str | None = "event_timestamp"
    device_column: str | None = "device_type"
    location_column: str | None = "location"
    amount_column: str | None = "transaction_amount"
    failed_login_column: str | None = "failed_login_attempts"
    top_k_features: int = 20
    test_size: float = 0.15
    validation_size: float = 0.15
    random_state: int = 42

    @classmethod
    def from_json(cls, path: str | Path) -> "ProjectConfig":
        content = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**content)
