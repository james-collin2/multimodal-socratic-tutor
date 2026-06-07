"""
src/schemas/base.py

Shared base Pydantic v2 models used across all modules.
Import these instead of redefining common structures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class BaseSchema(BaseModel):
    """Base for all schemas — immutable, validated on assignment."""

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class TimestampedSchema(BaseSchema):
    """Base schema with auto-managed timestamps."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class IdentifiedSchema(TimestampedSchema):
    """Base schema with UUID primary key and timestamps."""

    id: UUID = Field(default_factory=uuid4)
