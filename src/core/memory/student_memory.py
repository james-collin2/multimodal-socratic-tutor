"""
src/core/memory/student_memory.py

StudentMemory — persists per-student knowledge across sessions.
Single responsibility: read/write student memory to SQLite.

Schema (student_memory table):
    student_id   TEXT PRIMARY KEY
    weak_topics  TEXT  (JSON list)
    strong_topics TEXT (JSON list)
    mastery_scores TEXT (JSON dict topic→score)
    total_sessions INTEGER
    total_turns    INTEGER
    last_seen      TEXT (ISO datetime)
    notes          TEXT (free-form observations)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.utils.logger import logger


@dataclass
class MemoryRecord:
    """In-memory representation of one student's memory."""

    student_id: str
    weak_topics: list[str] = field(default_factory=list)
    strong_topics: list[str] = field(default_factory=list)
    mastery_scores: dict[str, float] = field(default_factory=dict)
    total_sessions: int = 0
    total_turns: int = 0
    last_seen: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "weak_topics": json.dumps(self.weak_topics),
            "strong_topics": json.dumps(self.strong_topics),
            "mastery_scores": json.dumps(self.mastery_scores),
            "total_sessions": self.total_sessions,
            "total_turns": self.total_turns,
            "last_seen": self.last_seen,
            "notes": self.notes,
        }

    @classmethod
    def from_row(cls, row: dict) -> MemoryRecord:
        return cls(
            student_id=row["student_id"],
            weak_topics=json.loads(row.get("weak_topics", "[]")),
            strong_topics=json.loads(row.get("strong_topics", "[]")),
            mastery_scores=json.loads(row.get("mastery_scores", "{}")),
            total_sessions=int(row.get("total_sessions", 0)),
            total_turns=int(row.get("total_turns", 0)),
            last_seen=row.get("last_seen", ""),
            notes=row.get("notes", ""),
        )


class StudentMemory:
    """
    Async SQLite-backed student memory store.

    Usage:
        memory = StudentMemory()
        record = await memory.load("student_001")
        record.weak_topics.append("cranial_nerves")
        await memory.save(record)
    """

    def __init__(self) -> None:
        from src.config.settings import get_settings

        settings = get_settings()
        db_path = settings.database_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_url = str(db_path)
        self._initialised = False

    def _conn(self) -> object:
        """Return aiosqlite connection context manager — fresh each call."""
        import aiosqlite

        return aiosqlite.connect(self._db_url)

    async def init(self) -> None:
        if self._initialised:
            return
        async with self._conn() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS student_memory (
                    student_id     TEXT PRIMARY KEY,
                    weak_topics    TEXT DEFAULT '[]',
                    strong_topics  TEXT DEFAULT '[]',
                    mastery_scores TEXT DEFAULT '{}',
                    total_sessions INTEGER DEFAULT 0,
                    total_turns    INTEGER DEFAULT 0,
                    last_seen      TEXT,
                    notes          TEXT DEFAULT ''
                )
            """)
            await conn.commit()
        self._initialised = True

    async def load(self, student_id: str) -> MemoryRecord:
        """Load memory for a student. Returns empty record if not found."""
        await self.init()
        async with self._conn() as conn:
            conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
            async with conn.execute(
                "SELECT * FROM student_memory WHERE student_id = ?",
                (student_id,),
            ) as cur:
                row = await cur.fetchone()
        if row:
            return MemoryRecord.from_row(row)
        return MemoryRecord(student_id=student_id)

    async def save(self, record: MemoryRecord) -> None:
        """Upsert memory record."""
        await self.init()
        record.last_seen = datetime.now(tz=timezone.utc).isoformat()
        d = record.to_dict()
        async with self._conn() as conn:
            await conn.execute(
                """
                INSERT INTO student_memory
                    (student_id, weak_topics, strong_topics, mastery_scores,
                     total_sessions, total_turns, last_seen, notes)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(student_id) DO UPDATE SET
                    weak_topics    = excluded.weak_topics,
                    strong_topics  = excluded.strong_topics,
                    mastery_scores = excluded.mastery_scores,
                    total_sessions = excluded.total_sessions,
                    total_turns    = excluded.total_turns,
                    last_seen      = excluded.last_seen,
                    notes          = excluded.notes
            """,
                (
                    d["student_id"],
                    d["weak_topics"],
                    d["strong_topics"],
                    d["mastery_scores"],
                    d["total_sessions"],
                    d["total_turns"],
                    d["last_seen"],
                    d["notes"],
                ),
            )
            await conn.commit()
        logger.debug("Memory saved for student {id}", id=record.student_id)

    async def update_topic_score(
        self,
        student_id: str,
        topic: str,
        score: float,
    ) -> MemoryRecord:
        """Update mastery score for one topic and refresh weak/strong lists."""
        record = await self.load(student_id)
        record.mastery_scores[topic] = round(score, 1)

        # Rebuild weak/strong lists
        record.weak_topics = [t for t, s in record.mastery_scores.items() if s < 60]
        record.strong_topics = [t for t, s in record.mastery_scores.items() if s >= 80]

        await self.save(record)
        return record

    async def delete(self, student_id: str) -> None:
        """Erase all long-term memory for a student (clear learning history)."""
        await self.init()
        async with self._conn() as conn:
            await conn.execute(
                "DELETE FROM student_memory WHERE student_id = ?",
                (student_id,),
            )
            await conn.commit()
        logger.info("Cleared long-term memory for {sid}", sid=student_id)
