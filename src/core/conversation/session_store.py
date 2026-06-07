"""
src/core/conversation/session_store.py

Async SQLite session store using aiosqlite + SQLAlchemy core.
Single responsibility: persist and retrieve SessionState objects.

All DB operations are async — never block the event loop.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from src.config.settings import get_settings
from src.core.conversation.state import ConversationPhase, HintLevel
from src.utils.exceptions import DatabaseError, SessionNotFoundError
from src.utils.logger import logger


class SessionStore:
    """
    Async SQLite-backed store for conversation sessions.

    Schema (sessions table):
        session_id   TEXT PRIMARY KEY
        student_id   TEXT NOT NULL
        phase        TEXT NOT NULL
        turn_count   INTEGER DEFAULT 0
        hint_level   INTEGER DEFAULT 0
        current_topic TEXT
        history_json TEXT          -- JSON array of turn records
        created_at   TEXT
        updated_at   TEXT
        is_active    INTEGER DEFAULT 1
    """

    def __init__(self) -> None:
        settings = get_settings()
        db_path = settings.database_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_url = str(db_path)
        self._initialised = False

    async def _get_conn(self) -> object:
        """
        Return an *unopened* aiosqlite connection.

        Note: do NOT await ``aiosqlite.connect`` here. Callers use it as
        ``async with await self._get_conn() as conn``; the ``async with``
        opens the connection. Awaiting here too would start the underlying
        thread twice ("threads can only be started once").
        """
        try:
            import aiosqlite

            return aiosqlite.connect(self._db_url)
        except ImportError as e:
            raise DatabaseError("aiosqlite not installed", detail=str(e)) from e

    async def init(self) -> None:
        """Create tables if they don't exist. Call once at startup."""
        if self._initialised:
            return
        async with await self._get_conn() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id    TEXT PRIMARY KEY,
                    student_id    TEXT NOT NULL,
                    phase         TEXT NOT NULL DEFAULT 'rapport',
                    turn_count    INTEGER DEFAULT 0,
                    hint_level    INTEGER DEFAULT 0,
                    current_topic TEXT,
                    history_json  TEXT DEFAULT '[]',
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL,
                    is_active     INTEGER DEFAULT 1
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS student_profiles (
                    student_id      TEXT PRIMARY KEY,
                    name            TEXT,
                    program_semester INTEGER,
                    weak_topics     TEXT DEFAULT '[]',
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                )
            """)
            await conn.commit()
        self._initialised = True
        logger.info("SessionStore initialised at {db}", db=self._db_url)

    async def create_session(self, student_id: str) -> str:
        """Create a new session and return session_id."""
        await self.init()
        session_id = str(uuid4())
        now = datetime.now(tz=timezone.utc).isoformat()
        async with await self._get_conn() as conn:
            await conn.execute(
                """INSERT INTO sessions
                   (session_id, student_id, phase, turn_count, hint_level,
                    history_json, created_at, updated_at, is_active)
                   VALUES (?, ?, 'rapport', 0, 0, '[]', ?, ?, 1)""",
                (session_id, student_id, now, now),
            )
            await conn.commit()
        logger.info(
            "Created session {sid} for student {uid}",
            sid=session_id[:8],
            uid=student_id,
        )
        return session_id

    async def get_session(self, session_id: str) -> dict:
        """Retrieve session data by ID."""
        await self.init()
        async with await self._get_conn() as conn:
            conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
            async with conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            raise SessionNotFoundError(
                f"Session not found: {session_id[:8]}",
                detail="Session may have expired or never existed",
            )
        row["history"] = json.loads(row.pop("history_json", "[]"))
        return row

    async def update_session(
        self,
        session_id: str,
        phase: ConversationPhase | None = None,
        turn_count: int | None = None,
        hint_level: HintLevel | None = None,
        current_topic: str | None = None,
        history: list | None = None,
        is_active: bool | None = None,
    ) -> None:
        """Partial update — only provided fields are changed."""
        await self.init()
        updates: list[tuple] = []
        values: list = []

        if phase is not None:
            updates.append("phase = ?")
            values.append(phase.value)
        if turn_count is not None:
            updates.append("turn_count = ?")
            values.append(turn_count)
        if hint_level is not None:
            updates.append("hint_level = ?")
            values.append(int(hint_level))
        if current_topic is not None:
            updates.append("current_topic = ?")
            values.append(current_topic)
        if history is not None:
            updates.append("history_json = ?")
            values.append(json.dumps(history))
        if is_active is not None:
            updates.append("is_active = ?")
            values.append(1 if is_active else 0)

        if not updates:
            return

        updates.append("updated_at = ?")
        values.append(datetime.now(tz=timezone.utc).isoformat())
        values.append(session_id)

        # column names in `updates` are code-controlled; all values are parameterized
        sql = f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?"  # noqa: S608
        async with await self._get_conn() as conn:
            await conn.execute(sql, values)
            await conn.commit()

    async def save_student_profile(
        self,
        student_id: str,
        name: str | None = None,
        semester: int | None = None,
        weak_topics: list[str] | None = None,
    ) -> None:
        """Upsert student profile."""
        await self.init()
        now = datetime.now(tz=timezone.utc).isoformat()
        async with await self._get_conn() as conn:
            await conn.execute(
                """INSERT INTO student_profiles
                   (student_id, name, program_semester, weak_topics, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(student_id) DO UPDATE SET
                     name = COALESCE(excluded.name, name),
                     program_semester = COALESCE(excluded.program_semester, program_semester),
                     weak_topics = COALESCE(excluded.weak_topics, weak_topics),
                     updated_at = excluded.updated_at""",
                (
                    student_id,
                    name,
                    semester,
                    json.dumps(weak_topics or []),
                    now,
                    now,
                ),
            )
            await conn.commit()

    async def get_student_profile(self, student_id: str) -> dict | None:
        """Get student profile or None if not found."""
        await self.init()
        async with await self._get_conn() as conn:
            conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
            async with conn.execute(
                "SELECT * FROM student_profiles WHERE student_id = ?",
                (student_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row:
            row["weak_topics"] = json.loads(row.get("weak_topics", "[]"))
        return row

    async def list_student_sessions(self, student_id: str) -> list[dict]:
        """Return all sessions for a student, newest first."""
        await self.init()
        async with await self._get_conn() as conn:
            conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
            async with conn.execute(
                """SELECT session_id, phase, turn_count, current_topic,
                          created_at, updated_at, is_active
                   FROM sessions WHERE student_id = ?
                   ORDER BY created_at DESC""",
                (student_id,),
            ) as cursor:
                return await cursor.fetchall()
