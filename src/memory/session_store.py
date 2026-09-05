from __future__ import annotations

import hashlib
from uuid import uuid4

import redis
from redis.exceptions import RedisError

from src.config.settings import settings
from src.models.contracts import (
    MemorySnapshot,
    SessionDocumentReference,
)


class RedisSessionStore:
    """
    Redis-backed short-term/session memory.

    Purpose:
    - remember the recent conversation
    - remember all document references in the session
    - remember the most recently active document
    - remember active page
    - remember selected figure
    - remember current language
    - remember estimated grade

    This is temporary SESSION memory.

    Durable learning/profile memory will be handled
    separately by PostgreSQL in Phase 7 #5.
    """

    DEFAULT_TTL_SECONDS = 24 * 60 * 60
    MAX_RECENT_MESSAGES = 10
    MAX_SESSION_DOCUMENTS = 30

    def __init__(
        self,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(
                "ttl_seconds must be positive."
            )

        redis_url = (
            settings.redis_url
            or "redis://localhost:6379/0"
        )

        self.ttl_seconds = int(
            ttl_seconds
        )

        self.pool = (
            redis.ConnectionPool.from_url(
                redis_url,
                decode_responses=True,
                max_connections=50,
                socket_connect_timeout=2,
                socket_timeout=3,
                health_check_interval=30,
            )
        )

        self.client = redis.Redis(
            connection_pool=self.pool
        )

    # =====================================================
    # LOAD
    # =====================================================

    def load(
        self,
        *,
        user_id: str,
        session_id: str,
    ) -> MemorySnapshot | None:
        """
        Load one user's specific session memory.

        Redis failure is fail-open:
        the tutor can continue with empty session memory.
        """

        normalized_user_id = (
            self._normalize_identifier(
                user_id
            )
        )

        normalized_session_id = (
            self._normalize_identifier(
                session_id
            )
        )

        if (
            normalized_user_id is None
            or normalized_session_id is None
        ):
            return None

        key = self._session_key(
            user_id=normalized_user_id,
            session_id=(
                normalized_session_id
            ),
        )

        try:
            raw_memory = self.client.get(
                key
            )

        except RedisError:
            return None

        if raw_memory is None:
            return None

        try:
            memory = (
                MemorySnapshot
                .model_validate_json(
                    raw_memory
                )
            )

        except Exception:
            # Corrupted temporary memory should
            # never break the chat workflow.
            try:
                self.client.delete(
                    key
                )
            except RedisError:
                pass

            return None

        return self._trim_memory(
            memory
        )

    # =====================================================
    # SAVE
    # =====================================================

    def save(
        self,
        *,
        user_id: str,
        session_id: str,
        memory: MemorySnapshot,
    ) -> None:
        """
        Save temporary session memory.

        Every successful save refreshes the TTL.
        """

        normalized_user_id = (
            self._normalize_identifier(
                user_id
            )
        )

        normalized_session_id = (
            self._normalize_identifier(
                session_id
            )
        )

        if (
            normalized_user_id is None
            or normalized_session_id is None
        ):
            return

        if not isinstance(
            memory,
            MemorySnapshot,
        ):
            return

        # Load the already stored session first so a partial
        # incoming snapshot cannot accidentally forget older
        # document references.
        #
        # Example:
        # stored   -> [A, B]
        # incoming -> [B, C]
        # saved    -> [A, B, C]
        existing_memory = self.load(
            user_id=normalized_user_id,
            session_id=normalized_session_id,
        )

        memory_to_store = (
            self._merge_session_documents(
                existing_memory=existing_memory,
                incoming_memory=memory,
            )
        )

        memory_to_store = (
            self._trim_memory(
                memory_to_store
            )
        )

        key = self._session_key(
            user_id=normalized_user_id,
            session_id=(
                normalized_session_id
            ),
        )

        try:
            self.client.set(
                key,
                memory_to_store
                .model_dump_json(),
                ex=self.ttl_seconds,
            )

        except RedisError:
            # Session memory should not make
            # the whole tutor unavailable
            # if Redis temporarily fails.
            return

        # Keep a short-lived reversible index beside the hashed
        # session-memory key. This does NOT change the existing
        # memory payload or isolation scheme; it only lets the
        # frontend discover and resume still-alive sessions.
        #
        # The index uses the same TTL as the session snapshot, so
        # it disappears automatically when the session expires.
        index_key = self._session_index_key(
            user_id=normalized_user_id,
            session_id=normalized_session_id,
        )

        try:
            self.client.set(
                index_key,
                normalized_session_id,
                ex=self.ttl_seconds,
            )

        except RedisError:
            # Session memory was already saved successfully.
            # Failure to write the discovery index must not make
            # the tutoring request fail.
            pass

    # =====================================================
    # DELETE
    # =====================================================

    def delete(
        self,
        *,
        user_id: str,
        session_id: str,
    ) -> bool:
        """
        Delete exactly one user/session memory.
        """

        normalized_user_id = (
            self._normalize_identifier(
                user_id
            )
        )

        normalized_session_id = (
            self._normalize_identifier(
                session_id
            )
        )

        if (
            normalized_user_id is None
            or normalized_session_id is None
        ):
            return False

        key = self._session_key(
            user_id=normalized_user_id,
            session_id=(
                normalized_session_id
            ),
        )

        index_key = self._session_index_key(
            user_id=normalized_user_id,
            session_id=normalized_session_id,
        )

        try:
            self.client.delete(
                key,
                index_key,
            )
            return True

        except RedisError:
            return False

    # =====================================================
    # EXISTS
    # =====================================================

    def exists(
        self,
        *,
        user_id: str,
        session_id: str,
    ) -> bool:
        normalized_user_id = (
            self._normalize_identifier(
                user_id
            )
        )

        normalized_session_id = (
            self._normalize_identifier(
                session_id
            )
        )

        if (
            normalized_user_id is None
            or normalized_session_id is None
        ):
            return False

        key = self._session_key(
            user_id=normalized_user_id,
            session_id=(
                normalized_session_id
            ),
        )

        try:
            return bool(
                self.client.exists(
                    key
                )
            )

        except RedisError:
            return False

    # =====================================================
    # TTL
    # =====================================================

    def ttl(
        self,
        *,
        user_id: str,
        session_id: str,
    ) -> int | None:
        """
        Return remaining TTL in seconds.

        None means:
        - session absent
        - invalid identifiers
        - Redis unavailable
        """

        normalized_user_id = (
            self._normalize_identifier(
                user_id
            )
        )

        normalized_session_id = (
            self._normalize_identifier(
                session_id
            )
        )

        if (
            normalized_user_id is None
            or normalized_session_id is None
        ):
            return None

        key = self._session_key(
            user_id=normalized_user_id,
            session_id=(
                normalized_session_id
            ),
        )

        try:
            remaining = int(
                self.client.ttl(
                    key
                )
            )

        except RedisError:
            return None

        if remaining < 0:
            return None

        return remaining

    # =====================================================
    # RECOVERABLE SESSION DISCOVERY
    # =====================================================

    def list_recoverable_sessions(
        self,
        *,
        user_id: str,
    ) -> list[dict[str, object]]:
        """
        Return this user's still-alive Redis sessions.

        New sessions saved after this feature exists have a small
        reversible index containing the original session_id.

        Older Redis snapshots created before this feature are still
        discoverable. They are returned with an opaque legacy
        reference such as:

            legacy:abc123...

        The caller can pass that reference to recover_session().
        """

        normalized_user_id = (
            self._normalize_identifier(
                user_id
            )
        )

        if normalized_user_id is None:
            return []

        user_hash = self._short_hash(
            normalized_user_id
        )

        pattern = (
            "phymentor:session:v1:"
            f"{user_hash}:*"
        )

        sessions: list[
            dict[str, object]
        ] = []

        try:
            keys = list(
                self.client.scan_iter(
                    match=pattern
                )
            )

        except RedisError:
            return []

        for raw_key in keys:
            key = str(
                raw_key
            )

            session_hash = (
                key.rsplit(
                    ":",
                    1,
                )[-1]
            )

            if not self._is_short_hash(
                session_hash
            ):
                continue

            try:
                remaining_ttl = int(
                    self.client.ttl(
                        key
                    )
                )

                raw_memory = (
                    self.client.get(
                        key
                    )
                )

            except RedisError:
                continue

            if (
                remaining_ttl <= 0
                or raw_memory is None
            ):
                continue

            try:
                memory = (
                    MemorySnapshot
                    .model_validate_json(
                        raw_memory
                    )
                )

            except Exception:
                # Discovery is read-only. A bad legacy record is
                # ignored here rather than deleted.
                continue

            memory = self._trim_memory(
                memory
            )

            index_key = (
                self
                ._session_index_key_from_hashes(
                    user_hash=user_hash,
                    session_hash=session_hash,
                )
            )

            indexed_session_id: (
                str | None
            ) = None

            try:
                candidate = (
                    self.client.get(
                        index_key
                    )
                )

            except RedisError:
                candidate = None

            if candidate:
                normalized_candidate = (
                    self._normalize_identifier(
                        str(candidate)
                    )
                )

                if (
                    normalized_candidate
                    and self._short_hash(
                        normalized_candidate
                    )
                    == session_hash
                ):
                    indexed_session_id = (
                        normalized_candidate
                    )

            is_legacy = (
                indexed_session_id
                is None
            )

            session_reference = (
                (
                    f"legacy:{session_hash}"
                )
                if is_legacy
                else indexed_session_id
            )

            preview = ""
            for message in reversed(
                list(
                    memory.recent_messages
                )
            ):
                raw_role = getattr(
                    message,
                    "role",
                    "",
                )

                role = str(
                    getattr(
                        raw_role,
                        "value",
                        raw_role,
                    )
                    or ""
                ).strip().casefold()

                content = str(
                    getattr(
                        message,
                        "content",
                        "",
                    )
                    or ""
                ).strip()

                if (
                    role == "user"
                    and content
                ):
                    preview = content
                    break

            if (
                not preview
                and memory.recent_messages
            ):
                preview = str(
                    memory.recent_messages[
                        -1
                    ].content
                ).strip()

            preview = preview[:160]

            document_names = [
                str(
                    document.name
                )
                for document in (
                    memory
                    .available_documents
                )
                if str(
                    document.name
                ).strip()
            ]

            sessions.append(
                {
                    "session_reference": (
                        session_reference
                    ),
                    "session_id": (
                        indexed_session_id
                    ),
                    "legacy": is_legacy,
                    "ttl_seconds": (
                        remaining_ttl
                    ),
                    "message_count": len(
                        memory
                        .recent_messages
                    ),
                    "preview": preview,
                    "document_names": (
                        document_names
                    ),
                }
            )

        # Because every successful save refreshes the session TTL,
        # a larger remaining TTL is a useful recency approximation.
        sessions.sort(
            key=lambda item: int(
                item.get(
                    "ttl_seconds",
                    0,
                )
                or 0
            ),
            reverse=True,
        )

        return sessions

    def recover_session(
        self,
        *,
        user_id: str,
        session_reference: str,
    ) -> tuple[
        str,
        MemorySnapshot,
    ] | None:
        """
        Resume one still-alive Redis session.

        For a normal indexed session, the same session_id is reused
        and its TTL is refreshed.

        For a pre-index legacy session, the old hashed snapshot is
        copied to a fresh known session_id. Only after the new copy
        is verified does this method remove the legacy key, avoiding
        duplicate entries without risking data loss.
        """

        normalized_user_id = (
            self._normalize_identifier(
                user_id
            )
        )

        normalized_reference = (
            self._normalize_identifier(
                session_reference
            )
        )

        if (
            normalized_user_id is None
            or normalized_reference is None
        ):
            return None

        legacy_prefix = "legacy:"

        if not normalized_reference.startswith(
            legacy_prefix
        ):
            memory = self.load(
                user_id=normalized_user_id,
                session_id=(
                    normalized_reference
                ),
            )

            if memory is None:
                return None

            # Re-saving refreshes the 24-hour TTL and makes sure
            # the reversible session index exists.
            self.save(
                user_id=normalized_user_id,
                session_id=(
                    normalized_reference
                ),
                memory=memory,
            )

            return (
                normalized_reference,
                memory,
            )

        session_hash = (
            normalized_reference[
                len(legacy_prefix):
            ]
        )

        if not self._is_short_hash(
            session_hash
        ):
            return None

        user_hash = self._short_hash(
            normalized_user_id
        )

        legacy_key = (
            self
            ._session_key_from_hashes(
                user_hash=user_hash,
                session_hash=session_hash,
            )
        )

        try:
            raw_memory = (
                self.client.get(
                    legacy_key
                )
            )

        except RedisError:
            return None

        if raw_memory is None:
            return None

        try:
            memory = (
                MemorySnapshot
                .model_validate_json(
                    raw_memory
                )
            )

        except Exception:
            return None

        memory = self._trim_memory(
            memory
        )

        recovered_session_id = str(
            uuid4()
        )

        self.save(
            user_id=normalized_user_id,
            session_id=(
                recovered_session_id
            ),
            memory=memory,
        )

        if not self.exists(
            user_id=normalized_user_id,
            session_id=(
                recovered_session_id
            ),
        ):
            return None

        # The new recoverable copy now exists. Remove only the old
        # legacy memory key so it does not appear twice in the UI.
        # No index key existed for a true legacy session.
        try:
            self.client.delete(
                legacy_key
            )

        except RedisError:
            # A duplicate legacy entry is harmless and will expire
            # naturally under its original TTL.
            pass

        return (
            recovered_session_id,
            memory,
        )

    # =====================================================
    # HEALTH
    # =====================================================

    def health_check(
        self,
    ) -> bool:
        try:
            return bool(
                self.client.ping()
            )

        except RedisError:
            return False

    # =====================================================
    # CLOSE
    # =====================================================

    def close(
        self,
    ) -> None:
        try:
            self.client.close()

        finally:
            self.pool.disconnect()

    # =====================================================
    # MEMORY TRIMMING
    # =====================================================

    @classmethod
    def _trim_memory(
        cls,
        memory: MemorySnapshot,
    ) -> MemorySnapshot:
        """
        Bound Redis session payload growth.

        - keep only the most recent conversation messages
        - keep only lightweight document references
        - deduplicate documents by document_id

        Actual files, chunks, embeddings and indexes are
        never stored in this Redis snapshot.
        """

        recent_messages = list(
            memory.recent_messages
        )[
            -cls.MAX_RECENT_MESSAGES:
        ]

        available_documents = (
            cls._deduplicate_documents(
                memory.available_documents
            )
        )[
            -cls.MAX_SESSION_DOCUMENTS:
        ]

        active_document_id = (
            memory.active_document_id
        )

        # Defensive cleanup:
        # if an active document ID is no longer present in
        # the bounded registry, do not keep a stale pointer.
        available_document_ids = {
            document.document_id
            for document in available_documents
        }

        if (
            active_document_id
            and active_document_id
            not in available_document_ids
            and available_documents
        ):
            active_document_id = (
                available_documents[-1]
                .document_id
            )

        return memory.model_copy(
            update={
                "available_documents": (
                    available_documents
                ),
                "active_document_id": (
                    active_document_id
                ),
                "recent_messages": (
                    recent_messages
                ),
            }
        )

    @classmethod
    def _merge_session_documents(
        cls,
        *,
        existing_memory: MemorySnapshot | None,
        incoming_memory: MemorySnapshot,
    ) -> MemorySnapshot:
        """
        Merge the session bookshelf by document_id.

        Older references are preserved unless the same
        document_id arrives again, in which case the incoming
        reference replaces that one entry. No duplicate copy
        of the document is created.
        """

        existing_documents = (
            existing_memory.available_documents
            if existing_memory is not None
            else []
        )

        merged_documents = (
            cls._deduplicate_documents(
                [
                    *existing_documents,
                    *incoming_memory.available_documents,
                ]
            )
        )[
            -cls.MAX_SESSION_DOCUMENTS:
        ]

        return incoming_memory.model_copy(
            update={
                "available_documents": (
                    merged_documents
                )
            }
        )

    @staticmethod
    def _deduplicate_documents(
        documents: list[
            SessionDocumentReference
        ],
    ) -> list[
        SessionDocumentReference
    ]:
        """
        Deduplicate by document_id while preserving useful
        recency order.

        If the same document appears again, the newer copy
        wins and moves to the end of the registry.
        """

        registry: dict[
            str,
            SessionDocumentReference,
        ] = {}

        for document in documents:
            document_id = (
                document.document_id.strip()
            )

            if not document_id:
                continue

            # Re-inserting an existing key should move it to
            # the end so order reflects recent references.
            registry.pop(
                document_id,
                None,
            )

            registry[document_id] = document

        return list(
            registry.values()
        )

    # =====================================================
    # REDIS KEY
    # =====================================================

    @classmethod
    def _session_key(
        cls,
        *,
        user_id: str,
        session_id: str,
    ) -> str:
        """
        Isolation is based on BOTH user and session.

        Example:

        user-A + session-1
        !=
        user-B + session-1
        """

        user_hash = cls._short_hash(
            user_id
        )

        session_hash = cls._short_hash(
            session_id
        )

        return (
            "phymentor:session:v1:"
            f"{user_hash}:"
            f"{session_hash}"
        )

    @classmethod
    def _session_index_key(
        cls,
        *,
        user_id: str,
        session_id: str,
    ) -> str:
        return (
            cls
            ._session_index_key_from_hashes(
                user_hash=cls._short_hash(
                    user_id
                ),
                session_hash=cls._short_hash(
                    session_id
                ),
            )
        )

    @staticmethod
    def _session_index_key_from_hashes(
        *,
        user_hash: str,
        session_hash: str,
    ) -> str:
        return (
            "phymentor:session-index:v1:"
            f"{user_hash}:"
            f"{session_hash}"
        )

    @staticmethod
    def _session_key_from_hashes(
        *,
        user_hash: str,
        session_hash: str,
    ) -> str:
        return (
            "phymentor:session:v1:"
            f"{user_hash}:"
            f"{session_hash}"
        )

    @staticmethod
    def _is_short_hash(
        value: str,
    ) -> bool:
        if len(value) != 24:
            return False

        return all(
            character in (
                "0123456789abcdef"
            )
            for character in value
        )

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _normalize_identifier(
        value: str,
    ) -> str | None:
        normalized = value.strip()

        return (
            normalized
            if normalized
            else None
        )

    @staticmethod
    def _short_hash(
        value: str,
    ) -> str:
        return hashlib.sha256(
            value.encode(
                "utf-8"
            )
        ).hexdigest()[:24]


__all__ = [
    "RedisSessionStore",
]