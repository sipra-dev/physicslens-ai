from __future__ import annotations

from threading import Lock

from langgraph.checkpoint.postgres import (
    PostgresSaver,
)
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.config import settings


class LangGraphCheckpointManager:
    """
    Production PostgreSQL checkpoint manager
    for the PhyMentor AI LangGraph.

    Important:
    This is EXECUTION persistence.

    It is separate from:
    - Redis session memory
    - PostgreSQL learner profile memory
    - Pinecone semantic learning memory

    Responsibilities:
    - own a PostgreSQL connection pool
    - expose one reusable PostgresSaver
    - initialise LangGraph checkpoint tables
    - provide a lightweight health check
    - close database resources cleanly
    """

    DEFAULT_MIN_POOL_SIZE = 1
    DEFAULT_MAX_POOL_SIZE = 5
    DEFAULT_POOL_TIMEOUT_SECONDS = 10.0

    def __init__(
        self,
        *,
        database_url: str | None = None,
        min_pool_size: int = (
            DEFAULT_MIN_POOL_SIZE
        ),
        max_pool_size: int = (
            DEFAULT_MAX_POOL_SIZE
        ),
        pool_timeout_seconds: float = (
            DEFAULT_POOL_TIMEOUT_SECONDS
        ),
    ) -> None:
        resolved_database_url = (
            database_url
            or settings.database_url
        )

        if not resolved_database_url:
            raise RuntimeError(
                "DATABASE_URL is not configured."
            )

        if min_pool_size < 0:
            raise ValueError(
                "min_pool_size cannot be negative."
            )

        if max_pool_size <= 0:
            raise ValueError(
                "max_pool_size must be positive."
            )

        if (
            min_pool_size
            > max_pool_size
        ):
            raise ValueError(
                "min_pool_size cannot exceed "
                "max_pool_size."
            )

        if pool_timeout_seconds <= 0:
            raise ValueError(
                "pool_timeout_seconds "
                "must be positive."
            )

        self.database_url = (
            resolved_database_url
        )

        self._pool = ConnectionPool(
            conninfo=self.database_url,
            min_size=min_pool_size,
            max_size=max_pool_size,
            timeout=pool_timeout_seconds,

            # Do not connect merely because this
            # module was imported.
            open=False,

            # LangGraph's PostgresSaver requires
            # autocommit + dictionary-style rows.
            kwargs={
                "autocommit": True,
                "row_factory": dict_row,

                # Avoid prepared-statement issues
                # with pooled / proxied Postgres.
                "prepare_threshold": 0,
            },
        )

        self._checkpointer = (
            PostgresSaver(
                self._pool
            )
        )

        self._lock = Lock()

        self._opened = False
        self._setup_complete = False

    # =========================================================
    # LIFECYCLE
    # =========================================================

    def open(
        self,
    ) -> None:
        """
        Open the PostgreSQL pool once.

        Safe to call repeatedly.
        """

        if self._opened:
            return

        with self._lock:
            if self._opened:
                return

            self._pool.open(
                wait=True
            )

            self._opened = True

    def setup(
        self,
    ) -> None:
        """
        Create / migrate LangGraph checkpoint
        tables.

        PostgresSaver.setup() is migration-aware,
        so repeated application startup calls are
        safe from a schema-version perspective.

        We still avoid repeating it inside this
        process.
        """

        if self._setup_complete:
            return

        with self._lock:
            if self._setup_complete:
                return

            if not self._opened:
                self._pool.open(
                    wait=True
                )
                self._opened = True

            self._checkpointer.setup()

            self._setup_complete = True

    def close(
        self,
    ) -> None:
        """
        Close PostgreSQL pool resources.

        Safe to call more than once.
        """

        with self._lock:
            if not self._opened:
                return

            self._pool.close()

            self._opened = False

    # =========================================================
    # CHECKPOINTER ACCESS
    # =========================================================

    @property
    def checkpointer(
        self,
    ) -> PostgresSaver:
        """
        Return the LangGraph-compatible saver.

        Opening/setup remains explicit so merely
        importing application modules never causes
        hidden database mutations.
        """

        return self._checkpointer

    # =========================================================
    # HEALTH
    # =========================================================

    def health_check(
        self,
    ) -> bool:
        """
        Verify that the checkpoint PostgreSQL
        connection pool can execute a trivial query.

        Failure returns False rather than crashing
        the health endpoint.
        """

        try:
            self.open()

            with self._pool.connection() as conn:
                row = conn.execute(
                    "SELECT 1 AS ok"
                ).fetchone()

            return bool(
                row
                and row.get("ok") == 1
            )

        except Exception:
            return False


# =========================================================
# SHARED CHECKPOINT MANAGER
# =========================================================

_checkpoint_manager: (
    LangGraphCheckpointManager | None
) = None

_checkpoint_manager_lock = Lock()


def get_checkpoint_manager(
) -> LangGraphCheckpointManager:
    """
    Return the one shared LangGraph checkpoint
    manager for this application process.

    The manager is created lazily.

    Importing this module does NOT:
    - open PostgreSQL connections
    - create checkpoint tables
    - mutate the database
    """

    global _checkpoint_manager

    if _checkpoint_manager is not None:
        return _checkpoint_manager

    with _checkpoint_manager_lock:
        if _checkpoint_manager is None:
            _checkpoint_manager = (
                LangGraphCheckpointManager()
            )

    return _checkpoint_manager