from __future__ import annotations

from typing import Any

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

from src.config.settings import settings
from src.memory.long_term_models import (
    LongTermMemoryProfile,
    MisconceptionRecord,
    TopicProgress,
)


class PostgresLongTermMemoryStore:
    """
    PostgreSQL-backed durable learning memory.

    Stores:
    - preferred language
    - grade
    - learning style
    - topic progress
    - stable misconceptions

    This is intentionally separate from Redis
    short-term/session memory.
    """

    def __init__(
        self,
        *,
        database_url: str | None = None,
    ) -> None:
        resolved_url = (
            database_url
            or settings.database_url
        )

        if (
            resolved_url is None
            or not resolved_url.strip()
        ):
            raise RuntimeError(
                "DATABASE_URL is not configured."
            )

        self.database_url = (
            resolved_url.strip()
        )

    # =========================================================
    # CONNECTION
    # =========================================================

    def _connect(
        self,
    ) -> Connection:
        return psycopg.connect(
            self.database_url,
            connect_timeout=10,
        )

    # =========================================================
    # HEALTH CHECK
    # =========================================================

    def health_check(
        self,
    ) -> bool:
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT 1;"
                    )

                    row = cursor.fetchone()

                    return bool(
                        row
                        and row[0] == 1
                    )

        except psycopg.Error:
            return False

    # =========================================================
    # SCHEMA
    # =========================================================

    def initialize_schema(
        self,
    ) -> None:
        """
        Create durable-memory tables if they
        do not already exist.
        """

        with self._connect() as connection:
            with connection.cursor() as cursor:

                # ---------------------------------------------
                # 1. Main user learning profile
                # ---------------------------------------------

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    phymentor_user_profiles (
                        user_id VARCHAR(255)
                            PRIMARY KEY,

                        preferred_language
                            VARCHAR(50),

                        grade SMALLINT
                            CHECK (
                                grade IS NULL
                                OR (
                                    grade >= 1
                                    AND grade <= 12
                                )
                            ),

                        learning_style
                            VARCHAR(200),

                        created_at
                            TIMESTAMPTZ
                            NOT NULL,

                        updated_at
                            TIMESTAMPTZ
                            NOT NULL
                    );
                    """
                )

                # ---------------------------------------------
                # 2. Topic progress
                # ---------------------------------------------

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    phymentor_topic_progress (
                        user_id VARCHAR(255)
                            NOT NULL,

                        topic VARCHAR(200)
                            NOT NULL,

                        mastery_score
                            DOUBLE PRECISION
                            NOT NULL
                            DEFAULT 0.0
                            CHECK (
                                mastery_score >= 0.0
                                AND mastery_score <= 1.0
                            ),

                        attempts INTEGER
                            NOT NULL
                            DEFAULT 0
                            CHECK (
                                attempts >= 0
                            ),

                        correct_attempts INTEGER
                            NOT NULL
                            DEFAULT 0
                            CHECK (
                                correct_attempts >= 0
                            ),

                        last_seen_at
                            TIMESTAMPTZ
                            NOT NULL,

                        updated_at
                            TIMESTAMPTZ
                            NOT NULL,

                        PRIMARY KEY (
                            user_id,
                            topic
                        ),

                        CONSTRAINT
                            fk_topic_progress_user
                        FOREIGN KEY (
                            user_id
                        )
                        REFERENCES
                            phymentor_user_profiles(
                                user_id
                            )
                        ON DELETE CASCADE
                    );
                    """
                )

                # ---------------------------------------------
                # 3. Stable misconceptions
                # ---------------------------------------------

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    phymentor_misconceptions (
                        user_id VARCHAR(255)
                            NOT NULL,

                        concept VARCHAR(200)
                            NOT NULL,

                        description VARCHAR(2000)
                            NOT NULL,

                        confidence
                            DOUBLE PRECISION
                            NOT NULL
                            CHECK (
                                confidence >= 0.0
                                AND confidence <= 1.0
                            ),

                        status VARCHAR(20)
                            NOT NULL
                            CHECK (
                                status IN (
                                    'active',
                                    'improving',
                                    'resolved'
                                )
                            ),

                        source VARCHAR(500),

                        first_seen_at
                            TIMESTAMPTZ
                            NOT NULL,

                        last_seen_at
                            TIMESTAMPTZ
                            NOT NULL,

                        PRIMARY KEY (
                            user_id,
                            concept
                        ),

                        CONSTRAINT
                            fk_misconception_user
                        FOREIGN KEY (
                            user_id
                        )
                        REFERENCES
                            phymentor_user_profiles(
                                user_id
                            )
                        ON DELETE CASCADE
                    );
                    """
                )

                # ---------------------------------------------
                # Helpful indexes
                # ---------------------------------------------

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_phymentor_topic_progress_user
                    ON phymentor_topic_progress(
                        user_id
                    );
                    """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_phymentor_misconceptions_user_status
                    ON phymentor_misconceptions(
                        user_id,
                        status
                    );
                    """
                )

            connection.commit()

    # =========================================================
    # SAVE / UPSERT PROFILE
    # =========================================================

    def save_profile(
        self,
        profile: LongTermMemoryProfile,
    ) -> None:
        """
        Persist the full durable profile.

        The operation is transactional:
        either the whole profile is saved,
        or none of it is.
        """

        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:

                    # -----------------------------------------
                    # Main profile
                    # -----------------------------------------

                    cursor.execute(
                        """
                        INSERT INTO
                            phymentor_user_profiles (
                                user_id,
                                preferred_language,
                                grade,
                                learning_style,
                                created_at,
                                updated_at
                            )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s
                        )
                        ON CONFLICT (
                            user_id
                        )
                        DO UPDATE SET
                            preferred_language =
                                EXCLUDED.preferred_language,

                            grade =
                                EXCLUDED.grade,

                            learning_style =
                                EXCLUDED.learning_style,

                            updated_at =
                                EXCLUDED.updated_at;
                        """,
                        (
                            profile.user_id,

                            (
                                profile
                                .preferred_language
                                .value
                                if profile
                                .preferred_language
                                is not None
                                else None
                            ),

                            profile.grade,

                            profile.learning_style,

                            profile.created_at,

                            profile.updated_at,
                        ),
                    )

                    # -----------------------------------------
                    # Replace current topic-progress snapshot
                    # -----------------------------------------

                    cursor.execute(
                        """
                        DELETE FROM
                            phymentor_topic_progress
                        WHERE
                            user_id = %s;
                        """,
                        (
                            profile.user_id,
                        ),
                    )

                    if profile.progress:
                        cursor.executemany(
                            """
                            INSERT INTO
                                phymentor_topic_progress (
                                    user_id,
                                    topic,
                                    mastery_score,
                                    attempts,
                                    correct_attempts,
                                    last_seen_at,
                                    updated_at
                                )
                            VALUES (
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s
                            );
                            """,
                            [
                                (
                                    profile.user_id,
                                    item.topic,
                                    item.mastery_score,
                                    item.attempts,
                                    item.correct_attempts,
                                    item.last_seen_at,
                                    item.updated_at,
                                )
                                for item
                                in profile.progress
                            ],
                        )

                    # -----------------------------------------
                    # Replace misconception snapshot
                    # -----------------------------------------

                    cursor.execute(
                        """
                        DELETE FROM
                            phymentor_misconceptions
                        WHERE
                            user_id = %s;
                        """,
                        (
                            profile.user_id,
                        ),
                    )

                    if profile.misconceptions:
                        cursor.executemany(
                            """
                            INSERT INTO
                                phymentor_misconceptions (
                                    user_id,
                                    concept,
                                    description,
                                    confidence,
                                    status,
                                    source,
                                    first_seen_at,
                                    last_seen_at
                                )
                            VALUES (
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s
                            );
                            """,
                            [
                                (
                                    profile.user_id,
                                    item.concept,
                                    item.description,
                                    item.confidence,
                                    item.status,
                                    item.source,
                                    item.first_seen_at,
                                    item.last_seen_at,
                                )
                                for item
                                in profile.misconceptions
                            ],
                        )

                connection.commit()

            except Exception:
                connection.rollback()
                raise

    # =========================================================
    # LOAD PROFILE
    # =========================================================

    def load_profile(
        self,
        *,
        user_id: str,
    ) -> LongTermMemoryProfile | None:
        normalized_user_id = (
            user_id.strip()
        )

        if not normalized_user_id:
            return None

        with self._connect() as connection:
            with connection.cursor(
                row_factory=dict_row
            ) as cursor:

                # -----------------------------------------
                # Main profile
                # -----------------------------------------

                cursor.execute(
                    """
                    SELECT
                        user_id,
                        preferred_language,
                        grade,
                        learning_style,
                        created_at,
                        updated_at
                    FROM
                        phymentor_user_profiles
                    WHERE
                        user_id = %s;
                    """,
                    (
                        normalized_user_id,
                    ),
                )

                profile_row = (
                    cursor.fetchone()
                )

                if profile_row is None:
                    return None

                # -----------------------------------------
                # Topic progress
                # -----------------------------------------

                cursor.execute(
                    """
                    SELECT
                        topic,
                        mastery_score,
                        attempts,
                        correct_attempts,
                        last_seen_at,
                        updated_at
                    FROM
                        phymentor_topic_progress
                    WHERE
                        user_id = %s
                    ORDER BY
                        topic ASC;
                    """,
                    (
                        normalized_user_id,
                    ),
                )

                progress_rows = (
                    cursor.fetchall()
                )

                # -----------------------------------------
                # Misconceptions
                # -----------------------------------------

                cursor.execute(
                    """
                    SELECT
                        concept,
                        description,
                        confidence,
                        status,
                        source,
                        first_seen_at,
                        last_seen_at
                    FROM
                        phymentor_misconceptions
                    WHERE
                        user_id = %s
                    ORDER BY
                        last_seen_at DESC;
                    """,
                    (
                        normalized_user_id,
                    ),
                )

                misconception_rows = (
                    cursor.fetchall()
                )

        progress = [
            TopicProgress(
                topic=row["topic"],
                mastery_score=(
                    row[
                        "mastery_score"
                    ]
                ),
                attempts=row["attempts"],
                correct_attempts=(
                    row[
                        "correct_attempts"
                    ]
                ),
                last_seen_at=(
                    row[
                        "last_seen_at"
                    ]
                ),
                updated_at=(
                    row[
                        "updated_at"
                    ]
                ),
            )
            for row
            in progress_rows
        ]

        misconceptions = [
            MisconceptionRecord(
                concept=row["concept"],
                description=(
                    row[
                        "description"
                    ]
                ),
                confidence=(
                    row[
                        "confidence"
                    ]
                ),
                status=row["status"],
                source=row["source"],
                first_seen_at=(
                    row[
                        "first_seen_at"
                    ]
                ),
                last_seen_at=(
                    row[
                        "last_seen_at"
                    ]
                ),
            )
            for row
            in misconception_rows
        ]

        return LongTermMemoryProfile(
            user_id=(
                profile_row[
                    "user_id"
                ]
            ),
            preferred_language=(
                profile_row[
                    "preferred_language"
                ]
            ),
            grade=profile_row["grade"],
            learning_style=(
                profile_row[
                    "learning_style"
                ]
            ),
            progress=progress,
            misconceptions=(
                misconceptions
            ),
            created_at=(
                profile_row[
                    "created_at"
                ]
            ),
            updated_at=(
                profile_row[
                    "updated_at"
                ]
            ),
        )

    # =========================================================
    # DELETE
    # =========================================================

    def delete_profile(
        self,
        *,
        user_id: str,
    ) -> bool:
        normalized_user_id = (
            user_id.strip()
        )

        if not normalized_user_id:
            return False

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM
                        phymentor_user_profiles
                    WHERE
                        user_id = %s;
                    """,
                    (
                        normalized_user_id,
                    ),
                )

                deleted = (
                    cursor.rowcount > 0
                )

            connection.commit()

        return deleted


__all__ = [
    "PostgresLongTermMemoryStore",
]