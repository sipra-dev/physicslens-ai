from __future__ import annotations

import os
from typing import Any

import httpx


class APIClientError(RuntimeError):
    """
    Friendly frontend exception for FastAPI request failures.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class PhyMentorAPIClient:
    """
    Small HTTP client used by the Streamlit frontend.

    Architecture:

        Streamlit
            ↓ HTTP
        FastAPI
            ↓
        LangGraph / backend services

    The frontend never talks directly to Redis,
    FAISS, PostgreSQL, Pinecone, or OpenAI.
    """

    def __init__(
        self,
        base_url: str | None = None,
    ) -> None:

        configured_base_url = (
            base_url
            or os.getenv(
                "PHYMENTOR_API_BASE_URL",
                "http://127.0.0.1:8000/v1",
            )
        )

        self.base_url = (
            configured_base_url.rstrip("/")
        )

        self.client = httpx.Client(
            base_url=self.base_url,
            follow_redirects=True,
            timeout=httpx.Timeout(
                connect=10.0,
                read=120.0,
                write=120.0,
                pool=10.0,
            ),
        )

    # =====================================================
    # INTERNAL HELPERS
    # =====================================================

    @staticmethod
    def _normalize_required_string(
        value: str,
        *,
        field_name: str,
    ) -> str:

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return normalized

    @staticmethod
    def _normalize_optional_string(
        value: str | None,
    ) -> str | None:

        if value is None:
            return None

        normalized = value.strip()

        return normalized or None

    @classmethod
    def _normalize_available_documents(
        cls,
        documents: list[
            dict[str, Any]
        ]
        | None,
    ) -> list[
        dict[str, str]
    ]:
        """
        Normalize the lightweight document registry before
        sending it to FastAPI.

        Expected input:

        [
            {
                "document_id": "abc123",
                "name": "nuclear_fission.jpg",
            },
            {
                "document_id": "xyz789",
                "name": "shm_notes.pdf",
            },
        ]

        Invalid/incomplete entries are ignored.

        Duplicate document IDs are removed while preserving
        the original order.
        """

        if not documents:
            return []

        normalized_documents: list[
            dict[str, str]
        ] = []

        seen_document_ids: set[
            str
        ] = set()

        for document in documents:

            if not isinstance(
                document,
                dict,
            ):
                continue

            raw_document_id = (
                document.get(
                    "document_id"
                )
            )

            raw_name = document.get(
                "name"
            )

            if not isinstance(
                raw_document_id,
                str,
            ):
                continue

            if not isinstance(
                raw_name,
                str,
            ):
                continue

            document_id = (
                cls._normalize_optional_string(
                    raw_document_id
                )
            )

            name = (
                cls._normalize_optional_string(
                    raw_name
                )
            )

            if (
                document_id is None
                or name is None
            ):
                continue

            if (
                document_id
                in seen_document_ids
            ):
                continue

            seen_document_ids.add(
                document_id
            )

            normalized_documents.append(
                {
                    "document_id": (
                        document_id
                    ),
                    "name": name,
                }
            )

        return normalized_documents

    @staticmethod
    def _extract_error_message(
        response: httpx.Response,
    ) -> str:

        try:
            payload = response.json()

        except ValueError:
            return (
                response.text.strip()
                or "The API request failed."
            )

        detail = payload.get(
            "detail"
        )

        if isinstance(
            detail,
            str,
        ):
            return detail

        if detail is not None:
            return str(detail)

        return (
            payload.get("message")
            or "The API request failed."
        )

    def _handle_response(
        self,
        response: httpx.Response,
    ) -> dict[str, Any]:

        if response.is_error:
            raise APIClientError(
                self._extract_error_message(
                    response
                ),
                status_code=(
                    response.status_code
                ),
            )

        try:
            payload = response.json()

        except ValueError as exc:
            raise APIClientError(
                (
                    "The API returned an "
                    "invalid JSON response."
                ),
                status_code=(
                    response.status_code
                ),
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise APIClientError(
                (
                    "The API returned an "
                    "unexpected response format."
                ),
                status_code=(
                    response.status_code
                ),
            )

        return payload

    # =====================================================
    # DOCUMENT UPLOAD
    # =====================================================

    def upload_document(
        self,
        *,
        user_id: str,
        filename: str,
        content_type: str,
        file_bytes: bytes,
    ) -> dict[str, Any]:

        normalized_user_id = (
            self._normalize_required_string(
                user_id,
                field_name="user_id",
            )
        )

        normalized_filename = (
            self._normalize_required_string(
                filename,
                field_name="filename",
            )
        )

        response = self.client.post(
            "/documents/upload",
            data={
                "user_id": (
                    normalized_user_id
                ),
            },
            files={
                "file": (
                    normalized_filename,
                    file_bytes,
                    content_type,
                ),
            },
        )

        return self._handle_response(
            response
        )

    # =====================================================
    # DOCUMENT STATUS
    # =====================================================

    def get_document_status(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> dict[str, Any]:

        normalized_user_id = (
            self._normalize_required_string(
                user_id,
                field_name="user_id",
            )
        )

        normalized_document_id = (
            self._normalize_required_string(
                document_id,
                field_name="document_id",
            )
        )

        response = self.client.get(
            (
                f"/documents/"
                f"{normalized_document_id}/status"
            ),
            params={
                "user_id": (
                    normalized_user_id
                ),
            },
        )

        return self._handle_response(
            response
        )

    # =====================================================
    # CHAT
    # =====================================================

    def chat(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,

        # Backward-compatible current/recent document.
        document_id: str | None = None,

        # All documents known to this frontend session.
        available_documents: (
            list[dict[str, Any]]
            | None
        ) = None,

        selected_page: int | None = None,
        selected_figure_id: (
            str | None
        ) = None,

        selected_model: str | None = None,

        language: str = "unknown",
    ) -> dict[str, Any]:

        normalized_user_id = (
            self._normalize_required_string(
                user_id,
                field_name="user_id",
            )
        )

        normalized_session_id = (
            self._normalize_required_string(
                session_id,
                field_name="session_id",
            )
        )

        normalized_query = (
            self._normalize_required_string(
                query,
                field_name="query",
            )
        )

        normalized_document_id = (
            self._normalize_optional_string(
                document_id
            )
        )

        normalized_documents = (
            self._normalize_available_documents(
                available_documents
            )
        )

        normalized_figure_id = (
            self._normalize_optional_string(
                selected_figure_id
            )
        )

        normalized_selected_model = (
            self._normalize_optional_string(
                selected_model
            )
        )

        if (
            selected_page is not None
            and selected_page < 1
        ):
            raise ValueError(
                (
                    "selected_page must "
                    "be at least 1."
                )
            )

        payload = {
            "session_id": (
                normalized_session_id
            ),

            "query": (
                normalized_query
            ),

            # Still included for backward compatibility
            # and recent-document conversational context.
            "document_id": (
                normalized_document_id
            ),

            # NEW:
            # Give the backend all documents currently
            # available in this chat session.
            "available_documents": (
                normalized_documents
            ),

            "selected_page": (
                selected_page
            ),

            "selected_figure_id": (
                normalized_figure_id
            ),

            "selected_model": (
                normalized_selected_model
            ),

            "language": language,
        }

        response = self.client.post(
            "/chat",
            json=payload,

            # user_id is deliberately NOT placed
            # inside ChatRequest JSON.
            headers={
                "X-User-ID": (
                    normalized_user_id
                ),
            },
        )

        return self._handle_response(
            response
        )

    # =====================================================
    # RECOVERABLE CHAT SESSIONS
    # =====================================================

    def list_recoverable_sessions(
        self,
        *,
        user_id: str,
    ) -> dict[str, Any]:
        """
        Return this user's Redis-backed chat sessions that are
        still inside the short-term recovery window.

        The frontend stays behind the normal FastAPI HTTP boundary;
        it never talks to Redis directly.
        """

        normalized_user_id = (
            self._normalize_required_string(
                user_id,
                field_name="user_id",
            )
        )

        response = self.client.get(
            "/chat/sessions",
            headers={
                "X-User-ID": (
                    normalized_user_id
                ),
            },
        )

        return self._handle_response(
            response
        )

    def recover_chat_session(
        self,
        *,
        user_id: str,
        session_reference: str,
    ) -> dict[str, Any]:
        """
        Recover one still-alive Redis chat session.

        Indexed sessions resume with their original session ID.
        Legacy pre-index sessions may be migrated by the backend
        to a fresh recoverable session ID.
        """

        normalized_user_id = (
            self._normalize_required_string(
                user_id,
                field_name="user_id",
            )
        )

        normalized_session_reference = (
            self._normalize_required_string(
                session_reference,
                field_name=(
                    "session_reference"
                ),
            )
        )

        response = self.client.post(
            "/chat/sessions/recover",
            json={
                "session_reference": (
                    normalized_session_reference
                ),
            },
            headers={
                "X-User-ID": (
                    normalized_user_id
                ),
            },
        )

        return self._handle_response(
            response
        )

    # =====================================================
    # SELECTED-TEXT EXPLANATION
    # =====================================================

    def explain_selection(
        self,
        *,
        user_id: str,
        session_id: str,
        selected_text: str,
        surrounding_text: str | None = None,
        document_id: str | None = None,
        selected_page: int | None = None,
        selected_figure_id: (
            str | None
        ) = None,
        selected_model: str | None = None,
    ) -> dict[str, Any]:
        """
        Ask the lightweight backend endpoint to explain text
        selected from a rendered PhyMentor answer.

        This does not call Redis/Pinecone/OpenAI directly from
        the frontend. It stays inside the normal HTTP boundary.
        """

        normalized_user_id = (
            self._normalize_required_string(
                user_id,
                field_name="user_id",
            )
        )

        normalized_session_id = (
            self._normalize_required_string(
                session_id,
                field_name="session_id",
            )
        )

        normalized_selected_text = (
            self._normalize_required_string(
                selected_text,
                field_name="selected_text",
            )
        )

        normalized_surrounding_text = (
            self._normalize_optional_string(
                surrounding_text
            )
        )

        normalized_document_id = (
            self._normalize_optional_string(
                document_id
            )
        )

        normalized_figure_id = (
            self._normalize_optional_string(
                selected_figure_id
            )
        )

        normalized_selected_model = (
            self._normalize_optional_string(
                selected_model
            )
        )

        if (
            selected_page is not None
            and selected_page < 1
        ):
            raise ValueError(
                (
                    "selected_page must "
                    "be at least 1."
                )
            )

        payload = {
            "session_id": (
                normalized_session_id
            ),
            "selected_text": (
                normalized_selected_text
            ),
            "surrounding_text": (
                normalized_surrounding_text
            ),
            "document_id": (
                normalized_document_id
            ),
            "selected_page": (
                selected_page
            ),
            "selected_figure_id": (
                normalized_figure_id
            ),
            "selected_model": (
                normalized_selected_model
            ),
        }

        response = self.client.post(
            "/chat/selection-explain",
            json=payload,
            headers={
                "X-User-ID": (
                    normalized_user_id
                ),
            },
        )

        return self._handle_response(
            response
        )

    # =====================================================
    # REINDEX DOCUMENT
    # =====================================================

    def reindex_document(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> dict[str, Any]:

        normalized_user_id = (
            self._normalize_required_string(
                user_id,
                field_name="user_id",
            )
        )

        normalized_document_id = (
            self._normalize_required_string(
                document_id,
                field_name="document_id",
            )
        )

        response = self.client.post(
            (
                f"/documents/"
                f"{normalized_document_id}/reindex"
            ),
            params={
                "user_id": (
                    normalized_user_id
                ),
            },
            headers={
                "X-User-ID": (
                    normalized_user_id
                ),
            },
        )

        return self._handle_response(
            response
        )

    # =====================================================
    # DELETE DOCUMENT
    # =====================================================

    def delete_document(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> dict[str, Any]:

        normalized_user_id = (
            self._normalize_required_string(
                user_id,
                field_name="user_id",
            )
        )

        normalized_document_id = (
            self._normalize_required_string(
                document_id,
                field_name="document_id",
            )
        )

        response = self.client.delete(
            (
                f"/documents/"
                f"{normalized_document_id}"
            ),
            params={
                "user_id": (
                    normalized_user_id
                ),
            },
            headers={
                "X-User-ID": (
                    normalized_user_id
                ),
            },
        )

        return self._handle_response(
            response
        )

    # =====================================================
    # CLEANUP
    # =====================================================

    def close(
        self,
    ) -> None:

        self.client.close()