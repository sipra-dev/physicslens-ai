from __future__ import annotations

from uuid import uuid4

import streamlit as st


def initialize_session_state() -> None:
    """
    Create all frontend state needed by PhyMentor AI.

    Important idea:

        one Streamlit chat session
            =
        one document bookshelf

    Every uploaded document stays in that bookshelf
    until:
    - the student deletes it, or
    - the student starts a completely new session.
    """

    defaults = {
        # -------------------------------------------------
        # USER / SESSION
        # -------------------------------------------------
        "user_id": "local-user",
        "session_id": str(uuid4()),

        # -------------------------------------------------
        # DOCUMENT BOOKSHELF FOR THIS SESSION
        #
        # Shape:
        #
        # {
        #     document_id: {
        #         "document_id": "...",
        #         "name": "...",
        #         "status": "...",
        #         "processing_stage": "...",
        #         "active_page": None,
        #         "active_figure": None,
        #     }
        # }
        #
        # Python dict insertion order is useful here:
        # the last item is normally the most recently
        # uploaded/re-uploaded document.
        # -------------------------------------------------
        "uploaded_documents": {},

        # -------------------------------------------------
        # RECENT / CONTEXTUAL DOCUMENT
        #
        # This is NOT a manual mode switch.
        #
        # It simply remembers the document most recently
        # uploaded or used so phrases such as:
        #
        #     "explain this diagram"
        #     "based on this"
        #
        # have a sensible conversational fallback.
        #
        # The backend may still automatically choose any
        # other document from uploaded_documents.
        # -------------------------------------------------
        "active_document_id": None,
        "active_document_name": None,
        "active_document_status": None,
        "active_processing_stage": None,

        # -------------------------------------------------
        # CURRENT PAGE / FIGURE CONTEXT
        # -------------------------------------------------
        "active_page": None,
        "active_figure": None,

        # -------------------------------------------------
        # STUDENT PREFERENCES
        # -------------------------------------------------
        "language": "en",
        "grade": None,

        # -------------------------------------------------
        # CHAT HISTORY
        # -------------------------------------------------
        "messages": [],

        # -------------------------------------------------
        # FRONTEND REQUEST STATE
        # -------------------------------------------------
        "is_uploading": False,
        "is_processing": False,
        "is_chatting": False,

        # -------------------------------------------------
        # UI FEEDBACK
        # -------------------------------------------------
        "last_error": None,
        "last_notice": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # -----------------------------------------------------
    # DEFENSIVE REGISTRY REPAIR
    # -----------------------------------------------------

    if not isinstance(
        st.session_state.uploaded_documents,
        dict,
    ):
        st.session_state.uploaded_documents = {}

    # -----------------------------------------------------
    # HOT-RELOAD / OLD FRONTEND MIGRATION
    #
    # If Streamlit already had one document stored in the
    # older single-document fields, preserve it by putting
    # it into the new bookshelf.
    # -----------------------------------------------------

    active_document_id = (
        st.session_state.active_document_id
    )

    if (
        active_document_id
        and active_document_id
        not in st.session_state.uploaded_documents
    ):
        document_name = (
            st.session_state.active_document_name
            or (
                "Document "
                f"{active_document_id[:8]}"
            )
        )

        st.session_state.uploaded_documents[
            active_document_id
        ] = {
            "document_id": (
                active_document_id
            ),
            "name": document_name,
            "status": (
                st.session_state
                .active_document_status
            ),
            "processing_stage": (
                st.session_state
                .active_processing_stage
            ),
            "active_page": (
                st.session_state.active_page
            ),
            "active_figure": (
                st.session_state.active_figure
            ),
        }


def register_document(
    *,
    document_id: str,
    name: str,
    status: str | None = None,
    processing_stage: str | None = None,
    make_active: bool = True,
) -> None:
    """
    Add or update ONE document in this session's bookshelf.

    Uploading another document does NOT remove any older
    document.

    Example:

        upload A -> [A]
        upload B -> [A, B]
        upload C -> [A, B, C]

    The newest/re-uploaded document becomes the recent
    contextual document by default.
    """

    normalized_document_id = (
        document_id.strip()
    )

    normalized_name = (
        name.strip()
    )

    if not normalized_document_id:
        raise ValueError(
            "document_id cannot be empty."
        )

    if not normalized_name:
        raise ValueError(
            "document name cannot be empty."
        )

    registry = (
        st.session_state.uploaded_documents
    )

    # Pop first so that a re-uploaded/deduplicated document
    # moves to the end of the ordered registry and therefore
    # becomes the most recently referenced document.
    existing = registry.pop(
        normalized_document_id,
        {},
    )

    registry[
        normalized_document_id
    ] = {
        "document_id": (
            normalized_document_id
        ),
        "name": (
            normalized_name
        ),
        "status": (
            status
            if status is not None
            else existing.get(
                "status"
            )
        ),
        "processing_stage": (
            processing_stage
            if processing_stage is not None
            else existing.get(
                "processing_stage"
            )
        ),
        "active_page": (
            existing.get(
                "active_page"
            )
        ),
        "active_figure": (
            existing.get(
                "active_figure"
            )
        ),
    }

    if make_active:
        set_active_document(
            normalized_document_id
        )


def set_active_document(
    document_id: str,
) -> None:
    """
    Internally mark a document as the most recent/contextual
    document.

    IMPORTANT:

    This function does NOT mean the student manually switches
    documents.

    The automatic backend resolver may call/use another
    document for a later question.

    This recent document is mainly useful for phrases like:

        "this PDF"
        "this diagram"
        "based on this"
    """

    normalized_document_id = (
        document_id.strip()
    )

    if not normalized_document_id:
        raise ValueError(
            "document_id cannot be empty."
        )

    registry = (
        st.session_state.uploaded_documents
    )

    document = registry.get(
        normalized_document_id
    )

    if document is None:
        raise ValueError(
            (
                "The requested document is not "
                "registered in this frontend session."
            )
        )

    # -----------------------------------------------------
    # SAVE PAGE / FIGURE OF PREVIOUS RECENT DOCUMENT
    # -----------------------------------------------------

    current_document_id = (
        st.session_state.active_document_id
    )

    if (
        current_document_id
        and current_document_id
        in registry
        and current_document_id
        != normalized_document_id
    ):
        registry[
            current_document_id
        ]["active_page"] = (
            st.session_state.active_page
        )

        registry[
            current_document_id
        ]["active_figure"] = (
            st.session_state.active_figure
        )

    # -----------------------------------------------------
    # LOAD NEW RECENT DOCUMENT
    # -----------------------------------------------------

    st.session_state.active_document_id = (
        normalized_document_id
    )

    st.session_state.active_document_name = (
        document.get("name")
    )

    st.session_state.active_document_status = (
        document.get("status")
    )

    st.session_state.active_processing_stage = (
        document.get(
            "processing_stage"
        )
    )

    st.session_state.active_page = (
        document.get(
            "active_page"
        )
    )

    st.session_state.active_figure = (
        document.get(
            "active_figure"
        )
    )

    st.session_state.last_error = None


def update_document_status(
    *,
    document_id: str,
    status: str | None = None,
    processing_stage: str | None = None,
) -> None:
    """
    Update the processing state of exactly one document.

    Other uploaded documents are untouched.
    """

    normalized_document_id = (
        document_id.strip()
    )

    if not normalized_document_id:
        return

    registry = (
        st.session_state.uploaded_documents
    )

    document = registry.get(
        normalized_document_id
    )

    if document is None:
        return

    if status is not None:
        document["status"] = status

    if processing_stage is not None:
        document[
            "processing_stage"
        ] = processing_stage

    if (
        st.session_state.active_document_id
        == normalized_document_id
    ):
        if status is not None:
            st.session_state.active_document_status = (
                status
            )

        if processing_stage is not None:
            st.session_state.active_processing_stage = (
                processing_stage
            )


def update_document_context(
    *,
    document_id: str,
    page: int | None = None,
    figure_id: str | None = None,
) -> None:
    """
    Remember page / figure context separately for each
    uploaded document.

    Example:

        Doc A -> page 4
        Doc B -> page 2

    Switching between A and B does not mix those page
    contexts.
    """

    normalized_document_id = (
        document_id.strip()
    )

    if not normalized_document_id:
        return

    registry = (
        st.session_state.uploaded_documents
    )

    document = registry.get(
        normalized_document_id
    )

    if document is None:
        return

    if page is not None:
        if page < 1:
            raise ValueError(
                "page must be at least 1."
            )

        document[
            "active_page"
        ] = page

    if figure_id is not None:
        normalized_figure_id = (
            figure_id.strip()
        )

        document[
            "active_figure"
        ] = (
            normalized_figure_id
            or None
        )

    if (
        st.session_state.active_document_id
        == normalized_document_id
    ):
        if page is not None:
            st.session_state.active_page = (
                page
            )

        if figure_id is not None:
            st.session_state.active_figure = (
                document.get(
                    "active_figure"
                )
            )


def get_session_documents() -> list[
    dict[str, object]
]:
    """
    Return ALL documents belonging to the current frontend
    session.

    Useful for displaying:

        Documents in this session:
        - A
        - B
        - C
    """

    return [
        dict(document)
        for document
        in (
            st.session_state
            .uploaded_documents
            .values()
        )
    ]


def get_available_documents_for_chat() -> list[
    dict[str, str]
]:
    """
    Return the lightweight document list required by
    ChatRequest.available_documents.

    Only document_id + name are sent to the backend.

    Actual PDF/image contents are NOT sent here.
    """

    available: list[
        dict[str, str]
    ] = []

    for document in (
        st.session_state
        .uploaded_documents
        .values()
    ):
        document_id = str(
            document.get(
                "document_id",
                "",
            )
        ).strip()

        name = str(
            document.get(
                "name",
                "",
            )
        ).strip()

        if (
            not document_id
            or not name
        ):
            continue

        available.append(
            {
                "document_id": (
                    document_id
                ),
                "name": name,
            }
        )

    return available


def get_ready_documents_for_chat() -> list[
    dict[str, str]
]:
    """
    Return only documents whose processing status is READY.

    This is useful when we want the automatic resolver to
    consider only documents that already have usable indexes.

    Documents that are still processing remain visible in the
    session bookshelf; they are simply not retrieval-ready yet.
    """

    available: list[
        dict[str, str]
    ] = []

    for document in (
        st.session_state
        .uploaded_documents
        .values()
    ):
        status = str(
            document.get(
                "status",
                "",
            )
            or ""
        ).strip().upper()

        if status != "READY":
            continue

        document_id = str(
            document.get(
                "document_id",
                "",
            )
        ).strip()

        name = str(
            document.get(
                "name",
                "",
            )
        ).strip()

        if (
            not document_id
            or not name
        ):
            continue

        available.append(
            {
                "document_id": (
                    document_id
                ),
                "name": name,
            }
        )

    return available


def remove_document(
    document_id: str,
) -> None:
    """
    Remove exactly one document from this frontend session.

    Backend deletion is handled separately by APIClient.

    Other documents remain available.
    """

    normalized_document_id = (
        document_id.strip()
    )

    if not normalized_document_id:
        return

    registry = (
        st.session_state.uploaded_documents
    )

    was_active = (
        st.session_state.active_document_id
        == normalized_document_id
    )

    registry.pop(
        normalized_document_id,
        None,
    )

    if not was_active:
        return

    # -----------------------------------------------------
    # IF OTHER DOCUMENTS REMAIN
    #
    # Automatically use the most recently registered one as
    # the conversational fallback.
    #
    # No manual switcher is required.
    # -----------------------------------------------------

    if registry:
        most_recent_document_id = (
            next(
                reversed(registry)
            )
        )

        # Clear stale current-document fields first.
        _clear_active_document_fields()

        set_active_document(
            most_recent_document_id
        )

        return

    _clear_active_document_fields()


def _clear_active_document_fields() -> None:
    """
    Clear only the recent/current document pointer.

    Does NOT touch the document bookshelf.
    """

    st.session_state.active_document_id = None
    st.session_state.active_document_name = None
    st.session_state.active_document_status = None
    st.session_state.active_processing_stage = None

    st.session_state.active_page = None
    st.session_state.active_figure = None

    st.session_state.is_processing = False


def reset_document_context() -> None:
    """
    Clear the current contextual document while preserving
    every document registered in THIS session.

    This is different from start_new_session().
    """

    _clear_active_document_fields()

    st.session_state.is_uploading = False

    st.session_state.last_error = None
    st.session_state.last_notice = None



def restore_recovered_session(
    payload: dict[str, object],
) -> None:
    """
    Restore one Redis-backed session returned by the recovery API.

    Important:
    - this only rebuilds Streamlit state;
    - Redis remains the source of short-term conversation context;
    - recovered assistant messages are plain text because the current
      MemorySnapshot stores bounded conversation text, not the original
      full TutorAnswer/citation payload;
    - document references are restored as READY because only lightweight
      references that reached chat memory are being recovered here.
    """

    if not isinstance(payload, dict):
        raise ValueError(
            "Recovered session payload must be an object."
        )

    raw_session_id = payload.get(
        "session_id"
    )

    session_id = (
        str(raw_session_id).strip()
        if raw_session_id is not None
        else ""
    )

    if not session_id:
        raise ValueError(
            "Recovered session is missing session_id."
        )

    # -----------------------------------------------------
    # REBUILD THE VISIBLE CHAT HISTORY
    # -----------------------------------------------------
    #
    # Redis intentionally stores only the bounded recent text
    # conversation. It does not store the original rich TutorAnswer
    # object with citations/expanders. Convert that saved text into the
    # existing frontend message shape so old turns can render normally.
    # -----------------------------------------------------

    recovered_messages: list[
        dict[str, object]
    ] = []

    raw_messages = payload.get(
        "messages",
        [],
    )

    if isinstance(raw_messages, list):
        for raw_message in raw_messages:
            if not isinstance(
                raw_message,
                dict,
            ):
                continue

            role = str(
                raw_message.get(
                    "role",
                    "",
                )
                or ""
            ).strip().casefold()

            content = str(
                raw_message.get(
                    "content",
                    "",
                )
                or ""
            ).strip()

            if (
                role not in {
                    "user",
                    "assistant",
                }
                or not content
            ):
                continue

            if role == "user":
                recovered_messages.append(
                    {
                        "role": "user",
                        "content": content,
                    }
                )
                continue

            recovered_messages.append(
                {
                    "role": "assistant",
                    "answer": {
                        "direct_answer": content,
                    },
                    "resolved_document_id": None,
                    "resolved_document_name": None,
                    "selected_model": None,
                    "recovered_from_session_memory": True,
                }
            )

    # -----------------------------------------------------
    # REBUILD THE DOCUMENT BOOKSHELF
    # -----------------------------------------------------

    recovered_documents: dict[
        str,
        dict[str, object],
    ] = {}

    raw_documents = payload.get(
        "documents",
        [],
    )

    if isinstance(raw_documents, list):
        for raw_document in raw_documents:
            if not isinstance(
                raw_document,
                dict,
            ):
                continue

            document_id = str(
                raw_document.get(
                    "document_id",
                    "",
                )
                or ""
            ).strip()

            name = str(
                raw_document.get(
                    "name",
                    "",
                )
                or ""
            ).strip()

            if (
                not document_id
                or not name
            ):
                continue

            recovered_documents[
                document_id
            ] = {
                "document_id": document_id,
                "name": name,
                "status": "READY",
                "processing_stage": "READY",
                "active_page": None,
                "active_figure": None,
            }

    raw_active_document_id = (
        payload.get(
            "active_document_id"
        )
    )

    active_document_id = (
        str(
            raw_active_document_id
        ).strip()
        if raw_active_document_id
        is not None
        else ""
    )

    raw_active_page = payload.get(
        "active_page"
    )

    active_page = (
        raw_active_page
        if isinstance(
            raw_active_page,
            int,
        )
        and raw_active_page >= 1
        else None
    )

    raw_active_figure = payload.get(
        "active_figure"
    )

    active_figure = (
        str(
            raw_active_figure
        ).strip()
        if raw_active_figure
        is not None
        else ""
    )

    active_figure = (
        active_figure
        or None
    )

    if (
        active_document_id
        and active_document_id
        in recovered_documents
    ):
        recovered_documents[
            active_document_id
        ]["active_page"] = (
            active_page
        )

        recovered_documents[
            active_document_id
        ]["active_figure"] = (
            active_figure
        )

    # -----------------------------------------------------
    # ATOMICALLY SWITCH THE FRONTEND TO THE RECOVERED CHAT
    # -----------------------------------------------------

    st.session_state.session_id = (
        session_id
    )

    st.session_state.messages = (
        recovered_messages
    )

    st.session_state.uploaded_documents = (
        recovered_documents
    )

    _clear_active_document_fields()

    if (
        active_document_id
        and active_document_id
        in recovered_documents
    ):
        set_active_document(
            active_document_id
        )

    raw_language = payload.get(
        "language"
    )

    recovered_language = (
        str(raw_language).strip()
        if raw_language is not None
        else ""
    )

    if (
        recovered_language
        and recovered_language.casefold()
        != "unknown"
    ):
        st.session_state.language = (
            recovered_language
        )

    raw_grade = payload.get(
        "grade"
    )

    st.session_state.grade = (
        raw_grade
        if isinstance(
            raw_grade,
            int,
        )
        and 1 <= raw_grade <= 12
        else None
    )

    st.session_state.is_uploading = False
    st.session_state.is_processing = False
    st.session_state.is_chatting = False

    st.session_state.last_error = None
    st.session_state.last_notice = (
        "Previous chat restored."
    )

def clear_conversation() -> None:
    """
    Clear visible chat messages while keeping the SAME
    session and SAME document bookshelf.

    Example:

        before:
        session S1 -> A, B, C + messages

        after:
        session S1 -> A, B, C + no messages
    """

    st.session_state.messages = []

    st.session_state.last_error = None
    st.session_state.last_notice = None


def start_new_session() -> None:
    """
    Start a completely fresh chat session.

    IMPORTANT:

        old session:
            S1 -> Doc A, Doc B, Doc C

        Start new session

        new session:
            S2 -> empty bookshelf

    The physical backend files are NOT deleted here.
    This only creates a new conversation/session context.

    Actual document deletion happens only through the
    document delete API.
    """

    st.session_state.session_id = str(
        uuid4()
    )

    st.session_state.messages = []

    # -----------------------------------------------------
    # NEW SESSION = NEW DOCUMENT BOOKSHELF
    # -----------------------------------------------------

    st.session_state.uploaded_documents = {}

    _clear_active_document_fields()

    st.session_state.is_uploading = False
    st.session_state.is_chatting = False

    st.session_state.last_error = None
    st.session_state.last_notice = None