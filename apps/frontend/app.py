from __future__ import annotations

import streamlit as st

from api_client import (
    APIClientError,
    PhyMentorAPIClient,
)
from components.chat_panel import (
    render_chat_panel,
)
from components.status_panel import (
    render_document_status,
)
from components.upload_panel import (
    render_upload_panel,
)
from session_state import (
    clear_conversation,
    get_session_documents,
    initialize_session_state,
    restore_recovered_session,
    start_new_session,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PhyMentor AI",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# FRONTEND SESSION
# =========================================================

initialize_session_state()

# A recovered chat is staged by the Resume button and applied only on the
# following Streamlit rerun. This must happen before widgets such as
# key="language" and key="grade" are instantiated; Streamlit does not allow
# widget-owned session-state values to be changed later in the same run.
pending_recovered_session = st.session_state.pop(
    "_pending_recovered_session",
    None,
)

if isinstance(
    pending_recovered_session,
    dict,
):
    try:
        restore_recovered_session(
            pending_recovered_session
        )
    except ValueError as exc:
        st.session_state[
            "_pending_recovery_error"
        ] = str(exc)


# =========================================================
# SHARED API CLIENT
# =========================================================

@st.cache_resource
def get_api_client() -> (
    PhyMentorAPIClient
):
    """
    Create one reusable HTTP client for this
    Streamlit application process.
    """

    return PhyMentorAPIClient()


api_client = get_api_client()

pending_recovery_error = st.session_state.pop(
    "_pending_recovery_error",
    None,
)

if pending_recovery_error:
    st.error(
        str(pending_recovery_error)
    )


# =========================================================
# PREVIOUS-CHAT HELPERS
# =========================================================

def _format_recovery_ttl(
    ttl_seconds: object,
) -> str:
    """
    Format the remaining Redis recovery lifetime for display.
    """

    try:
        total_seconds = max(
            0,
            int(ttl_seconds),
        )
    except (
        TypeError,
        ValueError,
    ):
        return "Recovery window active"

    hours, remainder = divmod(
        total_seconds,
        3600,
    )

    minutes = remainder // 60

    if hours:
        return (
            f"Available for about "
            f"{hours}h {minutes}m"
        )

    if minutes:
        return (
            f"Available for about "
            f"{minutes}m"
        )

    return "Expiring soon"


def _render_previous_chats(
    *,
    api_client: PhyMentorAPIClient,
) -> None:
    """
    Show Redis-backed chats that are still recoverable.

    Recovery stays behind FastAPI:
    Streamlit never talks to Redis directly.
    """

    st.subheader(
        "Previous chats"
    )

    st.caption(
        (
            "Recent chats can be resumed while "
            "their short-term session memory is "
            "still available (up to about 24 hours)."
        )
    )

    try:
        payload = (
            api_client
            .list_recoverable_sessions(
                user_id=(
                    st.session_state
                    .user_id
                ),
            )
        )

    except APIClientError:
        st.caption(
            (
                "Previous chats are temporarily "
                "unavailable."
            )
        )
        return

    raw_sessions = payload.get(
        "sessions",
        [],
    )

    if not isinstance(
        raw_sessions,
        list,
    ):
        raw_sessions = []

    current_session_id = str(
        st.session_state.session_id
        or ""
    ).strip()

    previous_sessions: list[
        dict[str, object]
    ] = []

    for raw_session in raw_sessions:

        if not isinstance(
            raw_session,
            dict,
        ):
            continue

        session_id = str(
            raw_session.get(
                "session_id",
                "",
            )
            or ""
        ).strip()

        # Do not show the chat that is already open.
        if (
            session_id
            and session_id
            == current_session_id
        ):
            continue

        session_reference = str(
            raw_session.get(
                "session_reference",
                "",
            )
            or ""
        ).strip()

        if not session_reference:
            continue

        previous_sessions.append(
            raw_session
        )

    if not previous_sessions:

        st.caption(
            "No other recoverable chats."
        )
        return

    for index, session in enumerate(
        previous_sessions
    ):

        session_reference = str(
            session.get(
                "session_reference",
                "",
            )
            or ""
        ).strip()

        preview = str(
            session.get(
                "preview",
                "",
            )
            or ""
        ).strip()

        if not preview:
            preview = (
                "Previous Physics chat"
            )

        if len(preview) > 90:
            preview = (
                preview[:87].rstrip()
                + "..."
            )

        raw_document_names = (
            session.get(
                "document_names",
                [],
            )
        )

        document_names = (
            [
                str(name).strip()
                for name
                in raw_document_names
                if str(name).strip()
            ]
            if isinstance(
                raw_document_names,
                list,
            )
            else []
        )

        message_count_raw = (
            session.get(
                "message_count",
                0,
            )
        )

        try:
            message_count = max(
                0,
                int(
                    message_count_raw
                ),
            )
        except (
            TypeError,
            ValueError,
        ):
            message_count = 0

        legacy = bool(
            session.get(
                "legacy",
                False,
            )
        )

        with st.container(
            border=True
        ):

            st.markdown(
                f"**{preview}**"
            )

            if document_names:

                document_text = ", ".join(
                    document_names[:3]
                )

                if len(
                    document_names
                ) > 3:
                    document_text += (
                        f" +{len(document_names) - 3} more"
                    )

                st.caption(
                    (
                        "Documents: "
                        f"{document_text}"
                    )
                )

            st.caption(
                (
                    f"{message_count} saved message"
                    + (
                        ""
                        if message_count == 1
                        else "s"
                    )
                    + " · "
                    + _format_recovery_ttl(
                        session.get(
                            "ttl_seconds",
                            0,
                        )
                    )
                )
            )

            if legacy:
                st.caption(
                    (
                        "Older session: it will be "
                        "safely migrated to a new "
                        "recoverable session ID."
                    )
                )

            if st.button(
                "Resume chat",
                key=(
                    "resume_previous_chat_"
                    f"{index}"
                ),
                use_container_width=True,
            ):

                try:
                    recovered = (
                        api_client
                        .recover_chat_session(
                            user_id=(
                                st.session_state
                                .user_id
                            ),
                            session_reference=(
                                session_reference
                            ),
                        )
                    )

                    # Do not restore immediately here: the Language and Grade
                    # widgets have already been instantiated in this run.
                    # Stage the payload and restore it at the very top of the
                    # next rerun, before any widget-owned state is created.
                    st.session_state[
                        "_pending_recovered_session"
                    ] = recovered

                except (
                    APIClientError,
                    ValueError,
                ) as exc:
                    st.error(
                        str(exc)
                    )
                    return

                st.rerun()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "PhyMentor AI"
    )

    st.caption(
        (
            "Multimodal school-level "
            "Physics tutor"
        )
    )

    st.divider()

    # -----------------------------------------------------
    # SESSION
    # -----------------------------------------------------

    st.subheader(
        "Session"
    )

    st.text_input(
        "User ID",
        key="user_id",
        help=(
            "Local-development identity. "
            "Authentication will replace this "
            "in the production security phase."
        ),
    )

    st.text_input(
        "Session ID",
        value=(
            st.session_state.session_id
        ),
        disabled=True,
    )

    documents = (
        get_session_documents()
    )

    document_count = len(
        documents
    )

    if document_count == 0:

        st.caption(
            "No documents in this session."
        )

    elif document_count == 1:

        st.caption(
            "1 document in this session."
        )

    else:

        st.caption(
            (
                f"{document_count} documents "
                "in this session."
            )
        )

    st.divider()

    # -----------------------------------------------------
    # LEARNING CONTEXT
    # -----------------------------------------------------

    st.subheader(
        "Learning context"
    )

    st.selectbox(
        "Language",
        options=[
            "unknown",
            "en",
            "bn",
            "hi",
        ],
        key="language",
    )

    st.number_input(
        "Grade",
        min_value=1,
        max_value=12,
        value=None,
        step=1,
        key="grade",
        placeholder="Optional",
    )

    st.divider()

    # -----------------------------------------------------
    # CONVERSATION CONTROLS
    # -----------------------------------------------------

    if st.button(
        "Clear conversation",
        use_container_width=True,
        help=(
            "Clears visible chat messages but "
            "keeps the documents belonging to "
            "this session."
        ),
    ):
        clear_conversation()
        st.rerun()

    if st.button(
        "Start new session",
        use_container_width=True,
        help=(
            "Starts a fresh chat session with "
            "an empty document bookshelf. "
            "Previously uploaded backend files "
            "are not physically deleted."
        ),
    ):
        start_new_session()
        st.rerun()

    st.divider()

    # -----------------------------------------------------
    # PREVIOUS CHATS
    # -----------------------------------------------------

    _render_previous_chats(
        api_client=api_client,
    )


# =========================================================
# MAIN PAGE
# =========================================================

st.title(
    "⚛️ PhyMentor AI"
)

st.write(
    (
        "Ask school-level Physics questions "
        "or upload Physics PDFs, pages, "
        "diagrams, and numerical problems."
    )
)

st.caption(
    (
        "PhyMentor can answer both general "
        "school-level Physics questions and "
        "questions based on your uploaded "
        "Physics materials."
    )
)

st.warning(
    (
        "📌 Important for document-based questions\n\n"
        "If you want an answer specifically from an uploaded file, "
        "always mention **“from the document”** in your question.\n\n"
        "If you have uploaded more than one document, mention the "
        "**document name or topic** as well so PhyMentor can safely "
        "use the correct source.\n\n"
        "For the safest multi-document use, keep **no more than "
        "3 documents in one session**.\n\n"
        "**Example:** “Explain the displacement equation from the SHM document.”"
    )
)

st.divider()


# =========================================================
# SESSION DOCUMENT SUMMARY
# =========================================================

session_documents = (
    get_session_documents()
)

ready_count = 0
processing_count = 0
failed_count = 0

for document in session_documents:

    status = str(
        document.get(
            "status",
            "",
        )
        or ""
    ).strip().upper()

    if status == "READY":
        ready_count += 1

    elif status in {
        "FAILED",
        "REJECTED",
    }:
        failed_count += 1

    else:
        processing_count += 1


if session_documents:

    st.subheader(
        "Session documents"
    )

    summary_columns = st.columns(
        3
    )

    with summary_columns[0]:
        st.metric(
            "Total",
            len(
                session_documents
            ),
        )

    with summary_columns[1]:
        st.metric(
            "Ready",
            ready_count,
        )

    with summary_columns[2]:
        st.metric(
            "Processing",
            processing_count,
        )

    if failed_count:

        st.caption(
            (
                f"{failed_count} document"
                + (
                    ""
                    if failed_count == 1
                    else "s"
                )
                + (
                    " failed or were rejected."
                )
            )
        )

    if ready_count:

        if len(
            session_documents
        ) == 1:

            st.info(
                (
                    "No manual document selector is needed. "
                    "For a document-grounded answer, mention "
                    "“from the document” in your question."
                )
            )

        else:

            st.info(
                (
                    "No manual document selector is needed. "
                    "For a document-grounded answer, mention "
                    "“from the document” and include the "
                    "document name or topic in your question."
                )
            )

else:

    st.info(
        (
            "No Physics documents have been "
            "uploaded in this session yet. "
            "You can still ask general "
            "school-level Physics questions."
        )
    )


st.divider()


# =========================================================
# MAIN WORKSPACE
# =========================================================

upload_column, chat_column = (
    st.columns(
        [1, 2]
    )
)


# =========================================================
# DOCUMENT UPLOAD + STATUS
# =========================================================

with upload_column:

    render_upload_panel(
        api_client=api_client,
    )

    render_document_status(
        api_client=api_client,
    )


# =========================================================
# PHYSICS TUTOR CHAT
# =========================================================

with chat_column:

    render_chat_panel(
        api_client=api_client,
    )
