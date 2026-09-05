from __future__ import annotations

from typing import Any

import streamlit as st

from api_client import (
    APIClientError,
    PhyMentorAPIClient,
)
from session_state import (
    get_session_documents,
    update_document_status,
)


TERMINAL_DOCUMENT_STATUSES = {
    "READY",
    "REJECTED",
    "FAILED",
}


def _normalized_status(
    value: Any,
) -> str:
    """
    Convert any backend/frontend status value
    into one clean uppercase string.
    """

    return str(
        value
        or ""
    ).strip().upper()


def _save_document_status(
    result: dict[str, Any],
) -> None:
    """
    Update the status of exactly ONE document
    inside the multi-document session registry.

    Other documents remain untouched.
    """

    document_id = str(
        result.get(
            "document_id",
            "",
        )
        or ""
    ).strip()

    if not document_id:
        raise ValueError(
            (
                "The backend status response "
                "did not contain a valid document_id."
            )
        )

    status = str(
        result.get(
            "status",
            "",
        )
        or ""
    ).strip()

    processing_stage = str(
        result.get(
            "processing_stage",
            "",
        )
        or ""
    ).strip()

    update_document_status(
        document_id=document_id,
        status=(
            status
            or None
        ),
        processing_stage=(
            processing_stage
            or None
        ),
    )

    message = result.get(
        "message"
    )

    processing_error = result.get(
        "processing_error"
    )

    if message:
        st.session_state.last_notice = (
            str(message)
        )

    if processing_error:
        st.session_state.last_error = (
            str(processing_error)
        )


def _recompute_processing_flag() -> None:
    """
    Recalculate whether ANY document in this
    session is still being processed.

    Example:

        A = READY
        B = PROCESSING
        C = READY

    is_processing -> True

    When B becomes READY:

    is_processing -> False
    """

    documents = (
        get_session_documents()
    )

    processing_exists = False

    for document in documents:

        status = _normalized_status(
            document.get(
                "status"
            )
        )

        if (
            status
            not in TERMINAL_DOCUMENT_STATUSES
        ):
            processing_exists = True
            break

    st.session_state.is_processing = (
        processing_exists
    )


def _render_document_status(
    *,
    document: dict[str, Any],
) -> None:
    """
    Render the status of one document.

    This is display-only.
    It does NOT switch the active document.
    """

    name = str(
        document.get(
            "name",
            "Document",
        )
        or "Document"
    ).strip()

    status = _normalized_status(
        document.get(
            "status"
        )
    )

    processing_stage = str(
        document.get(
            "processing_stage",
            "",
        )
        or ""
    ).strip()

    if status == "READY":

        st.success(
            f"{name} — READY"
        )

    elif status == "REJECTED":

        st.warning(
            f"{name} — REJECTED"
        )

        st.caption(
            (
                "This document was rejected by "
                "PhyMentor's validation rules."
            )
        )

    elif status == "FAILED":

        st.error(
            f"{name} — FAILED"
        )

        st.caption(
            "Document processing failed."
        )

    else:

        st.info(
            (
                f"{name} — "
                f"{status or 'PROCESSING'}"
            )
        )

        if processing_stage:
            st.caption(
                (
                    "Current stage: "
                    f"{processing_stage}"
                )
            )


def _refresh_all_document_statuses(
    *,
    api_client: PhyMentorAPIClient,
) -> int:
    """
    Ask FastAPI for the latest status of every
    document currently registered in this session.

    Returns the number of successfully refreshed
    documents.
    """

    documents = (
        get_session_documents()
    )

    refreshed_count = 0
    errors: list[str] = []

    for document in documents:

        document_id = str(
            document.get(
                "document_id",
                "",
            )
            or ""
        ).strip()

        name = str(
            document.get(
                "name",
                "Document",
            )
            or "Document"
        ).strip()

        if not document_id:
            continue

        try:

            result = (
                api_client
                .get_document_status(
                    user_id=(
                        st.session_state
                        .user_id
                    ),
                    document_id=(
                        document_id
                    ),
                )
            )

            _save_document_status(
                result
            )

            refreshed_count += 1

        except (
            APIClientError,
            ValueError,
        ) as exc:

            errors.append(
                f"{name}: {exc}"
            )

        except Exception:

            errors.append(
                (
                    f"{name}: could not "
                    "refresh status."
                )
            )

    _recompute_processing_flag()

    if errors:

        st.session_state.last_error = (
            " | ".join(
                errors
            )
        )

    else:

        st.session_state.last_error = None

    return refreshed_count


def render_document_status(
    *,
    api_client: PhyMentorAPIClient,
) -> None:
    """
    Display processing state for ALL documents
    in the current chat session.

    There is deliberately no manual document
    selector here.

    Example:

        nuclear_fission.jpg — READY
        shm.pdf             — PROCESSING
        optics.png          — READY

    When SHM becomes READY, it automatically
    becomes eligible for document resolution
    during chat.
    """

    documents = (
        get_session_documents()
    )

    if not documents:
        return

    st.subheader(
        "Document status"
    )

    # -----------------------------------------------------
    # SHOW STATUS OF EVERY DOCUMENT IN THIS SESSION
    # -----------------------------------------------------

    for document in documents:
        _render_document_status(
            document=document
        )

    # -----------------------------------------------------
    # ONE BUTTON REFRESHES THE WHOLE BOOKSHELF
    #
    # This is NOT a document switcher.
    # -----------------------------------------------------

    refresh_clicked = st.button(
        "Refresh document statuses",
        use_container_width=True,
    )

    if not refresh_clicked:
        return

    try:

        with st.spinner(
            "Checking document statuses..."
        ):

            refreshed_count = (
                _refresh_all_document_statuses(
                    api_client=api_client
                )
            )

        st.session_state.last_notice = (
            (
                f"Updated {refreshed_count} "
                "document status"
                + (
                    ""
                    if refreshed_count == 1
                    else "es"
                )
                + "."
            )
        )

        st.rerun()

    except Exception as exc:

        st.session_state.last_error = (
            (
                "An unexpected frontend "
                "error occurred."
            )
        )

        st.error(
            (
                "Could not refresh "
                "document statuses."
            )
        )

        st.exception(
            exc
        )