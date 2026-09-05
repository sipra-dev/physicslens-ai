from __future__ import annotations

from typing import Any

import streamlit as st

from api_client import (
    APIClientError,
    PhyMentorAPIClient,
)
from session_state import (
    get_session_documents,
    register_document,
    remove_document,
    update_document_status,
)


ALLOWED_UPLOAD_TYPES = [
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "webp",
]


def _save_upload_result(
    result: dict[str, Any],
    *,
    fallback_filename: str,
) -> None:
    """
    Register the uploaded document in this chat session.

    IMPORTANT:

    Uploading a new document does NOT replace older
    documents in the session bookshelf.

    Example:

        existing:
            Doc A
            Doc B

        upload Doc C

        result:
            Doc A
            Doc B
            Doc C
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
                "The backend upload response "
                "did not contain a valid document_id."
            )
        )

    original_filename = str(
        result.get(
            "original_filename",
            "",
        )
        or fallback_filename
    ).strip()

    if not original_filename:
        original_filename = (
            f"Document {document_id[:8]}"
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

    register_document(
        document_id=document_id,
        name=original_filename,
        status=(
            status
            or None
        ),
        processing_stage=(
            processing_stage
            or None
        ),
        make_active=True,
    )

    normalized_status = (
        status.upper()
    )

    st.session_state.is_processing = (
        normalized_status
        not in {
            "READY",
            "REJECTED",
            "FAILED",
        }
    )

    st.session_state.last_notice = (
        result.get("message")
        or (
            f"{original_filename} was "
            "added to this session."
        )
    )

    st.session_state.last_error = None


def _render_document_actions(
    *,
    api_client: PhyMentorAPIClient,
    document: dict[str, object],
) -> None:
    """
    Render reindex/delete controls for one document.

    These controls do not manually switch the active document.
    They only manage the backend copy of the specific document.
    """

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

    status = str(
        document.get(
            "status",
            "",
        )
        or ""
    ).strip().upper()

    processing_stage = str(
        document.get(
            "processing_stage",
            "",
        )
        or ""
    ).strip().upper()

    if not document_id:
        return

    busy = (
        status
        in {
            "PROCESSING",
            "INDEXING",
        }
        or processing_stage
        in {
            "QUEUED",
            "PARSING",
            "LAYOUT_ANALYSIS",
            "OCR",
            "EQUATION_EXTRACTION",
            "FIGURE_EXTRACTION",
            "SCOPE_CLASSIFICATION",
            "CHUNKING",
            "FAISS_BM25_INDEXING",
        }
    )

    reindex_col, delete_col = st.columns(2)

    with reindex_col:
        reindex_clicked = st.button(
            "Reindex",
            key=f"reindex_{document_id}",
            use_container_width=True,
            disabled=busy,
        )

    with delete_col:
        delete_clicked = st.button(
            "Delete",
            key=f"delete_{document_id}",
            use_container_width=True,
        )

    if reindex_clicked:
        st.session_state.last_error = None
        st.session_state.last_notice = None

        try:
            with st.spinner(
                f"Reindexing {name}..."
            ):
                result = (
                    api_client.reindex_document(
                        user_id=(
                            st.session_state.user_id
                        ),
                        document_id=document_id,
                    )
                )

            update_document_status(
                document_id=document_id,
                status=(
                    str(
                        result.get(
                            "status",
                            "",
                        )
                        or ""
                    ).strip()
                    or None
                ),
                processing_stage=(
                    str(
                        result.get(
                            "processing_stage",
                            "",
                        )
                        or ""
                    ).strip()
                    or None
                ),
            )

            st.session_state.last_notice = (
                result.get("message")
                or (
                    f"{name} was queued "
                    "for reindexing."
                )
            )

            st.rerun()

        except (
            APIClientError,
            ValueError,
        ) as exc:
            st.session_state.last_error = (
                str(exc)
            )

            st.error(
                f"Reindex failed: {exc}"
            )

        except Exception:
            st.session_state.last_error = (
                "An unexpected frontend error occurred."
            )

            st.error(
                "The document could not be reindexed."
            )

    if delete_clicked:
        st.session_state[
            "pending_delete_document_id"
        ] = document_id
        st.rerun()

    pending_delete_document_id = (
        st.session_state.get(
            "pending_delete_document_id"
        )
    )

    if (
        pending_delete_document_id
        != document_id
    ):
        return

    st.warning(
        (
            f"Delete {name}? This removes the "
            "document and its local indexed artifacts."
        )
    )

    confirm_col, cancel_col = st.columns(2)

    with confirm_col:
        confirm_delete = st.button(
            "Confirm delete",
            key=f"confirm_delete_{document_id}",
            type="primary",
            use_container_width=True,
        )

    with cancel_col:
        cancel_delete = st.button(
            "Cancel",
            key=f"cancel_delete_{document_id}",
            use_container_width=True,
        )

    if cancel_delete:
        st.session_state.pop(
            "pending_delete_document_id",
            None,
        )
        st.rerun()

    if not confirm_delete:
        return

    st.session_state.last_error = None
    st.session_state.last_notice = None

    try:
        with st.spinner(
            f"Deleting {name}..."
        ):
            result = (
                api_client.delete_document(
                    user_id=(
                        st.session_state.user_id
                    ),
                    document_id=document_id,
                )
            )

        remove_document(
            document_id
        )

        st.session_state.pop(
            "pending_delete_document_id",
            None,
        )

        st.session_state.last_notice = (
            result.get("message")
            or f"{name} was deleted."
        )

        st.rerun()

    except (
        APIClientError,
        ValueError,
    ) as exc:
        st.session_state.last_error = (
            str(exc)
        )

        st.error(
            f"Delete failed: {exc}"
        )

    except Exception:
        st.session_state.last_error = (
            "An unexpected frontend error occurred."
        )

        st.error(
            "The document could not be deleted."
        )


def _render_session_documents(
    *,
    api_client: PhyMentorAPIClient,
) -> None:
    """
    Show the documents currently remembered by this
    frontend chat session.

    There is deliberately NO manual document switcher.

    Reindex/delete controls manage a document without
    changing the backend resolver's automatic-selection policy.
    """

    documents = (
        get_session_documents()
    )

    if not documents:
        st.caption(
            "No documents in this session yet."
        )
        return

    st.markdown(
        "**Documents in this session**"
    )

    for document in documents:

        name = str(
            document.get(
                "name",
                "Document",
            )
        )

        status = str(
            document.get(
                "status",
                "",
            )
            or ""
        ).strip().upper()

        processing_stage = str(
            document.get(
                "processing_stage",
                "",
            )
            or ""
        ).strip()

        if status:
            label = (
                f"• {name} — {status}"
            )
        else:
            label = (
                f"• {name}"
            )

        if (
            processing_stage
            and status
            not in {
                "READY",
                "REJECTED",
                "FAILED",
            }
        ):
            label += (
                f" ({processing_stage})"
            )

        st.caption(label)

        _render_document_actions(
            api_client=api_client,
            document=document,
        )


def render_upload_panel(
    *,
    api_client: PhyMentorAPIClient,
) -> None:
    """
    Render the Physics document uploader.

    Documents are sent only to FastAPI.

    Streamlit remembers lightweight document identity
    information for the current chat session, while all
    parsing/indexing/retrieval remains in the backend.

    There is no manual document selector.

    Later chat questions are automatically matched to the
    appropriate uploaded document by the backend resolver.
    """

    st.subheader(
        "Upload Physics material"
    )

    _render_session_documents(
        api_client=api_client
    )

    uploaded_file = st.file_uploader(
        (
            "Upload another Physics PDF, page, "
            "diagram, or numerical problem"
        ),
        type=ALLOWED_UPLOAD_TYPES,
        accept_multiple_files=False,
        help=(
            "Supported formats: PDF, PNG, JPG, "
            "JPEG and WEBP. Previously uploaded "
            "documents in this session are preserved."
        ),
    )

    if uploaded_file is None:

        if not get_session_documents():
            st.caption(
                (
                    "Choose a Class 1–12 Physics "
                    "file to begin."
                )
            )
        else:
            st.caption(
                (
                    "You can keep chatting or upload "
                    "another Physics document."
                )
            )

        return

    st.caption(
        f"Selected: {uploaded_file.name}"
    )

    upload_clicked = st.button(
        "Upload and process",
        type="primary",
        use_container_width=True,
        disabled=(
            st.session_state.is_uploading
        ),
    )

    if not upload_clicked:
        return

    st.session_state.is_uploading = True
    st.session_state.last_error = None
    st.session_state.last_notice = None

    try:

        with st.spinner(
            "Sending the document to PhyMentor..."
        ):
            result = (
                api_client.upload_document(
                    user_id=(
                        st.session_state.user_id
                    ),
                    filename=(
                        uploaded_file.name
                    ),
                    content_type=(
                        uploaded_file.type
                        or (
                            "application/"
                            "octet-stream"
                        )
                    ),
                    file_bytes=(
                        uploaded_file.getvalue()
                    ),
                )
            )

        _save_upload_result(
            result,
            fallback_filename=(
                uploaded_file.name
            ),
        )

        st.session_state.last_notice = (
            result.get(
                "message"
            )
            or (
                f"{uploaded_file.name} "
                "was added successfully."
            )
        )

        st.rerun()

    except (
        APIClientError,
        ValueError,
    ) as exc:

        st.session_state.last_error = (
            str(exc)
        )

        st.error(
            f"Upload failed: {exc}"
        )

    except Exception as exc:
        st.session_state.last_error = (
            (
                "An unexpected frontend "
                "error occurred."
            )
        )

        st.error(
            (
                "The upload could not "
                "be completed."
            )
        )

        st.exception(exc)

    finally:
        st.session_state.is_uploading = False
