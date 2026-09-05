from __future__ import annotations

from typing import Any

import streamlit as st

from api_client import (
    APIClientError,
    PhyMentorAPIClient,
)
from session_state import (
    get_ready_documents_for_chat,
    set_active_document,
    update_document_context,
)


_MODEL_OPTIONS: dict[str, str | None] = {
    "Auto": None,
    "GPT-4o": "gpt-4o",
    "GPT-5.6 Sol": "gpt-5.6-sol",
    "GPT-5.6 Terra": "gpt-5.6-terra",
    "GPT-5.6 Luna": "gpt-5.6-luna",
}


# =========================================================
# SELECTED-TEXT "ASK PHYMENTOR" COMPONENT
# =========================================================
#
# Streamlit 1.61.x already supports Custom Components v2.
# The component is mounted INSIDE each assistant chat message.
# Its JavaScript therefore accepts selections only from that
# exact assistant message, not from user messages or the rest
# of the page.
#
# Important security rule:
# LLM/user text is passed only through component `data`.
# It is never interpolated into HTML/CSS/JS source.
# =========================================================

_SELECTION_COMPONENT = (
    st.components.v2.component(
        name=(
            "phymentor_answer_text_selection"
        ),
        html="""
        <div
            class="phymentor-selection-root"
            aria-live="polite"
        >
            <div
                class="phymentor-selection-result"
                hidden
            >
                <div
                    class="phymentor-selection-label"
                ></div>
                <div
                    class="phymentor-selection-explanation"
                ></div>
            </div>
        </div>
        """,
        css="""
        .phymentor-selection-root {
            width: 100%;
            font-family: var(--st-font);
            color: var(--st-text-color);
        }

        .phymentor-selection-result {
            margin-top: 0.45rem;
            padding: 0.7rem 0.8rem;
            border: 1px solid
                color-mix(
                    in srgb,
                    var(--st-text-color) 18%,
                    transparent
                );
            border-radius: 0.7rem;
            background:
                color-mix(
                    in srgb,
                    var(--st-background-color) 94%,
                    var(--st-primary-color) 6%
                );
        }

        .phymentor-selection-label {
            margin-bottom: 0.35rem;
            font-size: 0.8rem;
            font-weight: 600;
            opacity: 0.78;
        }

        .phymentor-selection-explanation {
            font-size: 0.95rem;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        """,
        js=r"""
        export default function({
            parentElement,
            data,
            key,
            setTriggerValue
        }) {
            const root = parentElement.querySelector(
                ".phymentor-selection-root"
            );

            const resultPanel = (
                parentElement.querySelector(
                    ".phymentor-selection-result"
                )
            );

            const resultLabel = (
                parentElement.querySelector(
                    ".phymentor-selection-label"
                )
            );

            const resultExplanation = (
                parentElement.querySelector(
                    ".phymentor-selection-explanation"
                )
            );

            // With isolate_styles=True, Streamlit gives this
            // component a ShadowRoot as parentElement. ShadowRoot does not
            // implement Element.closest(), so resolve the real host element
            // before walking up to the surrounding chat message.
            const componentHost = (
                parentElement instanceof ShadowRoot
                ? parentElement.host
                : parentElement
            );

            const messageRoot = (
                componentHost.closest(
                    '[data-testid="stChatMessage"]'
                )
            );

            if (
                !root
                || !resultPanel
                || !resultLabel
                || !resultExplanation
                || !messageRoot
            ) {
                return;
            }

            // -------------------------------------------------
            // Render the latest Python-side explanation safely.
            //
            // Never inject answer text through innerHTML.
            // -------------------------------------------------

            const response = data?.response ?? null;

            if (
                response
                && typeof response.explanation === "string"
                && response.explanation.trim()
            ) {
                const selectedText = (
                    typeof response.selected_text === "string"
                    ? response.selected_text.trim()
                    : ""
                );

                resultLabel.textContent = selectedText
                    ? `Ask PhyMentor · “${selectedText}”`
                    : "Ask PhyMentor";

                resultExplanation.textContent = (
                    response.explanation
                );

                resultPanel.hidden = false;
            } else {
                resultPanel.hidden = true;
                resultLabel.textContent = "";
                resultExplanation.textContent = "";
            }

            // -------------------------------------------------
            // Floating button shown beside a valid selection.
            // It is attached to document.body so a 1px component
            // wrapper cannot clip it.
            // -------------------------------------------------

            const buttonId = (
                `phymentor-selection-button-${key}`
            );

            const oldButton = document.getElementById(
                buttonId
            );

            if (oldButton) {
                oldButton.remove();
            }

            const button = document.createElement(
                "button"
            );

            button.id = buttonId;
            button.type = "button";
            button.textContent = "Ask PhyMentor";
            button.hidden = true;

            Object.assign(
                button.style,
                {
                    position: "fixed",
                    zIndex: "999999",
                    padding: "7px 11px",
                    border: "0",
                    borderRadius: "999px",
                    background: (
                        "var(--st-primary-color, #ff4b4b)"
                    ),
                    color: "white",
                    fontFamily: "inherit",
                    fontSize: "13px",
                    fontWeight: "600",
                    lineHeight: "1.2",
                    cursor: "pointer",
                    boxShadow: (
                        "0 4px 16px rgba(0,0,0,0.18)"
                    )
                }
            );

            document.body.appendChild(
                button
            );

            let selectedText = "";
            let surroundingText = "";

            function hideButton() {
                button.hidden = true;
                selectedText = "";
                surroundingText = "";
            }

            function nodeBelongsToMessage(
                node
            ) {
                if (!node) {
                    return false;
                }

                const element = (
                    node.nodeType === Node.ELEMENT_NODE
                    ? node
                    : node.parentElement
                );

                return Boolean(
                    element
                    && messageRoot.contains(element)
                );
            }

            function showButtonForSelection() {
                const selection = (
                    window.getSelection()
                );

                if (
                    !selection
                    || selection.rangeCount === 0
                    || selection.isCollapsed
                ) {
                    hideButton();
                    return;
                }

                if (
                    !nodeBelongsToMessage(
                        selection.anchorNode
                    )
                    || !nodeBelongsToMessage(
                        selection.focusNode
                    )
                ) {
                    hideButton();
                    return;
                }

                const text = (
                    selection.toString().trim()
                );

                if (
                    !text
                    || text.length > 2000
                ) {
                    hideButton();
                    return;
                }

                const range = selection.getRangeAt(0);
                const rect = (
                    range.getBoundingClientRect()
                );

                if (
                    !rect
                    || (
                        rect.width === 0
                        && rect.height === 0
                    )
                ) {
                    hideButton();
                    return;
                }

                selectedText = text;

                surroundingText = (
                    messageRoot.innerText
                    || ""
                )
                    .trim()
                    .slice(0, 6000);

                button.hidden = false;

                const horizontalPadding = 8;
                const preferredLeft = (
                    rect.left
                    + (rect.width / 2)
                    - 55
                );

                const left = Math.max(
                    horizontalPadding,
                    Math.min(
                        preferredLeft,
                        window.innerWidth - 125
                    )
                );

                const top = Math.max(
                    8,
                    Math.min(
                        rect.bottom + 8,
                        window.innerHeight - 42
                    )
                );

                button.style.left = `${left}px`;
                button.style.top = `${top}px`;
            }

            function onPointerUp(event) {
                if (
                    event.target === button
                    || button.contains(
                        event.target
                    )
                ) {
                    return;
                }

                window.setTimeout(
                    showButtonForSelection,
                    0
                );
            }

            function onKeyUp() {
                window.setTimeout(
                    showButtonForSelection,
                    0
                );
            }

            function onButtonClick(event) {
                event.preventDefault();
                event.stopPropagation();

                if (!selectedText) {
                    hideButton();
                    return;
                }

                button.textContent = "Opening…";
                button.disabled = true;

                const eventId = (
                    globalThis.crypto?.randomUUID?.()
                    ?? (
                        `${Date.now()}-`
                        + Math.random()
                            .toString(16)
                            .slice(2)
                    )
                );

                setTriggerValue(
                    "ask",
                    {
                        event_id: eventId,
                        selected_text: selectedText,
                        surrounding_text: (
                            surroundingText
                        )
                    }
                );

                const selection = (
                    window.getSelection()
                );

                if (selection) {
                    selection.removeAllRanges();
                }

                button.hidden = true;
            }

            document.addEventListener(
                "mouseup",
                onPointerUp
            );

            document.addEventListener(
                "touchend",
                onPointerUp
            );

            document.addEventListener(
                "keyup",
                onKeyUp
            );

            button.addEventListener(
                "click",
                onButtonClick
            );

            return () => {
                document.removeEventListener(
                    "mouseup",
                    onPointerUp
                );

                document.removeEventListener(
                    "touchend",
                    onPointerUp
                );

                document.removeEventListener(
                    "keyup",
                    onKeyUp
                );

                button.removeEventListener(
                    "click",
                    onButtonClick
                );

                button.remove();
            };
        }
        """,
        isolate_styles=True,
    )
)


_SELECTION_RESULT_STATE_KEY = (
    "_phymentor_selection_results"
)

_SELECTION_EVENT_STATE_KEY = (
    "_phymentor_selection_event_ids"
)


def _document_name(
    document_id: str | None,
) -> str | None:
    """
    Resolve a document ID to the friendly filename stored
    in this Streamlit session.
    """

    if not document_id:
        return None

    document = (
        st.session_state
        .uploaded_documents
        .get(document_id)
    )

    if not isinstance(
        document,
        dict,
    ):
        return None

    name = str(
        document.get(
            "name",
            "",
        )
        or ""
    ).strip()

    return name or None


def _sync_resolved_document_context(
    *,
    response: dict[str, Any],
    answer: dict[str, Any],
) -> tuple[
    str | None,
    str | None,
]:
    """
    After the backend automatically resolves a document,
    make that document the recent conversational document.

    Example:

        recent before question:
            SHM.pdf

        student:
            "Explain the fission diagram again"

        backend resolves:
            nuclear_fission.jpg

        recent after answer:
            nuclear_fission.jpg

    Therefore a following question like:

        "Why does this happen?"

    naturally continues from the fission document.
    """

    resolved_document_id = str(
        response.get(
            "document_id",
            "",
        )
        or ""
    ).strip()

    if not resolved_document_id:
        # General Physics answer.
        #
        # Do NOT erase the previous document context.
        return None, None

    if (
        resolved_document_id
        not in st.session_state.uploaded_documents
    ):
        # Defensive boundary.
        #
        # Do not create a frontend document entry from an
        # unknown backend ID.
        return (
            resolved_document_id,
            None,
        )

    # -----------------------------------------------------
    # BACKEND AUTOMATICALLY CHOSE THIS DOCUMENT
    #
    # This is an INTERNAL context update.
    # The student did not manually switch anything.
    # -----------------------------------------------------

    set_active_document(
        resolved_document_id
    )

    source_pages = answer.get(
        "source_pages",
        [],
    )

    resolved_page: int | None = None

    if (
        isinstance(source_pages, list)
        and source_pages
    ):
        first_page = source_pages[0]

        if isinstance(
            first_page,
            int,
        ):
            resolved_page = (
                first_page
            )

    citations = answer.get(
        "citations",
        [],
    )

    resolved_figure_id: (
        str | None
    ) = None

    if isinstance(
        citations,
        list,
    ):
        for citation in citations:

            if not isinstance(
                citation,
                dict,
            ):
                continue

            figure_id = str(
                citation.get(
                    "figure_id",
                    "",
                )
                or ""
            ).strip()

            if figure_id:
                resolved_figure_id = (
                    figure_id
                )
                break

    # -----------------------------------------------------
    # SAVE PAGE / FIGURE CONTEXT TO THE CORRECT DOCUMENT
    #
    # Each document keeps its own little page/figure context.
    # -----------------------------------------------------

    if (
        resolved_page is not None
        or resolved_figure_id is not None
    ):
        update_document_context(
            document_id=(
                resolved_document_id
            ),
            page=resolved_page,
            figure_id=(
                resolved_figure_id
            ),
        )

    return (
        resolved_document_id,
        _document_name(
            resolved_document_id
        ),
    )


def _render_source_document(
    document_name: str | None,
) -> None:
    """
    Show which uploaded document the backend used.

    Display-only.
    There is no document selector here.
    """

    if not document_name:
        return

    st.caption(
        f"Using uploaded source: {document_name}"
    )


def _normalize_markdown_math(
    value: object,
) -> str:
    """
    Normalize only LaTeX delimiter syntax for Streamlit Markdown.

    The backend Physics content is left untouched.

    Supported conversions:
        \\( ... \\) -> $ ... $
        \\[ ... \\] -> $$ ... $$

    Existing $...$ and $$...$$ expressions remain unchanged.
    """

    text = str(value)

    return (
        text
        .replace(r"\[", "$$")
        .replace(r"\]", "$$")
        .replace(r"\(", "$")
        .replace(r"\)", "$")
    )


def _normalize_st_latex(
    value: object,
) -> str:
    """
    st.latex expects the LaTeX expression itself, not surrounding
    Markdown/display delimiters. Strip one matching outer pair only.
    """

    text = str(value).strip()

    wrappers = (
        ("$$", "$$"),
        (r"\[", r"\]"),
        (r"\(", r"\)"),
        ("$", "$"),
    )

    for opening, closing in wrappers:
        if (
            text.startswith(opening)
            and text.endswith(closing)
            and len(text)
            > len(opening) + len(closing)
        ):
            return text[
                len(opening):
                len(text) - len(closing)
            ].strip()

    return text


def _render_tutor_answer(
    answer: dict[str, Any],
) -> None:
    """
    Render the structured TutorAnswer returned
    by the FastAPI backend.

    Display rules:
    - preserve returned Physics text/math without rewriting it;
    - show "Not enough verified" clearly when the backend marks a
      best-effort draft as unverified;
    - show a retrieved document numerical's problem statement before
      the attempted solution;
    - render dedicated formula fields through Streamlit LaTeX.
    """

    direct_answer = answer.get(
        "direct_answer"
    )

    normalized_direct_answer = (
        str(direct_answer)
        if direct_answer is not None
        else ""
    )

    unverified_label = (
        "Not enough verified"
    )

    is_unverified_draft = (
        normalized_direct_answer
        .lstrip()
        .startswith(
            unverified_label
        )
    )

    if is_unverified_draft:
        st.warning(
            unverified_label
        )

        # The backend already embeds the label in direct_answer so the API
        # contract remains self-describing. Avoid showing the same label twice
        # in Streamlit while leaving all other returned content untouched.
        visible_direct_answer = (
            normalized_direct_answer
            .lstrip()
        )

        visible_direct_answer = (
            visible_direct_answer[
                len(unverified_label):
            ]
            .lstrip(
                "\n\r :—-"
            )
        )

    else:
        visible_direct_answer = (
            normalized_direct_answer
        )

    # -----------------------------------------------------
    # DOCUMENT NUMERICAL SOURCE PROBLEM
    #
    # The backend stores this separately from the solution so
    # the UI can satisfy:
    #
    #     source problem first
    #         ↓
    #     explanation / solution
    # -----------------------------------------------------

    problem_statement = answer.get(
        "problem_statement"
    )

    if problem_statement:
        st.markdown(
            "**Problem statement**"
        )

        # Do not normalize, replace, or reconstruct Unicode/math here.
        st.markdown(
            _normalize_markdown_math(
                problem_statement
            )
        )

    if visible_direct_answer:
        st.markdown(
            _normalize_markdown_math(
                visible_direct_answer
            )
        )

    steps = answer.get(
        "steps",
        [],
    )

    if steps:
        st.markdown(
            "**Steps**"
        )

        for index, step in enumerate(
            steps,
            start=1,
        ):
            st.markdown(
                (
                    f"{index}. "
                    + _normalize_markdown_math(
                        step
                    )
                )
            )

    formulae = answer.get(
        "formulae",
        [],
    )

    if formulae:
        st.markdown(
            "**Formulae**"
        )

        for formula in formulae:

            if not isinstance(
                formula,
                dict,
            ):
                continue

            latex = formula.get(
                "latex"
            )

            meaning = formula.get(
                "meaning"
            )

            if latex:
                # Dedicated LaTeX field: render directly.
                # Never rewrite symbols or apply regex substitutions.
                normalized_latex = (
                    _normalize_st_latex(
                        latex
                    )
                )

                if normalized_latex:
                    st.latex(
                        normalized_latex
                    )

            if meaning:
                st.caption(
                    _normalize_markdown_math(
                        meaning
                    )
                )

    diagram_explanation = (
        answer.get(
            "diagram_explanation"
        )
    )

    if diagram_explanation:
        st.markdown(
            "**Diagram explanation**"
        )

        st.markdown(
            _normalize_markdown_math(
                diagram_explanation
            )
        )

    common_mistake = answer.get(
        "common_mistake"
    )

    if common_mistake:
        st.warning(
            (
                "Common mistake: "
                + _normalize_markdown_math(
                    common_mistake
                )
            )
        )

    final_result = answer.get(
        "final_result"
    )

    if final_result:
        if is_unverified_draft:
            st.info(
                (
                    "Unverified result: "
                    + _normalize_markdown_math(
                        final_result
                    )
                )
            )
        else:
            st.success(
                (
                    "Final result: "
                    + _normalize_markdown_math(
                        final_result
                    )
                )
            )

    source_pages = answer.get(
        "source_pages",
        [],
    )

    citations = answer.get(
        "citations",
        [],
    )

    if source_pages or citations:

        with st.expander(
            "Sources and citations"
        ):

            if source_pages:
                pages = ", ".join(
                    str(page)
                    for page
                    in source_pages
                )

                st.markdown(
                    (
                        "**Source pages:** "
                        f"{pages}"
                    )
                )

            for citation in citations:

                if not isinstance(
                    citation,
                    dict,
                ):
                    continue

                page_number = (
                    citation.get(
                        "page_number"
                    )
                )

                figure_id = (
                    citation.get(
                        "figure_id"
                    )
                )

                chunk_ids = (
                    citation.get(
                        "source_chunk_ids",
                        [],
                    )
                )

                if (
                    page_number
                    is not None
                ):
                    st.markdown(
                        (
                            f"**Page "
                            f"{page_number}**"
                        )
                    )

                if figure_id:
                    st.caption(
                        (
                            "Figure: "
                            f"{figure_id}"
                        )
                    )

                if chunk_ids:
                    st.caption(
                        (
                            "Evidence chunks: "
                            + ", ".join(
                                str(chunk_id)
                                for chunk_id
                                in chunk_ids
                            )
                        )
                    )


def _selection_result_store(
) -> dict[str, dict[str, Any]]:
    store = st.session_state.get(
        _SELECTION_RESULT_STATE_KEY
    )

    if not isinstance(
        store,
        dict,
    ):
        store = {}
        st.session_state[
            _SELECTION_RESULT_STATE_KEY
        ] = store

    return store


def _selection_event_store(
) -> dict[str, str]:
    store = st.session_state.get(
        _SELECTION_EVENT_STATE_KEY
    )

    if not isinstance(
        store,
        dict,
    ):
        store = {}
        st.session_state[
            _SELECTION_EVENT_STATE_KEY
        ] = store

    return store


def _selection_source_page(
    answer: dict[str, Any],
) -> int | None:
    source_pages = answer.get(
        "source_pages",
        [],
    )

    if not isinstance(
        source_pages,
        list,
    ):
        return None

    for page in source_pages:
        if (
            isinstance(page, int)
            and page >= 1
        ):
            return page

    return None


def _selection_figure_id(
    answer: dict[str, Any],
) -> str | None:
    citations = answer.get(
        "citations",
        [],
    )

    if not isinstance(
        citations,
        list,
    ):
        return None

    for citation in citations:
        if not isinstance(
            citation,
            dict,
        ):
            continue

        figure_id = citation.get(
            "figure_id"
        )

        if not isinstance(
            figure_id,
            str,
        ):
            continue

        normalized = figure_id.strip()

        if normalized:
            return normalized

    return None


def _handle_selection_request(
    *,
    api_client: PhyMentorAPIClient,
    component_key: str,
    message_key: str,
    document_id: str | None,
    answer: dict[str, Any],
    selected_model: str | None,
) -> None:
    """
    Component callback.

    Read the transient JavaScript selection event from
    Streamlit Session State, send it through the normal
    frontend API client, then store the public response
    for this assistant message.
    """

    component_state = (
        st.session_state.get(
            component_key
        )
    )

    ask_event = getattr(
        component_state,
        "ask",
        None,
    )

    if not isinstance(
        ask_event,
        dict,
    ):
        return

    selected_text = str(
        ask_event.get(
            "selected_text",
            "",
        )
        or ""
    ).strip()

    if not selected_text:
        return

    event_id = str(
        ask_event.get(
            "event_id",
            "",
        )
        or ""
    ).strip()

    event_store = (
        _selection_event_store()
    )

    if (
        event_id
        and event_store.get(
            message_key
        )
        == event_id
    ):
        return

    if event_id:
        event_store[
            message_key
        ] = event_id

    surrounding_text = str(
        ask_event.get(
            "surrounding_text",
            "",
        )
        or ""
    ).strip()

    if not surrounding_text:
        surrounding_text = None

    try:
        response = (
            api_client.explain_selection(
                user_id=(
                    st.session_state.user_id
                ),
                session_id=(
                    st.session_state.session_id
                ),
                selected_text=(
                    selected_text
                ),
                surrounding_text=(
                    surrounding_text
                ),
                document_id=(
                    document_id
                ),
                selected_page=(
                    _selection_source_page(
                        answer
                    )
                ),
                selected_figure_id=(
                    _selection_figure_id(
                        answer
                    )
                ),
                selected_model=(
                    selected_model
                ),
            )
        )

    except (
        APIClientError,
        ValueError,
    ) as exc:
        response = {
            "selected_text": (
                selected_text
            ),
            "found": False,
            "explanation": (
                "Selection explanation failed: "
                f"{exc}"
            ),
        }

    except Exception:
        response = {
            "selected_text": (
                selected_text
            ),
            "found": False,
            "explanation": (
                "The selected text could not "
                "be explained right now."
            ),
        }

    _selection_result_store()[
        message_key
    ] = response


def _render_selection_explainer(
    *,
    api_client: PhyMentorAPIClient,
    message_index: int,
    answer: dict[str, Any],
    document_id: str | None,
    selected_model: str | None,
) -> None:
    """
    Mount one invisible selection listener inside this
    assistant message.

    The actual assistant answer keeps using Streamlit's
    existing Markdown/LaTeX renderer. The component only
    listens for a text selection and shows the tiny
    "Ask PhyMentor" action.
    """

    message_key = (
        f"assistant-{message_index}"
    )

    component_key = (
        "phymentor_selection_"
        f"{message_index}"
    )

    stored_response = (
        _selection_result_store().get(
            message_key
        )
    )

    _SELECTION_COMPONENT(
        data={
            "response": stored_response,
        },
        on_ask_change=lambda: (
            _handle_selection_request(
                api_client=api_client,
                component_key=(
                    component_key
                ),
                message_key=message_key,
                document_id=document_id,
                answer=answer,
                selected_model=(
                    selected_model
                ),
            )
        ),
        key=component_key,
        width="stretch",
        height=(
            110
            if stored_response
            else 1
        ),
    )


def _render_chat_history(
    *,
    api_client: PhyMentorAPIClient,
) -> None:
    """
    Re-render conversation messages after
    Streamlit reruns.
    """

    for message_index, message in enumerate(
        st.session_state.messages
    ):

        role = message.get(
            "role",
            "assistant",
        )

        with st.chat_message(
            role
        ):

            if role == "user":

                st.markdown(
                    message.get(
                        "content",
                        "",
                    )
                )

                continue

            resolved_document_name = (
                message.get(
                    "resolved_document_name"
                )
            )

            _render_source_document(
                resolved_document_name
            )

            answer = message.get(
                "answer"
            )

            if isinstance(
                answer,
                dict,
            ):
                _render_tutor_answer(
                    answer
                )

                _render_selection_explainer(
                    api_client=api_client,
                    message_index=(
                        message_index
                    ),
                    answer=answer,
                    document_id=(
                        message.get(
                            "resolved_document_id"
                        )
                    ),
                    selected_model=(
                        message.get(
                            "selected_model"
                        )
                    ),
                )


def render_chat_panel(
    *,
    api_client: PhyMentorAPIClient,
) -> None:
    """
    Render the student chat interface.

    There is NO manual General/Document mode.

    There is also NO manual document switcher.

    Every READY document in this session is supplied as a
    lightweight candidate reference.

    LangGraph decides:

        1. Does this question need a document?

        2. If yes, which uploaded document does it refer to?

        3. Retrieval then runs only inside that resolved
           document.
    """

    st.subheader(
        "Physics Tutor"
    )

    _render_chat_history(
        api_client=api_client
    )

    # -----------------------------------------------------
    # SESSION BOOKSHELF
    #
    # These are all retrieval-ready documents available to
    # the automatic backend resolver.
    # -----------------------------------------------------

    ready_documents = (
        get_ready_documents_for_chat()
    )

    active_document_id = (
        st.session_state
        .active_document_id
    )

    active_document_status = str(
        st.session_state
        .active_document_status
        or ""
    ).strip().upper()

    # -----------------------------------------------------
    # RECENT DOCUMENT HINT
    #
    # Only send it as the recent contextual document when
    # that document is actually retrieval-ready.
    #
    # Its presence does NOT force RAG.
    # -----------------------------------------------------

    recent_ready_document_id: (
        str | None
    ) = None

    if (
        active_document_id
        and active_document_status
        == "READY"
    ):
        recent_ready_document_id = (
            active_document_id
        )

    # -----------------------------------------------------
    # FRIENDLY CHAT CAPTION
    # -----------------------------------------------------

    if ready_documents:

        count = len(
            ready_documents
        )

        if count == 1:
            st.caption(
                (
                    "1 uploaded Physics document "
                    "is ready. PhyMentor will use it "
                    "automatically when relevant."
                )
            )

        else:
            st.caption(
                (
                    f"{count} uploaded Physics documents "
                    "are ready. PhyMentor will choose "
                    "the relevant one automatically."
                )
            )

    elif active_document_id:

        st.caption(
            (
                "Your uploaded material is still "
                "being processed. General Physics "
                "questions can still be asked."
            )
        )

    else:

        st.caption(
            (
                "Ask any school-level "
                "Physics question."
            )
        )

    selected_model_label = st.selectbox(
        "Model",
        options=list(
            _MODEL_OPTIONS.keys()
        ),
        key="phymentor_selected_model",
        help=(
            "Auto uses PhyMentor's normal task-based model routing. "
            "Choosing a model applies that model to this chat request."
        ),
    )

    selected_model_for_request = (
        _MODEL_OPTIONS[
            selected_model_label
        ]
    )

    # -----------------------------------------------------
    # OPTIONAL PAGE / FIGURE CONTEXT
    #
    # These are request hints only. They never select a
    # document and they never force document-grounded mode.
    # If left blank, the existing automatic resolver decides
    # the relevant page/figure exactly as before.
    # -----------------------------------------------------

    explicit_page_hint: int | None = None
    explicit_figure_hint: str | None = None

    if ready_documents:
        with st.expander(
            "Optional page / figure context",
            expanded=False,
        ):
            context_columns = st.columns(
                2
            )

            with context_columns[0]:
                page_hint_value = st.number_input(
                    "Page number",
                    min_value=1,
                    value=None,
                    step=1,
                    key="phymentor_optional_page_hint",
                    placeholder="Optional",
                    help=(
                        "Optional page preference for the question. "
                        "Leave blank to let PhyMentor choose automatically."
                    ),
                )

            with context_columns[1]:
                figure_hint_value = st.text_input(
                    "Figure ID",
                    value="",
                    key="phymentor_optional_figure_hint",
                    placeholder="Optional",
                    help=(
                        "Optional canonical figure ID, such as one shown "
                        "in Sources and citations. Leave blank for automatic "
                        "figure resolution."
                    ),
                )

            st.caption(
                (
                    "These hints narrow source context only after "
                    "PhyMentor resolves the relevant document. They do "
                    "not manually select a document. With multiple "
                    "documents, still identify the source naturally in "
                    "your question when needed."
                )
            )

        if page_hint_value is not None:
            explicit_page_hint = int(
                page_hint_value
            )

        normalized_figure_hint = str(
            figure_hint_value
            or ""
        ).strip()

        if normalized_figure_hint:
            explicit_figure_hint = (
                normalized_figure_hint
            )

    query = st.chat_input(
        "Ask a Physics question..."
    )

    if not query:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message(
        "user"
    ):
        st.markdown(
            query
        )

    st.session_state.is_chatting = True
    st.session_state.last_error = None

    try:

        # -------------------------------------------------
        # PAGE / FIGURE SAFETY IN MULTI-DOCUMENT MODE
        #
        # If several documents exist, do NOT blindly attach
        # a page/figure belonging to the recent document.
        #
        # Otherwise:
        #
        # recent = Optics page 3
        #
        # query =
        # "Explain the fission diagram"
        #
        # could accidentally carry Optics page 3 into the
        # fission request.
        #
        # The backend/session memory can resolve contextual
        # page/figure references after the document itself
        # has been chosen.
        # -------------------------------------------------

        if (
            explicit_page_hint is not None
            or explicit_figure_hint is not None
        ):
            # A student-entered hint is intentionally request-local
            # context. It takes precedence over remembered page/figure
            # context but still does not choose the document.
            selected_page_for_request = (
                explicit_page_hint
            )
            selected_figure_for_request = (
                explicit_figure_hint
            )

        elif (
            len(ready_documents)
            <= 1
            and recent_ready_document_id
        ):
            # Preserve the existing single-document conversational
            # behavior when the student did not provide a new hint.
            selected_page_for_request = (
                st.session_state.active_page
            )

            selected_figure_for_request = (
                st.session_state.active_figure
            )

        else:
            # In multi-document mode, never leak remembered page/figure
            # context from the recent document into a different source.
            selected_page_for_request = None
            selected_figure_for_request = None

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "PhyMentor is thinking..."
            ):

                response = (
                    api_client.chat(
                        user_id=(
                            st.session_state
                            .user_id
                        ),
                        session_id=(
                            st.session_state
                            .session_id
                        ),
                        query=query,

                        # Do not send the recent document as an explicit
                        # document choice. The backend can still use
                        # available_documents + Redis session memory to resolve
                        # document-grounded questions and natural follow-ups.
                        #
                        # This keeps standalone questions such as
                        # "What is Newton's law?" in general Physics mode.
                        document_id=None,

                        # CRITICAL MULTI-DOC INPUT:
                        #
                        # [
                        #     fission,
                        #     SHM,
                        #     optics,
                        #     ...
                        # ]
                        available_documents=(
                            ready_documents
                        ),

                        selected_page=(
                            selected_page_for_request
                        ),

                        selected_figure_id=(
                            selected_figure_for_request
                        ),

                        selected_model=(
                            selected_model_for_request
                        ),

                        language=(
                            st.session_state
                            .language
                        ),
                    )
                )

            answer = response.get(
                "answer",
                {},
            )

            if not isinstance(
                answer,
                dict,
            ):
                answer = {}

            (
                resolved_document_id,
                resolved_document_name,
            ) = (
                _sync_resolved_document_context(
                    response=response,
                    answer=answer,
                )
            )

            # Display which source was automatically chosen.
            _render_source_document(
                resolved_document_name
            )

            _render_tutor_answer(
                answer
            )

            assistant_message_index = (
                len(
                    st.session_state.messages
                )
            )

            response_selected_model = (
                response.get(
                    "selected_model"
                )
                or selected_model_for_request
            )

            _render_selection_explainer(
                api_client=api_client,
                message_index=(
                    assistant_message_index
                ),
                answer=answer,
                document_id=(
                    resolved_document_id
                ),
                selected_model=(
                    response_selected_model
                ),
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "answer": answer,

                # Useful for rendering history after
                # Streamlit reruns.
                "resolved_document_id": (
                    resolved_document_id
                ),

                "resolved_document_name": (
                    resolved_document_name
                ),

                # Preserve the model that produced this
                # assistant turn so a later selection
                # explanation can keep the same model
                # when the student explicitly chose one.
                "selected_model": (
                    response_selected_model
                ),
            }
        )

    except (
        APIClientError,
        ValueError,
    ) as exc:

        st.session_state.last_error = (
            str(exc)
        )

        with st.chat_message(
            "assistant"
        ):
            st.error(
                (
                    "Chat request failed: "
                    f"{exc}"
                )
            )

    except Exception as exc:

        st.session_state.last_error = (
            (
                "An unexpected frontend "
                "error occurred."
            )
        )

        with st.chat_message(
            "assistant"
        ):

            st.error(
                (
                    "The question could "
                    "not be completed."
                )
            )

            st.exception(
                exc
            )

    finally:

        st.session_state.is_chatting = (
            False
        )