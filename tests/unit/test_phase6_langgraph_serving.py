from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from src.graph.builder import build_physics_tutor_graph
from src.graph.nodes.serving_nodes import ServingNodes
from src.models.contracts import (
    AnswerType,
    IntentDecision,
    LanguageCode,
    MemorySnapshot,
    QueryRewriteResult,
    QueryScopeDecision,
    QueryUnderstandingResult,
    RequestIntent,
    ScopeStatus,
    SourceCitation,
    TutorAnswer,
    VerificationAction,
    VerificationResult,
)
from src.models.routing import UserSelectableModel
from src.retrieval.models import (
    ContextBundle,
    ContextItem,
    HybridRetrievalResult,
)


class FakeQueryService:
    def __init__(
        self,
        understanding: QueryUnderstandingResult,
    ) -> None:
        self.understanding = understanding
        self.calls: list[dict[str, Any]] = []

    def understand(
        self,
        *,
        query: str,
        memory: MemorySnapshot | None = None,
        upload_present: bool = False,
        selected_model: (
            UserSelectableModel
            | str
            | None
        ) = None,
    ) -> QueryUnderstandingResult:
        self.calls.append(
            {
                "query": query,
                "memory": memory,
                "upload_present": upload_present,
                "selected_model": selected_model,
            }
        )
        return self.understanding


class FakeRetrievalService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        **kwargs,
    ) -> HybridRetrievalResult:
        self.calls.append(dict(kwargs))

        query = kwargs["query"]
        user_id = kwargs["user_id"]
        document_id = kwargs["document_id"]

        text = (
            f"Evidence for {query}. "
            "Acceleration is the rate of change of velocity."
        )

        context = ContextBundle(
            query=query,
            user_id=user_id,
            document_id=document_id,
            items=[
                ContextItem(
                    context_id=(
                        f"ctx-{len(self.calls)}"
                    ),
                    user_id=user_id,
                    document_id=document_id,
                    page_number=2,
                    source_chunk_ids=[
                        f"chunk-{len(self.calls)}"
                    ],
                    parent_id="parent-2",
                    text=text,
                    content_type="text",
                    linked_figure_ids=[],
                    equations=[],
                    image_path=None,
                    caption=None,
                    rerank_score=0.95,
                )
            ],
            total_characters=len(text),
            truncated=False,
        )

        return HybridRetrievalResult(
            query=query,
            context=context,
            evidence_found=True,
        )


class FakeDocumentResolutionRetrievalService:
    """
    Resolver-only fake with document-specific evidence.

    The document resolver compares:
    - CrossEncoder rerank scores
    - lexical overlap with returned context

    These tests therefore need different evidence for different
    candidate documents instead of the generic FakeRetrievalService,
    which intentionally returns the same acceleration text for every
    document.
    """

    def __init__(
        self,
        evidence_by_document: dict[
            str,
            tuple[
                str,
                list[float],
            ],
        ],
    ) -> None:
        self.evidence_by_document = (
            evidence_by_document
        )
        self.calls: list[
            dict[str, Any]
        ] = []

    def retrieve(
        self,
        **kwargs: Any,
    ) -> HybridRetrievalResult:
        self.calls.append(
            dict(kwargs)
        )

        query = kwargs["query"]
        user_id = kwargs["user_id"]
        document_id = kwargs[
            "document_id"
        ]

        evidence = (
            self.evidence_by_document.get(
                document_id
            )
        )

        if evidence is None:
            return HybridRetrievalResult(
                query=query,
                context=ContextBundle(
                    query=query,
                    user_id=user_id,
                    document_id=document_id,
                    items=[],
                    total_characters=0,
                    truncated=False,
                ),
                evidence_found=False,
            )

        text, scores = evidence

        context = ContextBundle(
            query=query,
            user_id=user_id,
            document_id=document_id,
            items=[
                ContextItem(
                    context_id=(
                        f"resolver-{document_id}"
                    ),
                    user_id=user_id,
                    document_id=document_id,
                    page_number=1,
                    source_chunk_ids=[
                        (
                            "resolver-chunk-"
                            f"{document_id}"
                        )
                    ],
                    parent_id=(
                        "resolver-parent-"
                        f"{document_id}"
                    ),
                    text=text,
                    content_type="text",
                    linked_figure_ids=[],
                    equations=[],
                    image_path=None,
                    caption=None,
                    rerank_score=(
                        scores[0]
                        if scores
                        else None
                    ),
                )
            ],
            total_characters=len(text),
            truncated=False,
        )

        # Resolver code needs only `.rerank_score` from these items.
        # model_construct keeps this test lightweight and avoids
        # fabricating the entire production RetrievalHit hierarchy.
        reranked_hits = [
            SimpleNamespace(
                rerank_score=score
            )
            for score in scores
        ]

        return (
            HybridRetrievalResult
            .model_construct(
                query=query,
                dense_hits=[],
                bm25_hits=[],
                fused_hits=[],
                reranked_hits=(
                    reranked_hits
                ),
                context=context,
                evidence_found=True,
                failure_reason=None,
            )
        )


class FakeTutorAgent:
    def __init__(
        self,
        answers: list[TutorAnswer],
    ) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    def answer(
        self,
        **kwargs,
    ) -> TutorAnswer:
        self.calls.append(dict(kwargs))

        index = min(
            len(self.calls) - 1,
            len(self.answers) - 1,
        )

        return self.answers[index]


class FakeVerifierAgent:
    def __init__(
        self,
        results: list[VerificationResult],
    ) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def verify(
        self,
        **kwargs,
    ) -> VerificationResult:
        self.calls.append(dict(kwargs))

        index = min(
            len(self.calls) - 1,
            len(self.results) - 1,
        )

        return self.results[index]


class FakeQueryCache:
    def __init__(self) -> None:
        self.get_calls: list[
            dict[str, Any]
        ] = []
        self.set_calls: list[
            dict[str, Any]
        ] = []

    def get(
        self,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        self.get_calls.append(
            dict(kwargs)
        )
        return None

    def set(
        self,
        **kwargs: Any,
    ) -> bool:
        self.set_calls.append(
            dict(kwargs)
        )
        return True


class Phase6LangGraphServingTests(
    unittest.TestCase
):
    def _intent(
        self,
        request_intent: RequestIntent,
        *,
        language: LanguageCode = (
            LanguageCode.ENGLISH
        ),
        prefer_visual: bool = False,
    ) -> IntentDecision:
        return IntentDecision(
            intent=request_intent,
            confidence=0.99,
            language=language,
            estimated_grade=9,
            has_physics_request=(
                request_intent
                not in {
                    RequestIntent.GREETING,
                    RequestIntent.OUT_OF_SCOPE,
                    RequestIntent.UNSUPPORTED,
                    RequestIntent.UPLOAD_DOCUMENT,
                }
            ),
            is_follow_up=(
                request_intent
                == RequestIntent.FOLLOW_UP
            ),
            prefer_visual=prefer_visual,
        )

    def _scope(
        self,
        *,
        in_scope: bool = True,
    ) -> QueryScopeDecision:
        if in_scope:
            return QueryScopeDecision(
                is_physics=True,
                school_level=True,
                supported=True,
                status=ScopeStatus.IN_SCOPE,
                estimated_grade_range=[8, 10],
                topics=[
                    "motion_and_kinematics"
                ],
                confidence=0.99,
                reason=(
                    "Supported school-level "
                    "Physics."
                ),
            )

        return QueryScopeDecision(
            is_physics=False,
            school_level=False,
            supported=False,
            status=ScopeStatus.OUT_OF_SCOPE,
            estimated_grade_range=None,
            topics=[],
            confidence=0.99,
            reason=(
                "Outside supported Physics scope."
            ),
        )

    def _rewrite(
        self,
    ) -> QueryRewriteResult:
        return QueryRewriteResult(
            original_query=(
                "What is acceleration?"
            ),
            rewritten_query=(
                "What is acceleration?"
            ),
            retrieval_queries=[
                "What is acceleration?"
            ],
            was_rewritten=False,
            prefer_visual=False,
            preferred_page_numbers=[],
            referenced_figure_id=None,
            use_hyde=False,
            hyde_text=None,
        )

    def _understanding(
        self,
        *,
        intent: RequestIntent = (
            RequestIntent.PHYSICS_QUESTION
        ),
        active_document_id: (
            str | None
        ) = "doc-test",
        in_scope: bool = True,
        language: LanguageCode = (
            LanguageCode.ENGLISH
        ),
    ) -> QueryUnderstandingResult:
        return QueryUnderstandingResult(
            normalized_query=(
                "What is acceleration?"
            ),
            intent=self._intent(
                intent,
                language=language,
                prefer_visual=(
                    intent
                    == RequestIntent
                    .DIAGRAM_QUESTION
                ),
            ),
            scope=(
                None
                if intent
                in {
                    RequestIntent.GREETING,
                    RequestIntent
                    .UPLOAD_DOCUMENT,
                }
                else self._scope(
                    in_scope=in_scope
                )
            ),
            rewrite=(
                None
                if intent
                in {
                    RequestIntent.GREETING,
                    RequestIntent
                    .UPLOAD_DOCUMENT,
                }
                or not in_scope
                else self._rewrite()
            ),
            active_document_id=(
                active_document_id
            ),
        )

    def _answer(
        self,
        text: str = (
            "Acceleration is the rate of "
            "change of velocity."
        ),
    ) -> TutorAnswer:
        return TutorAnswer(
            answer_type=(
                AnswerType.CONCEPT_EXPLANATION
            ),
            direct_answer=text,
            steps=[],
            formulae=[],
            diagram_explanation=None,
            common_mistake=None,
            final_result=None,
            source_pages=[2],
            citations=[
                SourceCitation(
                    page_number=2,
                    source_chunk_ids=[
                        "chunk-1"
                    ],
                    figure_id=None,
                )
            ],
        )

    def _verification(
        self,
        action: VerificationAction,
        *,
        issues: list[str] | None = None,
        grounded: bool = True,
    ) -> VerificationResult:
        is_pass = (
            action
            == VerificationAction.PASS
        )

        return VerificationResult(
            grounded=(
                True
                if is_pass
                else grounded
            ),
            physics_correct=True,
            calculation_correct=True,
            units_correct=True,
            diagram_claims_supported=True,
            within_school_scope=True,
            citation_valid=True,
            issues=issues or [],
            action=action,
            confidence=0.99,
        )

    def _graph(
        self,
        *,
        understanding: (
            QueryUnderstandingResult
        ),
        tutor_answers: (
            list[TutorAnswer] | None
        ) = None,
        verification_results: (
            list[VerificationResult] | None
        ) = None,
        query_cache: Any | None = None,
    ):
        query = FakeQueryService(
            understanding
        )
        retrieval = (
            FakeRetrievalService()
        )
        tutor = FakeTutorAgent(
            tutor_answers
            or [
                self._answer()
            ]
        )
        verifier = FakeVerifierAgent(
            verification_results
            or [
                self._verification(
                    VerificationAction.PASS
                )
            ]
        )

        nodes = ServingNodes(
            query_service=query,
            retrieval_service=retrieval,
            tutor_agent=tutor,
            verifier_agent=verifier,
            query_cache=query_cache,
        )

        graph = (
            build_physics_tutor_graph(
                nodes=nodes
            )
        )

        return (
            graph,
            query,
            retrieval,
            tutor,
            verifier,
        )

    def _state(
        self,
        *,
        query: str = (
            "What is acceleration?"
        ),
        document_id: (
            str | None
        ) = "doc-test",
        memory: (
            MemorySnapshot | None
        ) = None,
        selected_model: (
            UserSelectableModel
            | None
        ) = None,
    ) -> dict[str, Any]:
        return {
            "request_id": "req-1",
            "user_id": "local-user",
            "session_id": "session-1",
            "raw_query": query,
            "explicit_document_id": (
                document_id
            ),
            "memory": (
                memory
                if memory is not None
                else MemorySnapshot()
            ),
            "upload_present": False,
            "selected_model": (
                selected_model
            ),
        }

    def _resolver_nodes(
        self,
        retrieval_service: Any,
    ) -> ServingNodes:
        return ServingNodes(
            query_service=FakeQueryService(
                self._understanding(
                    active_document_id=None
                )
            ),
            retrieval_service=(
                retrieval_service
            ),
            tutor_agent=FakeTutorAgent(
                [self._answer()]
            ),
            verifier_agent=(
                FakeVerifierAgent(
                    [
                        self._verification(
                            VerificationAction.PASS
                        )
                    ]
                )
            ),
        )

    def test_broad_kinematics_reference_semantically_selects_acc_document(
        self,
    ) -> None:
        """
        Regression target from the Phase-8 handoff:

        "kinematics related document" must be able to select
        acc.jpeg by document CONTENT even though the filename
        itself does not contain the word "kinematics".
        """

        retrieval = (
            FakeDocumentResolutionRetrievalService(
                {
                    "doc-nuclear": (
                        (
                            "Nuclear fission chain reaction, "
                            "uranium nuclei and neutrons."
                        ),
                        [0.18, 0.12],
                    ),
                    "doc-acc": (
                        (
                            "Kinematics describes motion using "
                            "displacement, velocity and acceleration."
                        ),
                        [0.95, 0.88],
                    ),
                }
            )
        )

        nodes = self._resolver_nodes(
            retrieval
        )

        documents = [
            {
                "document_id": "doc-nuclear",
                "name": "nuclear.jpg",
            },
            {
                "document_id": "doc-acc",
                "name": "acc.jpeg",
            },
        ]

        state: dict[str, Any] = {
            "user_id": "local-user",
            "normalized_query": (
                "kinematics related document"
            ),
            "memory": MemorySnapshot(),
            "intent": self._intent(
                RequestIntent
                .PHYSICS_QUESTION
            ),
            "scope": self._scope(),
            "query_understanding": (
                self._understanding(
                    active_document_id=None
                )
            ),
            "prefer_visual": False,
        }

        (
            resolved_document_id,
            ambiguous,
        ) = nodes._resolve_document_for_turn(
            state=state,
            available_documents=documents,
            recent_document_id=None,
        )

        self.assertEqual(
            resolved_document_id,
            "doc-acc",
        )
        self.assertFalse(
            ambiguous
        )

        probed_ids = {
            call["document_id"]
            for call in retrieval.calls
        }

        self.assertEqual(
            probed_ids,
            {
                "doc-nuclear",
                "doc-acc",
            },
        )

    def test_short_second_point_followup_stays_on_previous_grounded_document(
        self,
    ) -> None:
        """
        "2nd point" refers to a point inside the previous answer,
        NOT to the second uploaded document.

        This is the exact short-follow-up grounding regression
        called out in the Phase-8 handoff.
        """

        nodes = self._resolver_nodes(
            FakeDocumentResolutionRetrievalService(
                {}
            )
        )

        documents = [
            {
                "document_id": "doc-acc",
                "name": "acc.jpeg",
            },
            {
                "document_id": "doc-nuclear",
                "name": "nuclear.jpg",
            },
        ]

        followup_understanding = (
            self._understanding(
                intent=(
                    RequestIntent.FOLLOW_UP
                ),
                active_document_id=(
                    "doc-acc"
                ),
            )
        )

        state: dict[str, Any] = {
            "user_id": "local-user",
            "normalized_query": (
                "please explain the 2nd point"
            ),
            "memory": MemorySnapshot(
                active_document_id=(
                    "doc-acc"
                ),
                last_turn_document_id=(
                    "doc-acc"
                ),
            ),
            "intent": (
                followup_understanding
                .intent
            ),
            "scope": (
                followup_understanding
                .scope
            ),
            "query_understanding": (
                followup_understanding
            ),
            "prefer_visual": False,
        }

        (
            resolved_document_id,
            ambiguous,
        ) = nodes._resolve_document_for_turn(
            state=state,
            available_documents=documents,
            recent_document_id="doc-acc",
        )

        self.assertEqual(
            resolved_document_id,
            "doc-acc",
        )
        self.assertFalse(
            ambiguous
        )

    def test_last_one_selects_latest_uploaded_document(
        self,
    ) -> None:
        """
        Natural positional wording from the handoff:
        "last one" means the newest document in the session bookshelf.
        """

        documents = [
            {
                "document_id": "doc-one",
                "name": "one.pdf",
            },
            {
                "document_id": "doc-two",
                "name": "two.pdf",
            },
            {
                "document_id": "doc-three",
                "name": "three.pdf",
            },
        ]

        resolved_document_id = (
            ServingNodes
            ._resolve_document_by_position(
                query=(
                    "please use the last one"
                ),
                documents=documents,
                recent_document_id=(
                    "doc-two"
                ),
            )
        )

        self.assertEqual(
            resolved_document_id,
            "doc-three",
        )

    def test_generic_document_request_remains_ambiguous_with_multiple_documents(
        self,
    ) -> None:
        """
        Safety invariant: a generic multi-document request must
        clarify instead of silently picking the latest document.
        """

        retrieval = (
            FakeDocumentResolutionRetrievalService(
                {
                    "doc-one": (
                        (
                            "Motion, force and energy "
                            "school physics notes."
                        ),
                        [0.50],
                    ),
                    "doc-two": (
                        (
                            "Waves, sound and optics "
                            "school physics notes."
                        ),
                        [0.50],
                    ),
                }
            )
        )

        nodes = self._resolver_nodes(
            retrieval
        )

        documents = [
            {
                "document_id": "doc-one",
                "name": "one.pdf",
            },
            {
                "document_id": "doc-two",
                "name": "two.pdf",
            },
        ]

        state: dict[str, Any] = {
            "user_id": "local-user",
            "normalized_query": (
                "please explain the document"
            ),
            "memory": MemorySnapshot(),
            "intent": self._intent(
                RequestIntent
                .PHYSICS_QUESTION
            ),
            "scope": self._scope(),
            "query_understanding": (
                self._understanding(
                    active_document_id=None
                )
            ),
            "prefer_visual": False,
        }

        (
            resolved_document_id,
            ambiguous,
        ) = nodes._resolve_document_for_turn(
            state=state,
            available_documents=documents,
            recent_document_id=None,
        )

        self.assertIsNone(
            resolved_document_id
        )
        self.assertTrue(
            ambiguous
        )

    def test_selected_model_reaches_query_tutor_and_verifier_and_bypasses_cache(
        self,
    ) -> None:
        models = (
            UserSelectableModel.GPT_4O,
            UserSelectableModel.GPT_5_6_SOL,
            UserSelectableModel.GPT_5_6_TERRA,
            UserSelectableModel.GPT_5_6_LUNA,
        )

        for selected_model in models:
            with self.subTest(
                selected_model=(
                    selected_model.value
                )
            ):
                cache = FakeQueryCache()

                (
                    graph,
                    query,
                    _retrieval,
                    tutor,
                    verifier,
                ) = self._graph(
                    understanding=(
                        self._understanding()
                    ),
                    query_cache=cache,
                )

                result = graph.invoke(
                    self._state(
                        selected_model=(
                            selected_model
                        )
                    )
                )

                self.assertEqual(
                    query.calls[0][
                        "selected_model"
                    ],
                    selected_model,
                )

                self.assertEqual(
                    tutor.calls[0][
                        "selected_model"
                    ],
                    selected_model,
                )

                self.assertEqual(
                    verifier.calls[0][
                        "selected_model"
                    ],
                    selected_model,
                )

                self.assertEqual(
                    result[
                        "selected_model"
                    ],
                    selected_model,
                )

                self.assertEqual(
                    len(cache.get_calls),
                    0,
                )

                self.assertEqual(
                    len(cache.set_calls),
                    0,
                )

    def test_auto_mode_keeps_existing_agent_call_shape(
        self,
    ) -> None:
        (
            graph,
            query,
            _retrieval,
            tutor,
            verifier,
        ) = self._graph(
            understanding=(
                self._understanding()
            )
        )

        result = graph.invoke(
            self._state(
                selected_model=None
            )
        )

        self.assertIsNone(
            query.calls[0][
                "selected_model"
            ]
        )

        self.assertNotIn(
            "selected_model",
            tutor.calls[0],
        )

        self.assertNotIn(
            "selected_model",
            verifier.calls[0],
        )

        self.assertIsNone(
            result.get(
                "selected_model"
            )
        )

    def test_greeting_skips_retrieval_and_agents(
        self,
    ) -> None:
        understanding = (
            self._understanding(
                intent=(
                    RequestIntent.GREETING
                ),
                active_document_id=None,
                language=(
                    LanguageCode.BENGALI
                ),
            )
        )

        (
            graph,
            query,
            retrieval,
            tutor,
            verifier,
        ) = self._graph(
            understanding=understanding
        )

        result = graph.invoke(
            self._state(
                query="হ্যালো",
                document_id=None,
                memory=MemorySnapshot(
                    language=(
                        LanguageCode.BENGALI
                    )
                ),
            )
        )

        self.assertEqual(
            len(query.calls),
            1,
        )
        self.assertEqual(
            len(retrieval.calls),
            0,
        )
        self.assertEqual(
            len(tutor.calls),
            0,
        )
        self.assertEqual(
            len(verifier.calls),
            0,
        )
        self.assertEqual(
            result[
                "generation_attempts"
            ],
            0,
        )
        self.assertEqual(
            result[
                "retrieval_rounds"
            ],
            0,
        )
        self.assertEqual(
            result[
                "final_answer"
            ].answer_type,
            AnswerType.DIRECT_ANSWER,
        )
        self.assertFalse(
            result[
                "should_write_memory"
            ]
        )

    def test_normal_document_rag_passes(
        self,
    ) -> None:
        (
            graph,
            _query,
            retrieval,
            tutor,
            verifier,
        ) = self._graph(
            understanding=(
                self._understanding()
            )
        )

        result = graph.invoke(
            self._state()
        )

        self.assertEqual(
            len(retrieval.calls),
            1,
        )
        self.assertEqual(
            len(tutor.calls),
            1,
        )
        self.assertEqual(
            len(verifier.calls),
            1,
        )
        self.assertEqual(
            result[
                "generation_attempts"
            ],
            1,
        )
        self.assertEqual(
            result[
                "retrieval_rounds"
            ],
            1,
        )
        self.assertEqual(
            result[
                "terminal_action"
            ],
            VerificationAction.PASS,
        )
        self.assertEqual(
            result[
                "final_answer"
            ].direct_answer,
            (
                "Acceleration is the rate of "
                "change of velocity."
            ),
        )
        self.assertEqual(
            len(
                result[
                    "next_memory"
                ].recent_messages
            ),
            2,
        )

    def test_regenerate_reuses_same_context(
        self,
    ) -> None:
        (
            graph,
            _query,
            retrieval,
            tutor,
            verifier,
        ) = self._graph(
            understanding=(
                self._understanding()
            ),
            tutor_answers=[
                self._answer(
                    "First draft."
                ),
                self._answer(
                    "Corrected draft."
                ),
            ],
            verification_results=[
                self._verification(
                    VerificationAction
                    .REGENERATE,
                    issues=[
                        "Wrong wording."
                    ],
                ),
                self._verification(
                    VerificationAction.PASS
                ),
            ],
        )

        result = graph.invoke(
            self._state()
        )

        self.assertEqual(
            len(retrieval.calls),
            1,
        )
        self.assertEqual(
            len(tutor.calls),
            2,
        )
        self.assertEqual(
            len(verifier.calls),
            2,
        )
        self.assertEqual(
            result[
                "generation_attempts"
            ],
            2,
        )
        self.assertEqual(
            result[
                "retrieval_rounds"
            ],
            1,
        )
        self.assertIn(
            "Wrong wording.",
            tutor.calls[1][
                "verifier_feedback"
            ],
        )
        self.assertEqual(
            result[
                "final_answer"
            ].direct_answer,
            "Corrected draft.",
        )

    def test_retry_retrieval_runs_broader_round(
        self,
    ) -> None:
        (
            graph,
            _query,
            retrieval,
            tutor,
            verifier,
        ) = self._graph(
            understanding=(
                self._understanding()
            ),
            tutor_answers=[
                self._answer(
                    "First retrieval draft."
                ),
                self._answer(
                    "Broader retrieval draft."
                ),
            ],
            verification_results=[
                self._verification(
                    VerificationAction
                    .RETRY_RETRIEVAL,
                    issues=[
                        "Need better evidence."
                    ],
                    grounded=False,
                ),
                self._verification(
                    VerificationAction.PASS
                ),
            ],
        )

        result = graph.invoke(
            self._state()
        )

        self.assertEqual(
            len(retrieval.calls),
            2,
        )
        self.assertEqual(
            len(tutor.calls),
            2,
        )
        self.assertEqual(
            len(verifier.calls),
            2,
        )
        self.assertEqual(
            result[
                "retrieval_rounds"
            ],
            2,
        )
        self.assertEqual(
            result[
                "generation_attempts"
            ],
            2,
        )

        self.assertEqual(
            retrieval.calls[0][
                "dense_top_k"
            ],
            20,
        )
        self.assertEqual(
            retrieval.calls[1][
                "dense_top_k"
            ],
            30,
        )
        self.assertEqual(
            retrieval.calls[1][
                "bm25_top_k"
            ],
            30,
        )
        self.assertEqual(
            retrieval.calls[1][
                "fused_top_k"
            ],
            40,
        )
        self.assertEqual(
            retrieval.calls[1][
                "max_contexts"
            ],
            8,
        )
        self.assertEqual(
            result[
                "final_answer"
            ].direct_answer,
            "Broader retrieval draft.",
        )

    def test_two_failures_never_create_third_tutor_call(
        self,
    ) -> None:
        (
            graph,
            _query,
            _retrieval,
            tutor,
            verifier,
        ) = self._graph(
            understanding=(
                self._understanding()
            ),
            tutor_answers=[
                self._answer(
                    "Bad first draft."
                ),
                self._answer(
                    "Bad second draft."
                ),
            ],
            verification_results=[
                self._verification(
                    VerificationAction
                    .REGENERATE,
                    issues=[
                        "First failure."
                    ],
                ),
                self._verification(
                    VerificationAction
                    .REGENERATE,
                    issues=[
                        "Second failure."
                    ],
                ),
            ],
        )

        result = graph.invoke(
            self._state()
        )

        self.assertEqual(
            len(tutor.calls),
            2,
        )
        self.assertEqual(
            len(verifier.calls),
            2,
        )
        self.assertEqual(
            result[
                "generation_attempts"
            ],
            2,
        )
        self.assertEqual(
            result[
                "terminal_action"
            ],
            VerificationAction
            .INSUFFICIENT_EVIDENCE,
        )
        self.assertEqual(
            result[
                "final_answer"
            ].answer_type,
            AnswerType
            .INSUFFICIENT_EVIDENCE,
        )
        self.assertNotIn(
            "Bad second draft.",
            result[
                "final_answer"
            ].direct_answer,
        )

    def test_rejected_tutor_draft_never_reaches_user(
        self,
    ) -> None:
        dangerous_draft = (
            "Unsupported diagram claim "
            "that must not reach the user."
        )

        (
            graph,
            _query,
            _retrieval,
            tutor,
            verifier,
        ) = self._graph(
            understanding=(
                self._understanding(
                    intent=(
                        RequestIntent
                        .DIAGRAM_QUESTION
                    )
                )
            ),
            tutor_answers=[
                self._answer(
                    dangerous_draft
                )
            ],
            verification_results=[
                self._verification(
                    VerificationAction
                    .ASK_FOR_CLEARER_IMAGE,
                    issues=[
                        (
                            "Visual evidence "
                            "is unreadable."
                        )
                    ],
                    grounded=False,
                )
            ],
        )

        result = graph.invoke(
            self._state()
        )

        self.assertEqual(
            len(tutor.calls),
            1,
        )
        self.assertEqual(
            len(verifier.calls),
            1,
        )
        self.assertEqual(
            result[
                "terminal_action"
            ],
            VerificationAction
            .ASK_FOR_CLEARER_IMAGE,
        )
        self.assertNotEqual(
            result[
                "final_answer"
            ].direct_answer,
            dangerous_draft,
        )
        self.assertNotIn(
            "Unsupported diagram claim",
            result[
                "final_answer"
            ].direct_answer,
        )
        self.assertEqual(
            result[
                "final_answer"
            ].answer_type,
            AnswerType
            .INSUFFICIENT_EVIDENCE,
        )


if __name__ == "__main__":
    unittest.main()