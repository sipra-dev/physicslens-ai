from __future__ import annotations

from typing import Any

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from src.graph.edges import (
    route_after_cache,
    route_after_verifier,
)
from src.graph.nodes.serving_nodes import (
    ServingNodes,
)
from src.graph.state import PhysicsTutorState


def route_after_structural_resolution(
    state: PhysicsTutorState,
) -> str:
    """
    Stop before retrieval only when the structural resolver has produced
    a bounded clarification question.

    Every resolved, no-match, unavailable, or disabled-resolver outcome
    continues through the existing rewrite/retrieval path. This preserves
    the established FAISS + BM25 + reranker workflow.
    """

    if (
        state.get(
            "structural_clarification_required",
            False,
        )
        and state.get("answer_draft")
        is not None
    ):
        return "output_guard"

    return "rewrite_contextual_query"


def build_physics_tutor_graph(
    *,
    nodes: ServingNodes,
    checkpointer: Any | None = None,
):
    """
    Build and compile the local PhyMentor AI serving LangGraph.

    Architecture:
        START
          ↓
        validate_request
          ↓
        load_session
          ↓
        classify_intent
          ↓
        scope_guard
          ↓
        resolve_active_document
          ↓
        load_short_term_memory
          ↓
        resolve_structural_reference
          ├── bounded ambiguity → output_guard
          └── resolved / fallback → rewrite_contextual_query
                                      ↓
                                check_query_cache
                                  ├── safe fast path/cache hit → output_guard
                                  └── cache miss → retrieval_planner
                                                       ↓
                                                 hybrid_retrieval
                                                       ↓
                                                    reranking
                                                       ↓
                                             visual_context_resolver
                                                       ↓
                                               context_compression
                                                       ↓
                                                  tutor_agent
                                                       ↓
                                                verifier_agent
                                                 ↙    ↓     ↘
                                             PASS   RETRY   REGENERATE
                                              ↓       ↓        ↓
                                        output_guard broader   tutor_agent
                                                     retrieval
                                                        ↓
                                                    reranking
                                                        ↓
                                             visual_context_resolver
                                                        ↓
                                               context_compression
                                                        ↓
                                                  tutor_agent

        When the two-generation budget is exhausted:
            insufficient_evidence_response
                      ↓
                output_guard
                      ↓
            memory_write_decision
                      ↓
                 cache_write
                      ↓
                   respond
                      ↓
                     END

    TutorAgent and VerifierAgent remain the only two agents.
    Structural resolution and all other routing/retrieval steps are
    deterministic orchestration components.

    `checkpointer` is optional here:
    - None: normal local graph execution without LangGraph workflow persistence.
    - A LangGraph-compatible checkpointer: workflow checkpoints can be enabled
      without changing this graph topology.
    """

    if nodes is None:
        raise ValueError(
            "nodes cannot be None."
        )

    builder = StateGraph(
        PhysicsTutorState
    )

    # =========================================================
    # NODES
    # =========================================================

    builder.add_node(
        "validate_request",
        nodes.validate_request,
    )

    builder.add_node(
        "load_session",
        nodes.load_session,
    )

    builder.add_node(
        "classify_intent",
        nodes.classify_intent,
    )

    builder.add_node(
        "scope_guard",
        nodes.scope_guard,
    )

    builder.add_node(
        "resolve_active_document",
        nodes.resolve_active_document,
    )

    builder.add_node(
        "load_short_term_memory",
        nodes.load_short_term_memory,
    )

    builder.add_node(
        "resolve_structural_reference",
        nodes.resolve_structural_reference,
    )

    builder.add_node(
        "rewrite_contextual_query",
        nodes.rewrite_contextual_query,
    )

    builder.add_node(
        "check_query_cache",
        nodes.check_query_cache,
    )

    builder.add_node(
        "retrieval_planner",
        nodes.retrieval_planner,
    )

    builder.add_node(
        "hybrid_retrieval",
        nodes.hybrid_retrieval,
    )

    builder.add_node(
        "reranking",
        nodes.reranking,
    )

    builder.add_node(
        "visual_context_resolver",
        nodes.visual_context_resolver,
    )

    builder.add_node(
        "context_compression",
        nodes.context_compression,
    )

    builder.add_node(
        "tutor_agent",
        nodes.tutor_agent_node,
    )

    builder.add_node(
        "verifier_agent",
        nodes.verifier_agent_node,
    )

    builder.add_node(
        "broader_retrieval",
        nodes.broader_retrieval,
    )

    builder.add_node(
        "insufficient_evidence_response",
        nodes.insufficient_evidence_response,
    )

    builder.add_node(
        "output_guard",
        nodes.output_guard,
    )

    builder.add_node(
        "memory_write_decision",
        nodes.memory_write_decision,
    )

    builder.add_node(
        "cache_write",
        nodes.cache_write,
    )

    builder.add_node(
        "respond",
        nodes.respond,
    )

    # =========================================================
    # FIXED EDGES
    # =========================================================

    builder.add_edge(
        START,
        "validate_request",
    )

    builder.add_edge(
        "validate_request",
        "load_session",
    )

    builder.add_edge(
        "load_session",
        "classify_intent",
    )

    builder.add_edge(
        "classify_intent",
        "scope_guard",
    )

    builder.add_edge(
        "scope_guard",
        "resolve_active_document",
    )

    builder.add_edge(
        "resolve_active_document",
        "load_short_term_memory",
    )

    builder.add_edge(
        "load_short_term_memory",
        "resolve_structural_reference",
    )

    # =========================================================
    # STRUCTURAL RESOLUTION BRANCH
    # =========================================================

    builder.add_conditional_edges(
        "resolve_structural_reference",
        route_after_structural_resolution,
        {
            "output_guard": (
                "output_guard"
            ),
            "rewrite_contextual_query": (
                "rewrite_contextual_query"
            ),
        },
    )

    builder.add_edge(
        "rewrite_contextual_query",
        "check_query_cache",
    )

    # =========================================================
    # CACHE / FAST-PATH BRANCH
    # =========================================================

    builder.add_conditional_edges(
        "check_query_cache",
        route_after_cache,
        {
            "output_guard": (
                "output_guard"
            ),
            "retrieval_planner": (
                "retrieval_planner"
            ),
        },
    )

    # =========================================================
    # NORMAL RETRIEVAL → TUTOR → VERIFIER PATH
    # =========================================================

    builder.add_edge(
        "retrieval_planner",
        "hybrid_retrieval",
    )

    builder.add_edge(
        "hybrid_retrieval",
        "reranking",
    )

    builder.add_edge(
        "reranking",
        "visual_context_resolver",
    )

    builder.add_edge(
        "visual_context_resolver",
        "context_compression",
    )

    builder.add_edge(
        "context_compression",
        "tutor_agent",
    )

    builder.add_edge(
        "tutor_agent",
        "verifier_agent",
    )

    # =========================================================
    # VERIFIER CONTROL BRANCH
    # =========================================================

    builder.add_conditional_edges(
        "verifier_agent",
        route_after_verifier,
        {
            "output_guard": (
                "output_guard"
            ),
            "broader_retrieval": (
                "broader_retrieval"
            ),
            "tutor_agent": (
                "tutor_agent"
            ),
            "insufficient_evidence_response": (
                "insufficient_evidence_response"
            ),
        },
    )

    # RETRY_RETRIEVAL gets one broader retrieval round, then the same
    # retrieval post-processing path before the final Tutor attempt.
    builder.add_edge(
        "broader_retrieval",
        "reranking",
    )

    # Retry budget exhausted or verifier outcome is unsafe.
    builder.add_edge(
        "insufficient_evidence_response",
        "output_guard",
    )

    # =========================================================
    # GUARDED TERMINAL PATH
    # =========================================================

    builder.add_edge(
        "output_guard",
        "memory_write_decision",
    )

    builder.add_edge(
        "memory_write_decision",
        "cache_write",
    )

    builder.add_edge(
        "cache_write",
        "respond",
    )

    builder.add_edge(
        "respond",
        END,
    )

    # =========================================================
    # COMPILE
    # =========================================================

    if checkpointer is None:
        return builder.compile()

    return builder.compile(
        checkpointer=checkpointer
    )
