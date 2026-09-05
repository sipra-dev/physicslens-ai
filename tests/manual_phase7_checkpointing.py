from __future__ import annotations

from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from src.graph.checkpointing import (
    LangGraphCheckpointManager,
)


class TinyState(TypedDict, total=False):
    value: int
    message: str


def increment(
    state: TinyState,
) -> dict[str, int]:
    return {
        "value": (
            state.get("value", 0) + 1
        )
    }


def main() -> None:
    manager = LangGraphCheckpointManager()

    thread_id = (
        "phase7-checkpoint-test-"
        + uuid4().hex
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    try:
        # ---------------------------------------------
        # 1. Open DB + create/migrate checkpoint tables
        # ---------------------------------------------

        manager.setup()

        print(
            "CHECKPOINT_DB_HEALTH="
            f"{manager.health_check()}"
        )

        # ---------------------------------------------
        # 2. Build a tiny lightweight LangGraph
        # ---------------------------------------------

        builder = StateGraph(
            TinyState
        )

        builder.add_node(
            "increment",
            increment,
        )

        builder.add_edge(
            START,
            "increment",
        )

        builder.add_edge(
            "increment",
            END,
        )

        graph = builder.compile(
            checkpointer=(
                manager.checkpointer
            )
        )

        # ---------------------------------------------
        # 3. Run graph
        # ---------------------------------------------

        result = graph.invoke(
            {
                "value": 10,
                "message": (
                    "PhyMentor checkpoint test"
                ),
            },
            config=config,
        )

        graph_result_ok = (
            result.get("value") == 11
        )

        print(
            "GRAPH_RESULT_OK="
            f"{graph_result_ok}"
        )

        # ---------------------------------------------
        # 4. Read checkpoint back FROM PostgreSQL
        # ---------------------------------------------

        checkpoint = (
            manager.checkpointer.get_tuple(
                config
            )
        )

        checkpoint_found = (
            checkpoint is not None
        )

        print(
            "CHECKPOINT_FOUND="
            f"{checkpoint_found}"
        )

        checkpoints = list(
            manager.checkpointer.list(
                config,
                limit=10,
            )
        )

        history_found = (
            len(checkpoints) > 0
        )

        print(
            "CHECKPOINT_HISTORY_FOUND="
            f"{history_found}"
        )

        # ---------------------------------------------
        # 5. Verify LangGraph can read saved state
        # ---------------------------------------------

        snapshot = graph.get_state(
            config
        )

        saved_values = (
            snapshot.values
        )

        saved_state_ok = (
            saved_values.get("value")
            == 11
            and saved_values.get("message")
            == "PhyMentor checkpoint test"
        )

        print(
            "SAVED_STATE_OK="
            f"{saved_state_ok}"
        )

        # ---------------------------------------------
        # 6. Isolation test
        # ---------------------------------------------

        other_config = {
            "configurable": {
                "thread_id": (
                    "phase7-other-"
                    + uuid4().hex
                )
            }
        }

        other_checkpoint = (
            manager.checkpointer.get_tuple(
                other_config
            )
        )

        isolation_ok = (
            other_checkpoint is None
        )

        print(
            "THREAD_ISOLATION_OK="
            f"{isolation_ok}"
        )

        all_ok = all(
            (
                manager.health_check(),
                graph_result_ok,
                checkpoint_found,
                history_found,
                saved_state_ok,
                isolation_ok,
            )
        )

        print()

        if all_ok:
            print(
                "PHASE7_CHECKPOINT_TEST=PASS"
            )
        else:
            print(
                "PHASE7_CHECKPOINT_TEST=FAIL"
            )

        # ---------------------------------------------
        # 7. Remove test-only checkpoint data
        # ---------------------------------------------

        manager.checkpointer.delete_thread(
            thread_id
        )

        deleted = (
            manager.checkpointer.get_tuple(
                config
            )
            is None
        )

        print(
            "TEST_CHECKPOINT_CLEANED="
            f"{deleted}"
        )

    finally:
        manager.close()


if __name__ == "__main__":
    main()