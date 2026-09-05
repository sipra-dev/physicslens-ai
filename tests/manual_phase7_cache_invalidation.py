from __future__ import annotations

import asyncio
from types import SimpleNamespace

from redis.exceptions import RedisError

import src.cache.invalidation as invalidation
from src.cache.semantic_cache import SemanticCache


class FakeRedisClient:
    def __init__(self) -> None:
        self.calls = []

    async def eval(
        self,
        script: str,
        number_of_keys: int,
        registry_key: str,
    ) -> int:
        self.calls.append(
            {
                "script": script,
                "number_of_keys": number_of_keys,
                "registry_key": registry_key,
            }
        )
        return 3


class FailingRedisClient:
    async def eval(
        self,
        script: str,
        number_of_keys: int,
        registry_key: str,
    ) -> int:
        raise RedisError(
            "Simulated Redis failure"
        )


async def main() -> None:
    original_redis_client = (
        invalidation.redis_client
    )

    try:
        # ==============================================
        # 1. NORMAL INVALIDATION
        # ==============================================

        fake_client = FakeRedisClient()

        invalidation.redis_client = (
            SimpleNamespace(
                client=fake_client
            )
        )

        result = await (
            invalidation
            .invalidate_document_cache(
                user_id="user-a",
                document_id="document-x",
            )
        )

        expected_registry_key = (
            SemanticCache
            ._document_registry_key(
                user_id="user-a",
                document_id="document-x",
            )
        )

        invalidation_returned_true = (
            result is True
        )

        eval_called_once = (
            len(fake_client.calls) == 1
        )

        correct_registry_key = (
            eval_called_once
            and fake_client.calls[0][
                "registry_key"
            ]
            == expected_registry_key
        )

        one_redis_key_passed = (
            eval_called_once
            and fake_client.calls[0][
                "number_of_keys"
            ]
            == 1
        )

        script = (
            fake_client.calls[0]["script"]
            if eval_called_once
            else ""
        )

        script_reads_registry = (
            "SMEMBERS" in script
        )

        script_deletes_entries = (
            "DEL" in script
        )

        # ==============================================
        # 2. USER / DOCUMENT ISOLATION
        # ==============================================

        user_a_doc_x = (
            SemanticCache
            ._document_registry_key(
                user_id="user-a",
                document_id="document-x",
            )
        )

        user_b_doc_x = (
            SemanticCache
            ._document_registry_key(
                user_id="user-b",
                document_id="document-x",
            )
        )

        user_a_doc_y = (
            SemanticCache
            ._document_registry_key(
                user_id="user-a",
                document_id="document-y",
            )
        )

        cache_isolated = (
            user_a_doc_x
            != user_b_doc_x
            and user_a_doc_x
            != user_a_doc_y
        )

        # ==============================================
        # 3. REDIS FAILURE MUST BE FAIL-OPEN
        # ==============================================

        invalidation.redis_client = (
            SimpleNamespace(
                client=FailingRedisClient()
            )
        )

        redis_failure_result = await (
            invalidation
            .invalidate_document_cache(
                user_id="user-a",
                document_id="document-x",
            )
        )

        redis_failure_fail_open = (
            redis_failure_result is False
        )

        # ==============================================
        # 4. INVALID INPUT
        # ==============================================

        empty_user_result = await (
            invalidation
            .invalidate_document_cache(
                user_id="   ",
                document_id="document-x",
            )
        )

        empty_document_result = await (
            invalidation
            .invalidate_document_cache(
                user_id="user-a",
                document_id="   ",
            )
        )

        invalid_input_rejected = (
            empty_user_result is False
            and empty_document_result is False
        )

        # ==============================================
        # RESULTS
        # ==============================================

        print(
            "INVALIDATION_RETURNED_TRUE="
            f"{invalidation_returned_true}"
        )

        print(
            "REDIS_EVAL_CALLED_ONCE="
            f"{eval_called_once}"
        )

        print(
            "CORRECT_REGISTRY_KEY="
            f"{correct_registry_key}"
        )

        print(
            "ONE_REDIS_KEY_PASSED="
            f"{one_redis_key_passed}"
        )

        print(
            "SCRIPT_READS_REGISTRY="
            f"{script_reads_registry}"
        )

        print(
            "SCRIPT_DELETES_CACHE="
            f"{script_deletes_entries}"
        )

        print(
            "USER_DOCUMENT_ISOLATED="
            f"{cache_isolated}"
        )

        print(
            "REDIS_FAILURE_FAIL_OPEN="
            f"{redis_failure_fail_open}"
        )

        print(
            "INVALID_INPUT_REJECTED="
            f"{invalid_input_rejected}"
        )

        all_passed = all(
            [
                invalidation_returned_true,
                eval_called_once,
                correct_registry_key,
                one_redis_key_passed,
                script_reads_registry,
                script_deletes_entries,
                cache_isolated,
                redis_failure_fail_open,
                invalid_input_rejected,
            ]
        )

        print()

        if all_passed:
            print(
                "PHASE7_CACHE_INVALIDATION_TEST=PASS"
            )
            return

        raise SystemExit(
            "PHASE7_CACHE_INVALIDATION_TEST=FAIL"
        )

    finally:
        invalidation.redis_client = (
            original_redis_client
        )


if __name__ == "__main__":
    asyncio.run(main())