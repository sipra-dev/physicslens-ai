from __future__ import annotations

import asyncio
from uuid import uuid4

from src.cache.redis_client import (
    redis_client,
)


async def main() -> None:
    test_key = (
        "phymentor:"
        "rate-limit-test:"
        f"{uuid4().hex}"
    )

    window_seconds = 60

    try:
        # ---------------------------------------------
        # 1. Real Redis Cloud health
        # ---------------------------------------------

        health_ok = (
            await redis_client.health_check()
        )

        print(
            "REDIS_HEALTH="
            f"{health_ok}"
        )

        # Ensure clean test state.
        await redis_client.delete(
            test_key
        )

        # ---------------------------------------------
        # 2. First atomic increment
        # ---------------------------------------------

        first = (
            await redis_client
            .rate_limit_increment(
                key=test_key,
                window_seconds=window_seconds,
            )
        )

        first_ok = (
            first is not None
            and first[0] == 1
            and 1 <= first[1] <= window_seconds
        )

        print(
            "FIRST_INCREMENT_OK="
            f"{first_ok}"
        )

        # ---------------------------------------------
        # 3. Second increment
        # ---------------------------------------------

        second = (
            await redis_client
            .rate_limit_increment(
                key=test_key,
                window_seconds=window_seconds,
            )
        )

        second_ok = (
            second is not None
            and second[0] == 2
            and 1 <= second[1] <= window_seconds
        )

        print(
            "SECOND_INCREMENT_OK="
            f"{second_ok}"
        )

        # ---------------------------------------------
        # 4. Third increment
        # ---------------------------------------------

        third = (
            await redis_client
            .rate_limit_increment(
                key=test_key,
                window_seconds=window_seconds,
            )
        )

        third_ok = (
            third is not None
            and third[0] == 3
            and 1 <= third[1] <= window_seconds
        )

        print(
            "THIRD_INCREMENT_OK="
            f"{third_ok}"
        )

        # ---------------------------------------------
        # 5. TTL should already exist
        # ---------------------------------------------

        ttl_ok = (
            first is not None
            and second is not None
            and third is not None
            and first[1] > 0
            and second[1] > 0
            and third[1] > 0
        )

        print(
            "RATE_LIMIT_TTL_OK="
            f"{ttl_ok}"
        )

        # ---------------------------------------------
        # 6. Final verdict
        # ---------------------------------------------

        all_ok = all(
            (
                health_ok,
                first_ok,
                second_ok,
                third_ok,
                ttl_ok,
            )
        )

        print()

        if all_ok:
            print(
                "PHASE7_REDIS_RATE_LIMIT_CORE_TEST=PASS"
            )
        else:
            print(
                "PHASE7_REDIS_RATE_LIMIT_CORE_TEST=FAIL"
            )

    finally:
        # Never leave test counters behind.
        cleaned = (
            await redis_client.delete(
                test_key
            )
        )

        print(
            "TEST_KEY_CLEANED="
            f"{cleaned}"
        )

        await redis_client.close()


if __name__ == "__main__":
    asyncio.run(
        main()
    )