from __future__ import annotations

import subprocess
import sys


TEST_MODULES = [
    "tests.manual_phase7_document_dedupe",
    "tests.manual_phase7_document_dedupe_route",
    "tests.manual_phase7_indexer_cleanup",
    "tests.manual_phase7_document_delete_cleanup",
    "tests.manual_phase7_processing_failure",
    "tests.manual_phase7_cache_invalidation",
]


COMPILE_FILES = [
    "src/ingestion/service.py",
    "src/ingestion/indexer.py",
    "src/cache/semantic_cache.py",
    "src/cache/invalidation.py",
    "apps/api/routes/documents.py",
]


def run_command(
    command: list[str],
) -> bool:
    result = subprocess.run(
        command,
        text=True,
    )

    return result.returncode == 0


def main() -> None:
    print("=" * 60)
    print("PHASE 7 FINAL SMOKE TEST")
    print("=" * 60)
    print()

    # =====================================================
    # 1. COMPILE IMPORTANT PHASE 7 FILES
    # =====================================================

    print("[1] Compiling Phase 7 files...")

    compile_ok = run_command(
        [
            sys.executable,
            "-m",
            "py_compile",
            *COMPILE_FILES,
        ]
    )

    print(
        f"PHASE7_COMPILE_OK={compile_ok}"
    )

    if not compile_ok:
        raise SystemExit(
            "PHASE7_FINAL_SMOKE_TEST=FAIL"
        )

    print()

    # =====================================================
    # 2. RUN PHASE 7 MANUAL TESTS
    # =====================================================

    results: dict[str, bool] = {}

    for module in TEST_MODULES:
        print("-" * 60)
        print(f"Running: {module}")
        print("-" * 60)

        passed = run_command(
            [
                sys.executable,
                "-m",
                module,
            ]
        )

        results[module] = passed

        print(
            f"{module}="
            f"{'PASS' if passed else 'FAIL'}"
        )

        print()

    # =====================================================
    # 3. FINAL RESULT
    # =====================================================

    all_passed = (
        compile_ok
        and all(results.values())
    )

    print("=" * 60)

    if all_passed:
        print(
            "PHASE7_FINAL_SMOKE_TEST=PASS"
        )
        print(
            "PHASE_7_COMPLETE=True"
        )
        return

    print(
        "PHASE7_FINAL_SMOKE_TEST=FAIL"
    )

    print()
    print("Failed tests:")

    for module, passed in results.items():
        if not passed:
            print(f"- {module}")

    raise SystemExit(1)


if __name__ == "__main__":
    main()