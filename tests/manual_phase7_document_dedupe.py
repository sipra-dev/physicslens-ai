from __future__ import annotations

import tempfile
from pathlib import Path

from src.storage.local import LocalStorage


def main() -> None:
    hash_a = "a" * 64
    hash_b = "b" * 64

    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalStorage(
            root_directory=Path(temp_dir)
        )

        # -------------------------------------------------
        # USER A — EXISTING DOCUMENT
        # -------------------------------------------------

        storage.write_document_metadata(
            user_id="user-a",
            document_id="doc-123",
            metadata={
                "document_id": "doc-123",
                "user_id": "user-a",
                "original_filename": "physics.pdf",
                "sha256": hash_a,
                "status": "READY",
            },
        )

        # -------------------------------------------------
        # 1. SAME USER + SAME HASH
        # -------------------------------------------------

        same_file = (
            storage.find_document_by_sha256(
                user_id="user-a",
                sha256=hash_a,
            )
        )

        same_file_ok = (
            same_file is not None
            and same_file.get("document_id")
            == "doc-123"
        )

        print(
            "SAME_USER_DUPLICATE_FOUND="
            f"{same_file_ok}"
        )

        # -------------------------------------------------
        # 2. SAME USER + DIFFERENT HASH
        # -------------------------------------------------

        different_file = (
            storage.find_document_by_sha256(
                user_id="user-a",
                sha256=hash_b,
            )
        )

        different_hash_ok = (
            different_file is None
        )

        print(
            "DIFFERENT_HASH_NOT_DUPLICATE="
            f"{different_hash_ok}"
        )

        # -------------------------------------------------
        # 3. DIFFERENT USER + SAME HASH
        # -------------------------------------------------

        cross_user = (
            storage.find_document_by_sha256(
                user_id="user-b",
                sha256=hash_a,
            )
        )

        cross_user_ok = (
            cross_user is None
        )

        print(
            "CROSS_USER_ISOLATION="
            f"{cross_user_ok}"
        )

        # -------------------------------------------------
        # 4. RENAMED FILE LOGIC
        # -------------------------------------------------
        #
        # Filename is deliberately irrelevant.
        # Matching is based only on content hash
        # inside the same user's storage.
        # -------------------------------------------------

        renamed_same_file = (
            storage.find_document_by_sha256(
                user_id="user-a",
                sha256=hash_a.upper(),
            )
        )

        rename_ok = (
            renamed_same_file is not None
            and renamed_same_file.get(
                "original_filename"
            )
            == "physics.pdf"
        )

        print(
            "HASH_NORMALIZATION_OK="
            f"{rename_ok}"
        )

        # -------------------------------------------------
        # FINAL
        # -------------------------------------------------

        all_ok = all(
            (
                same_file_ok,
                different_hash_ok,
                cross_user_ok,
                rename_ok,
            )
        )

        print()

        if all_ok:
            print(
                "PHASE7_DOCUMENT_DEDUPE_STORAGE_TEST=PASS"
            )
        else:
            print(
                "PHASE7_DOCUMENT_DEDUPE_STORAGE_TEST=FAIL"
            )


if __name__ == "__main__":
    main()