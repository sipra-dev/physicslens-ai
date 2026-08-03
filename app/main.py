from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pypdf import PdfReader


app = FastAPI(
    title="PhysicsLens AI",
    description="Backend API for the PhysicsLens multimodal tutor.",
    version="0.1.0",
)


# Uploaded files এখানে save হবে
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Allowed file formats
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
}


# Maximum file size: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024

# File একবারে 1 MB করে read হবে
CHUNK_SIZE = 1024 * 1024


def extract_pdf_text(file_path: Path) -> list[dict[str, object]]:
    """PDF-এর প্রতিটি page থেকে text extract করে."""

    try:
        reader = PdfReader(file_path)
        pages: list[dict[str, object]] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            pages.append(
                {
                    "page_number": page_number,
                    "text": text.strip(),
                }
            )

        return pages

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="PDF text could not be extracted.",
        ) from error


@app.get("/")
async def home() -> dict[str, str]:
    """API চলছে কি না check করে."""

    return {
        "message": "PhysicsLens API is running"
    }


@app.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
    file: UploadFile = File(...),
) -> dict[str, object]:
    """একটি PDF বা image validate করে local folder-এ save করে."""

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename was provided.",
        )

    original_filename = file.filename
    content_type = file.content_type or "unknown"

    extension = Path(original_filename).suffix.lower()

    # File extension check
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF, PNG, JPG and JPEG files are allowed.",
        )

    # File content type check
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file content type.",
        )

    # Unique filename তৈরি করা হচ্ছে
    saved_filename = f"{uuid4().hex}{extension}"
    destination = UPLOAD_DIR / saved_filename

    total_size = 0

    try:
        with destination.open("wb") as output_file:
            while True:
                chunk = await file.read(CHUNK_SIZE)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File is larger than the 10 MB limit.",
                    )

                output_file.write(chunk)

    except HTTPException:
        destination.unlink(missing_ok=True)
        raise

    except OSError as error:
        destination.unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The file could not be saved.",
        ) from error

    finally:
        await file.close()

    # Empty file check
    if total_size == 0:
        destination.unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    # PDF হলে text extraction হবে
    extracted_pages: list[dict[str, object]] = []

    if extension == ".pdf":
        extracted_pages = extract_pdf_text(destination)

    return {
        "message": "File uploaded successfully.",
        "original_filename": original_filename,
        "saved_filename": saved_filename,
        "content_type": content_type,
        "size_bytes": total_size,
        "number_of_pages": len(extracted_pages),
        "extracted_pages": extracted_pages,
    }