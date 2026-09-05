from apps.api.routes.documents import (
    router as documents_router,
)
from apps.api.routes.retrieval import (
    router as retrieval_router,
)
from apps.api.routes.chat import (
    router as chat_router,
)


__all__ = [
    "documents_router",
    "retrieval_router",
    "chat_router"
]