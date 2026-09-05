"""
PhyMentor retrieval package.

Keep this package initializer intentionally minimal.

Use direct module imports such as:

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.filters import RetrievalFilter
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.compression import ContextCompressor
from src.retrieval.pipeline import HybridRetrievalPipeline

This avoids circular imports as the retrieval pipeline grows.
"""