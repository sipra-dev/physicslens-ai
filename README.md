# PhyMentor AI

PhyMentor AI is a production-minded, multi-agent Physics tutoring system that combines multimodal document understanding, hybrid Retrieval-Augmented Generation (RAG), semantic memory, verification, and conversational follow-up handling.

Users can upload Physics documents containing text, equations, figures, worked examples, and numerical problems, then ask document-specific or general Physics questions through a Streamlit chat interface backed by FastAPI.

## Key Features

### Multi-Agent Tutoring
- Tutor Agent generates explanations, derivations, and numerical solutions.
- Verifier Agent independently checks groundedness and answer quality.
- LangGraph coordinates deterministic routing, state management, verification, retries, and serving logic.

### Hybrid RAG Pipeline
- Dense semantic retrieval using FAISS.
- Sparse lexical retrieval using BM25.
- Reciprocal Rank Fusion (RRF) combines candidate results.
- CrossEncoder reranking improves final evidence selection.
- Context filtering and compression keep retrieved evidence relevant and bounded.

The retrieval layer is separated into dense retrieval, BM25, fusion, reranking, filtering, compression, pipeline, and service components. :contentReference[oaicite:0]{index=0}

### Multimodal Document Ingestion
The ingestion pipeline handles:
- Native PDF text
- OCR
- Page layout
- Headings and structured items
- Equations
- Figures and captions
- Chunking
- Deduplication
- Scope classification
- Validation
- Index generation

These responsibilities are implemented as dedicated ingestion modules rather than a single monolithic parser. :contentReference[oaicite:1]{index=1}

### Document-Aware Query Routing
PhyMentor distinguishes between:
- general Physics questions,
- document-grounded questions,
- follow-up questions,
- ambiguous references,
- structural document queries.

This prevents unrelated questions from being incorrectly forced into previously uploaded documents.

### Memory and Session Recovery
The system includes:
- Redis-backed session state
- semantic response caching
- Pinecone-backed semantic learning memory
- long-term memory services
- PostgreSQL-backed persistence/checkpointing
- recoverable previous chat sessions

The codebase contains dedicated semantic, long-term, Pinecone, PostgreSQL, and session-memory services. :contentReference[oaicite:2]{index=2}

### Guardrails and Verification
- User/document isolation
- Input validation
- retrieval evidence validation
- numerical verification
- response verification
- model routing and fallback handling
- rate limiting
- structured API error handling

### Frontend and API
The application separates the user interface and backend API:
- Streamlit frontend
- FastAPI backend
- API client abstraction
- document upload/status handling
- chat interface
- previous-session recovery

The frontend includes dedicated API client, session-state, chat, upload, and status components. :contentReference[oaicite:3]{index=3}  
The backend contains separate API middleware, schemas, routes, and error-handling modules. :contentReference[oaicite:4]{index=4}

## Architecture

```text
                         ┌────────────────────┐
                         │ Streamlit Frontend │
                         └─────────┬──────────┘
                                   │ HTTP
                                   ▼
                         ┌────────────────────┐
                         │    FastAPI API     │
                         └─────────┬──────────┘
                                   │
                                   ▼
                    ┌───────────────────────────┐
                    │ Query Understanding /     │
                    │ Document Resolution       │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
              ┌─────────────────────────────────────┐
              │          Hybrid Retrieval           │
              │                                     │
              │ FAISS ─┐                            │
              │        ├─ RRF ─ CrossEncoder ─────┐ │
              │ BM25 ──┘                          │ │
              └───────────────────────────────────┼─┘
                                                  │
                                                  ▼
                                          ┌─────────────┐
                                          │ Tutor Agent │
                                          └──────┬──────┘
                                                 │
                                                 ▼
                                        ┌────────────────┐
                                        │ Verifier Agent │
                                        └───────┬────────┘
                                                │
                                                ▼
                                          Final Response
                                                │
                       ┌────────────────────────┼─────────────────────┐
                       ▼                        ▼                     ▼
                    Redis                   Pinecone             PostgreSQL
              Cache / Sessions        Learning Memory       Persistence

Technology Stack

Language

Python 3.11

Backend

FastAPI
Uvicorn
Pydantic

Frontend

Streamlit

Agent Orchestration

LangGraph

Retrieval / NLP

FAISS
BM25
Reciprocal Rank Fusion
Sentence Transformers
CrossEncoder reranking

Document Processing

PyMuPDF
PyPDF
Tesseract OCR
Pillow

Memory / Persistence

Redis
PostgreSQL
Pinecone

LLM Layer

OpenAI models
model routing and fallback

Evaluation

RAGAS
Pytest

Containerization

Docker
Docker Compose
GitHub Container Registry (GHCR)

Cloud Build

AWS EC2
RAG Evaluation

The hybrid RAG pipeline was evaluated using RAGAS on a curated Physics QA benchmark.

Metric	Result
Retrieval coverage	90%
Context Precision	0.90
Context Recall	0.83
Faithfulness	0.61
Answer Relevancy	0.63
Answer Correctness	0.49
Entity Recall	0.40

The evaluation highlights strong retrieval performance while also exposing generation-quality areas that can be improved further.

Docker

The complete application stack was validated using Docker Compose with:

Frontend
FastAPI API
Redis
PostgreSQL

All four services were successfully validated through container health checks.

The original image included unnecessary CUDA-enabled PyTorch dependencies and was approximately:

7.02 GB

A CPU-only PyTorch build reduced the final image to:

2.28 GB

without changing the normal local development environment.

This represents approximately a 68% reduction in container image size.

Pull the Public Docker Image

The optimized image is published publicly on GitHub Container Registry:

docker pull ghcr.io/sipra-dev/phymentor-ai:latest

Anonymous pull access was independently verified after publishing.

Run with Docker Compose

Clone the repository and configure the required environment variables:

git clone https://github.com/sipra-dev/physicslens-ai.git
cd physicslens-ai

Create your local .env file with the required credentials.

Then run:

docker compose -f infra/docker/compose.yaml up

The stack contains:

PostgreSQL
Redis
FastAPI API
Streamlit frontend

Secrets are not stored inside the Docker image or committed to Git.

Project Structure
physicslens-ai/
├── apps/
│   ├── api/
│   │   ├── middleware/
│   │   ├── routes/
│   │   └── schemas/
│   └── frontend/
│       ├── components/
│       ├── api_client.py
│       ├── app.py
│       └── session_state.py
│
├── src/
│   ├── agents/
│   ├── cache/
│   ├── config/
│   ├── graph/
│   ├── guardrails/
│   ├── ingestion/
│   ├── memory/
│   ├── models/
│   ├── prompts/
│   ├── query/
│   ├── retrieval/
│   ├── serving/
│   ├── storage/
│   └── verification/
│
├── evaluation/
├── tests/
├── infra/
│   └── docker/
│       ├── Dockerfile
│       └── compose.yaml
│
├── requirements.txt
├── .dockerignore
└── .gitignore

The test suite includes unit and integration coverage for the model gateway, agents, verifier, serving layer, equations, LangGraph wiring, chat API, and retrieval contracts.

Design Principles

PhyMentor AI was designed around several production-oriented principles:

Keep frontend and backend responsibilities separated.
Keep retrieval deterministic and inspectable.
Use agents only where reasoning or verification adds value.
Preserve user and document isolation.
Never expose environment secrets through source control or container images.
Maintain explicit fallbacks and error handling.
Evaluate retrieval quality quantitatively rather than relying only on demos.
Keep the architecture modular enough for individual components to be tested independently.
Current Status

Implemented and validated:

Multimodal document ingestion
Hybrid RAG
Tutor and Verifier agents
LangGraph orchestration
Semantic caching
Long-term learning memory
Session recovery
Query and document resolution
Guardrails
RAGAS evaluation
FastAPI backend
Streamlit frontend
Redis and PostgreSQL integration
Docker / Docker Compose
CPU-optimized container image
Public GHCR distribution
AWS EC2 remote container build workflow
Future Work

Potential next steps include:

public cloud deployment,
custom domain and HTTPS,
richer observability,
automated CI/CD,
further answer-quality optimization,
optional GPU-specific container image.
