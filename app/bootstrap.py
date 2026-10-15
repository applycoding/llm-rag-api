from app.config import Settings
from app.embeddings.factory import build_embedder
from app.llm.factory import build_llm
from app.rag.service import RagService
from app.store.chroma import ChromaVectorStore


def build_rag_service(settings: Settings) -> RagService:
    return RagService(
        settings=settings,
        embedder=build_embedder(settings),
        store=ChromaVectorStore(settings),
        llm=build_llm(settings),
    )
