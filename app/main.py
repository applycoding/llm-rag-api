import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import router
from app.bootstrap import build_rag_service
from app.config import Settings, get_settings
from app.errors import AppError

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.rag = build_rag_service(app.state.settings)
        try:
            yield
        finally:
            rag = getattr(app.state, "rag", None)
            if rag is not None:
                await rag.aclose()

    app = FastAPI(
        title=resolved.app_name,
        version=__version__,
        description=(
            "Portfolio demo of retrieval-augmented generation. "
            "The default provider is offline and does not call an external model."
        ),
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.include_router(router)
    app.add_exception_handler(AppError, _app_error_handler)
    return app


async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning("%s %s -> %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


app = create_app()
