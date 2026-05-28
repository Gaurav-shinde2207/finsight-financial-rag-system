from fastapi import FastAPI

from app.api.routes import router
from app.observability.logging import configure_logging
from app.utils.config import get_settings


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    api = FastAPI(
        title=settings.app_name,
        description="Financial document intelligence API with RAG and source citations.",
        version="0.1.0",
    )
    api.include_router(router, prefix="/api/v1")
    return api


app = create_app()
