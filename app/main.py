"""FastAPI application factory and entry point."""
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import Settings, get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("app")

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application (used by uvicorn and by tests)."""
    resolved = settings or get_settings()

    app = FastAPI(
        title=resolved.app_name,
        description=(
            "Educational web application demonstrating REST API integration "
            "with the DeepSeek LLM cloud API."
        ),
        version="1.0.0",
    )
    app.state.settings = resolved

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(router)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Last-resort handler: log internals, return a safe generic body."""
        logger.exception(
            "Unhandled exception on %s %s", request.method, request.url.path
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Please try again later."},
        )

    return app


app = create_app()
