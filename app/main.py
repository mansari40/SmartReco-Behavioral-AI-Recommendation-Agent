"""
App entrypoint. Wires up routers, DB init, CORS, templates, static files,
and the background scheduler.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.db.base import init_db
from app.routers import admin, auth, console, events, pages, products, recommendations
from app.services.scheduler import start_scheduler, stop_scheduler
from app.services.seeder import ensure_admin, seed_catalog

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Idempotent startup bootstrap: schema -> admin -> 52 courses ->
    # course embeddings. Safe on every boot; never duplicates anything.
    await init_db()
    await ensure_admin()
    seed_stats = await seed_catalog()
    logger.info("Startup bootstrap complete: %s", seed_stats)
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title="UPulse", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def revalidate_static_assets(request, call_next):
    """Force browsers to revalidate static assets on every load. The app is
    actively developed (templates/JS change constantly); a stale cached
    tracker.js has real consequences (missing tracker methods), so never
    let clients serve long-cached assets during the demo lifecycle."""
    response = await call_next(request)
    if request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(auth.router)
app.include_router(products.router)
app.include_router(events.router)
app.include_router(recommendations.router)
app.include_router(pages.router)
app.include_router(console.router)
app.include_router(admin.router)


@app.get("/health")
async def health():
    return {"status": "ok", "environment": settings.environment}