"""
App entrypoint. Wires up routers, DB init, CORS, templates, static files,
and the background scheduler.

TEMPORARY: instrumented with startup checkpoints to find the exact import
causing a SIGILL crash (status 132) on Render. Remove once root cause found.
"""
import sys


def _checkpoint(msg):
    print(f"[STARTUP CHECKPOINT] {msg}", flush=True)


_checkpoint("main.py starting")

from contextlib import asynccontextmanager
_checkpoint("stdlib imports done")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
_checkpoint("fastapi imports done")

from app.config import settings
_checkpoint("config imported")

from app.db.base import init_db
_checkpoint("db.base imported")

from app.routers import auth
_checkpoint("router: auth imported")

from app.routers import console
_checkpoint("router: console imported")

from app.routers import events
_checkpoint("router: events imported")

from app.routers import pages
_checkpoint("router: pages imported")

from app.routers import products
_checkpoint("router: products imported")

from app.routers import recommendations
_checkpoint("router: recommendations imported")

from app.services.scheduler import start_scheduler, stop_scheduler
_checkpoint("scheduler imported")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _checkpoint("lifespan: calling init_db")
    await init_db()
    _checkpoint("lifespan: init_db done, starting scheduler")
    start_scheduler()
    _checkpoint("lifespan: scheduler started, yielding")
    yield
    stop_scheduler()


_checkpoint("creating FastAPI app")
app = FastAPI(title="UPulse", version="0.1.0", lifespan=lifespan)
_checkpoint("FastAPI app created")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [],
    allow_methods=["*"],
    allow_headers=["*"],
)
_checkpoint("CORS middleware added")


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
_checkpoint("Jinja2Templates created")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
_checkpoint("static files mounted")

app.include_router(auth.router)
_checkpoint("router included: auth")

app.include_router(products.router)
_checkpoint("router included: products")

app.include_router(events.router)
_checkpoint("router included: events")

app.include_router(recommendations.router)
_checkpoint("router included: recommendations")

app.include_router(pages.router)
_checkpoint("router included: pages")

app.include_router(console.router)
_checkpoint("router included: console")

_checkpoint("main.py fully loaded, ready for uvicorn to serve")


@app.get("/health")
async def health():
    return {"status": "ok", "environment": settings.environment}