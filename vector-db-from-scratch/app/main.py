"""
FastAPI Application Entrypoint.
Mounts REST API routes, static assets, and single-page dashboard.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router, index_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize vector database state and default corpus on startup."""
    print("Starting up Vector Database service...")
    index_manager.ensure_initialized()
    active_count = index_manager.ivf_index.size()
    print(f"Vector Database ready with {active_count:,} vectors loaded into Exact & IVF indices.")
    yield
    print("Shutting down Vector Database service...")


app = FastAPI(
    title="Vector Database From Scratch",
    description="Educational Vector Database implementing Exact & IVF-Flat indexing using pure NumPy.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes at root (e.g. /vectors, /search, /stats, /rebuild)
app.include_router(api_router)

# Mount API routes with /api/v1 prefix as well
app.include_router(api_router, prefix="/api/v1")

# Mount frontend static directory if exists
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
