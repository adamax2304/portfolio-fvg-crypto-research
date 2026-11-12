"""
# v2.6 — premium terminal backend entrypoint (FastAPI)
#
# Run:
#   uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import api_router
from backend.ws import ws_router


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


app = FastAPI(
    title="FVG Trading Terminal API",
    version="2.7-mtf",
    description="Backend API + WebSocket streams for the premium FVG dashboard with MTF support.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "FVG Trading Terminal API is running",
        "docs": "/docs",
        "time": utc_now_iso(),
    }

