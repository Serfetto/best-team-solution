from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .banki import BankiRuParser
from .sravni import SravniRuParser
from .state import StateStore


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATE_FILE = DATA_DIR / "parser_state.json"
BANKI_CSV = DATA_DIR / "banki_ru_full.csv"
SRAVNI_CSV = DATA_DIR / "sravni_ru_full.csv"

state_store = StateStore(STATE_FILE)
banki_parser = BankiRuParser(BANKI_CSV, state_store)
sravni_parser = SravniRuParser(SRAVNI_CSV, state_store)

app = FastAPI(title="Banki/Sravni Parsing Service", version="1.0.0")


class FullRequest(BaseModel):
    pages_per_run: Optional[int] = Field(default=None, ge=1)
    start_page: Optional[int] = Field(default=None, ge=0)


class IncrementalRequest(BaseModel):
    max_pages: Optional[int] = Field(default=None, ge=1)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/state")
def get_state() -> dict:
    return state_store.load()


@app.post("/banki/full")
def banki_full(req: FullRequest) -> dict:
    result = banki_parser.full_parse(
        pages_per_run=req.pages_per_run,
        start_page=req.start_page,
    )
    snapshot = state_store.load()["banki"]
    return {
        "processed_pages": result.processed_pages,
        "added_reviews": result.added_reviews,
        "last_page": result.last_page,
        "has_more": result.has_more,
        "state": snapshot,
        "output": str(BANKI_CSV.resolve()),
    }


@app.post("/banki/incremental")
def banki_incremental(req: IncrementalRequest) -> dict:
    state_store.update(lambda s: s.setdefault("banki", {}).update({"running": True}))
    try:
        result = banki_parser.incremental(max_pages=req.max_pages)
        snapshot = state_store.load()["banki"]
        return {
            "added_reviews": result.added_reviews,
            "reached_existing": result.reached_existing,
            "state": snapshot,
            "output": str(BANKI_CSV.resolve()),
        }
    finally:
        state_store.update(lambda s: s.setdefault("banki", {}).update({"running": False}))


@app.post("/sravni/full")
def sravni_full(req: FullRequest) -> dict:
    result = sravni_parser.full_parse(
        pages_per_run=req.pages_per_run,
        start_page=req.start_page,
    )
    snapshot = state_store.load()["sravni"]
    return {
        "processed_pages": result.processed_pages,
        "added_reviews": result.added_reviews,
        "last_page": result.last_page,
        "has_more": result.has_more,
        "state": snapshot,
        "output": str(SRAVNI_CSV.resolve()),
    }


@app.post("/sravni/incremental")
def sravni_incremental(req: IncrementalRequest) -> dict:
    state_store.update(lambda s: s.setdefault("sravni", {}).update({"running": True}))
    try:
        result = sravni_parser.incremental(max_pages=req.max_pages)
        snapshot = state_store.load()["sravni"]
        return {
            "added_reviews": result.added_reviews,
            "reached_existing": result.reached_existing,
            "state": snapshot,
            "output": str(SRAVNI_CSV.resolve()),
        }
    finally:
        state_store.update(lambda s: s.setdefault("sravni", {}).update({"running": False}))


def get_app() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
