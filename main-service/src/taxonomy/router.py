import json
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from src.taxonomy.utils import _post_to_service, _put_to_service
from src.configs.config import settings
from src.models.schemas import (
    TaxonomyShufflePayload,
    TaxonomyUpdate,
    TaxonomyEditPayload,
)

router = APIRouter()

@router.get("/")
async def get_taxonomy():
    for item in ["base", "full"]:
        with open(settings.PROCESSED_DATA_DIR / f"taxonomy_{item}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if item == "base":
            base = data
        else:
            full = data
    return JSONResponse(status_code=status.HTTP_200_OK, content={"base": base, "full": list(set(full) - set(base))})

@router.put("/")
async def update_taxonomy(update: TaxonomyUpdate):
    payload = update.model_dump(mode="json")
    return await _put_to_service(settings.TAXONOMY_URL_MODEL, payload)

@router.post("/shuffle")
async def shuffle_taxonomy(payload: TaxonomyShufflePayload):
    body = payload.model_dump(mode="json")
    return await _post_to_service(settings.TAXONOMY_SHUFFLE_URL_MODEL, body)

@router.post("/edit")
async def edit_taxonomy(payload: TaxonomyEditPayload):
    body = payload.model_dump(mode="json")
    return await _post_to_service(settings.TAXONOMY_EDIT_URL_MODEL, body)
