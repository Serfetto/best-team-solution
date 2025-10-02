from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from src.configs.config import settings
from src.export.utils import _get_from_service

router = APIRouter()

@router.get("/combined/")
async def get_combined_export():
    return await _get_from_service(settings.EXPORT_COMBINED_URL_MODEL)


@router.get("/dataset/{name}/")
async def get_dataset_export(name: str):
    return await _get_from_service(settings.EXPORT_DATASET_URL_MODEL+{name})

@router.get("/derived_taxonomy/")
async def get_dataset_export():
    return await _get_from_service(settings.EXPORT_DERIVED_TAXONOMY_URL_MODEL)


@router.get("/enriched_combined.csv")
async def download_enriched_combined_xlsx():
    """Return the local enriched_combined.csv from processed data directory."""
    path = settings.PROCESSED_DATA_DIR / "enriched_combined.csv"
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="enriched_combined.csv not found")
    return FileResponse(
        path,
        media_type="text/csv",
        filename="enriched_combined.csv",
    )
