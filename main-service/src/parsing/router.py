# src/parsing/router.py
import json
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Request, status, Depends
from fastapi.responses import JSONResponse
from src.parsing.utils import JOB_ID, PARSING_STATE, _post_to_service, _tz, read_json_parsing_state, schedule_parsing_job
from src.configs.config import settings
from src.models.database import db_dependency
from src.models.schemas import (
    FullRequest,
    IncrementalRequest,
    ParsingStartRequest,
)
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
import asyncio
from datetime import datetime, timezone
from pydantic import BaseModel, Field

router = APIRouter()

@router.get("/status/")
async def get_parsing_status(request: Request):
    records = read_json_parsing_state()
    scheduler = getattr(request.app.state, "scheduler", None)
    job = scheduler.get_job(JOB_ID)
    if records["banki"]["running"] or records["sravni"]["running"]:
        if job is not None:
            PARSING_STATE["next_started_at"] = job.next_run_time.isoformat()
    else:
        if job is not None and PARSING_STATE.get("running"):
            PARSING_STATE["next_started_at"] = job.next_run_time.isoformat()
        else:
            PARSING_STATE.update({
                "job_id": None,
                "running": False,
                "started_at": None,
                "next_started_at": None,
                "parsing_running": False,
            })
    return JSONResponse(PARSING_STATE)

@router.post("/start/")
async def start_schedule_parsing(payload: ParsingStartRequest, request: Request):
    job_id = schedule_parsing_job(request.app, payload)
    return JSONResponse({"scheduled": True, "job_id": job_id, "state": PARSING_STATE})

@router.post("/edit/")
async def edit_schedule_parsing(payload: ParsingStartRequest, request: Request):
    payload_dict = payload.model_dump()
    scheduler = getattr(request.app.state, "scheduler", None)
    
    if not scheduler:
        raise HTTPException(status_code=500, detail="Scheduler is not initialized")
    
    if not PARSING_STATE.get("job_id"):
        raise HTTPException(status_code=400, detail="No active parsing job to edit")
    
    job = scheduler.get_job(PARSING_STATE["job_id"])
    if not job:
        raise HTTPException(status_code=404, detail="Parsing job not found")
    
    # Получаем текущие параметры задачи
    current_source = PARSING_STATE.get("source", "full")
    new_source = payload_dict["source"]
    new_every = payload_dict.get("every")
    
    # Если изменился источник или интервал - пересоздаем задачу
    if new_source != current_source or new_every != PARSING_STATE.get("every"):
        # Удаляем старую задачу
        scheduler.remove_job(PARSING_STATE["job_id"])
        
        # Создаем новую задачу с обновленными параметрами
        job_id = schedule_parsing_job(request.app, payload)
        
        return JSONResponse(PARSING_STATE)
    else:
        # Если параметры не изменились, возвращаем текущее состояние
        return JSONResponse(PARSING_STATE)

# Остальные прямые прокси-роуты оставляем как есть (ниже без изменений)
@router.post("/banki/full/")
async def run_parsing_banki_full(request: FullRequest):
    payload = request.model_dump(mode="json", exclude_none=True)
    return await _post_to_service(settings.PARSING_BANKI_FULL, payload)

@router.post("/banki/incremental/")
async def run_parsing_banki_incremental(request: IncrementalRequest):
    payload = request.model_dump(mode="json", exclude_none=True)
    return await _post_to_service(settings.PARSING_BANKI_INCREMENTAL, payload)

@router.post("/sravni/full/")
async def run_parsing_sravni_full(request: FullRequest):
    payload = request.model_dump(mode="json", exclude_none=True)
    return await _post_to_service(settings.PARSING_SRAVNI_FULL, payload)

@router.post("/sravni/incremental/")
async def run_parsing_sravni_incremental(request: IncrementalRequest):
    payload = request.model_dump(mode="json", exclude_none=True)
    return await _post_to_service(settings.PARSING_SRAVNI_INCREMENTAL, payload)
