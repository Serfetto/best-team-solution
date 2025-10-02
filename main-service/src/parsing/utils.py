# src/parsing/utils.py
import asyncio
from datetime import datetime, timedelta, timezone
import json
from typing import Any, Optional
from urllib.parse import urljoin
import aiohttp
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from fastapi import HTTPException, Request, status
from src.models.schemas import ParsingStartRequest
from src.configs.config import settings

PARSING_STATE: dict[str, Any] = {
    "job_id": None,
    "source": None,
    "every": None,
    "job_running": False,
    "started_at": None,
    "next_started_at": None,
    "parsing_running": False,
    "last_parsing_time": None,
}

JOB_ID = "parsing:scheduled"

parser_state_json_path = settings.BASE_DIR.parents[1] / "parser-service" / "data" / "parser_state.json"

def _tz():
    import pytz
    return pytz.timezone(getattr(settings, "TIMEZONE", "UTC"))

def read_json_parsing_state():
    with open(file=parser_state_json_path, encoding="UTF-8") as file_in:
        records = json.load(file_in)
    return records

async def run_incremental_by_source(source: str):
    try:
        if source == "full":
            await _post_to_service(settings.PARSING_BANKI_INCREMENTAL, {})
            await _post_to_service(settings.PARSING_SRAVNI_INCREMENTAL, {})
        elif source == "banki":
            await _post_to_service(settings.PARSING_BANKI_INCREMENTAL, {})
        elif source == "sravni":
            await _post_to_service(settings.PARSING_SRAVNI_INCREMENTAL, {})
        else:
            raise HTTPException(status_code=400, detail=f"Unknown source {source}")
        
        # Обновляем время последнего запуска при успешном выполнении
        PARSING_STATE["last_parsing_time"] = datetime.utcnow().isoformat()
    except Exception as e:
        # В случае ошибки не обновляем last_parsing_time
        raise

def schedule_parsing_job(app, payload: ParsingStartRequest):
    scheduler = getattr(app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=500, detail="Scheduler is not initialized")
    
    payload_dict = payload.model_dump()
    source = payload_dict["source"]
    every = payload_dict.get("every")

    # функция для джобы
    async def job_runner():
        await run_incremental_by_source(source)

        export_url = settings.Process_URL_MODEL
        await _post_to_service(export_url, {})
    
    started_at = datetime.utcnow()
    
    # строим триггер с правильными параметрами
    if every == "day":
        trigger = CronTrigger(day='*', hour=started_at.hour, minute=started_at.minute, second=started_at.second, timezone=_tz())
    elif every == "week":
        trigger = CronTrigger(day_of_week=(started_at + timedelta(days=7)).weekday(), hour=started_at.hour, minute=started_at.minute, second=started_at.second, timezone=_tz())
    elif every == "month":
        trigger = CronTrigger(month='*', day=started_at.day, hour=started_at.hour, minute=started_at.minute, second=started_at.second, timezone=_tz())
    elif every == "year":
        trigger = CronTrigger(year='*', month=started_at.month, day=started_at.day, hour=started_at.hour, minute=started_at.minute, second=started_at.second, timezone=_tz())
    elif every == "real":
        trigger = CronTrigger(minute=10, second=started_at.second, timezone=_tz())
    else:
        trigger = DateTrigger(run_date=started_at, timezone=_tz())

    scheduler.add_job(
        func=job_runner,
        trigger=trigger,
        id=JOB_ID,
        replace_existing=True,
    )

    job = scheduler.get_job(JOB_ID)
    next_rt = getattr(job, "next_run_time", None)
    
    # Сохраняем параметры задачи в состоянии
    PARSING_STATE.update({
        "job_id": JOB_ID,
        "source": source,
        "every": every,
        "running": True,
        "started_at": datetime.utcnow().isoformat(),
        "next_started_at": next_rt.isoformat() if next_rt else None,
        "parsing_running": True
    })
    return JOB_ID

async def _request_service(method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
    try:
        async with aiohttp.ClientSession() as session:
            request_kwargs: dict[str, Any] = {"json": payload} if payload is not None else {}
            async with session.request(method.upper(), url, **request_kwargs) as response:
                try:
                    body = await response.json()
                except aiohttp.ContentTypeError:
                    body = await response.text()

                if response.status >= 400:
                    detail = body.get("message") if isinstance(body, dict) else body
                    raise HTTPException(status_code=response.status, detail=detail)

                return body
    except aiohttp.ClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Service request failed: {exc}",
        ) from exc



async def _post_to_service(url: str, payload: dict) -> Any:
    return await _request_service("post", url, payload)


async def _put_to_service(url: str, payload: dict) -> Any:
    return await _request_service("put", url, payload)


async def _get_from_service(url: str) -> Any:
    return await _request_service("get", url)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
