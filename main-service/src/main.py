from contextlib import asynccontextmanager
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.models.database import Base, engine
from fastadmin import fastapi_app as admin_app
from fastadmin.settings import settings as admin_settings
from src.configs.config import ADMIN_PARAMS, settings
from src.auth.router import router as auth_router
from src.analysis.router import router as analysis_router
from src.taxonomy.router import router as taxonomy_router
from src.parsing.router import router as parsing_router
from src.export.router import router as export_router
from src.admin.router import UserAdmin
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.asyncio import AsyncIOExecutor
from pytz import timezone as tz

scheduler: AsyncIOScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # База и fastadmin как у тебя
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    for key in dir(admin_settings):
        if key.isupper() and key in ADMIN_PARAMS:
            setattr(admin_settings, key, ADMIN_PARAMS[key])

    #Планировщик
    global scheduler
    scheduler = AsyncIOScheduler(timezone=tz(getattr(settings, "TIMEZONE", "UTC")))
    scheduler.start()
    app.state.scheduler = scheduler

    try:
        yield
    finally:
        scheduler.shutdown(wait=False)

    yield


app = FastAPI(title="Main", lifespan=lifespan)

templates = Jinja2Templates(directory="src/templates")

app.mount("/static", StaticFiles(directory=settings.BASE_DIR / "static"), name="static")
app.mount("/admin", admin_app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.HOSTS,
    allow_methods=settings.METHODS,
    allow_headers=settings.HEADERS,
    allow_credentials=settings.CREDENTIALS,
    expose_headers=["Content-Security-Policy"],
)

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(analysis_router, prefix="/analysis", tags=["analysis"])
app.include_router(taxonomy_router, prefix="/taxonomy", tags=["taxonomy"])
app.include_router(parsing_router, prefix="/parsing", tags=["parsing"])
app.include_router(export_router, prefix="/export", tags=["export"])

# @app.get("/")
# def root(request: Request):
#     return RedirectResponse(settings.DASHBORD_URL)

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
