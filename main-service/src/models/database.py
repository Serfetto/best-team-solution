from typing import Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from src.configs.config import settings

#Sqlite
SQLITE_URL = f"{settings.DATABASE_SQLITE}:///{settings.SQLITE_DB_NAME}"
engine = create_async_engine(SQLITE_URL)#, echo=True)
async_session_maker = async_sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

async def get_db():
    db = async_session_maker()
    try:
        yield db
    finally:
        await db.close()
    
db_dependency = Annotated[AsyncSession, Depends(get_db)]
