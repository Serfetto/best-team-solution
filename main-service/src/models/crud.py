from datetime import datetime
from functools import wraps
import math
import uuid
from sqlalchemy import and_, delete, func, select, update
from src.models.schemas import UserBase
from src.models.models import Reviews, User
from src.models.database import db_dependency
from src.configs.config import settings

async def get_user_by_email(email: str, db: db_dependency):
    db_user = await db.execute(select(User).where(User.email == email))
    await db.close()
    result = db_user.scalar_one_or_none()
    return result

async def get_user_by_id(id: uuid, db: db_dependency):
    stmt = select(User).where(User.id == id)
    
    db_user = await db.execute(statement=stmt)
    await db.close()
    result = db_user.scalar_one_or_none()
    return result

async def create_user(email: str, hashed_password: str, is_active: bool, db: db_dependency):
    db_user = User(email=email, hashed_password=hashed_password, is_active=is_active)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    await db.close()
    return db_user

async def update_user(user_id: uuid, user: UserBase, db: db_dependency):
    stmt = update(User).where(User.id == user_id).values(**user.dict())
    result = await db.execute(stmt)
    await db.commit()
    await db.close()
    return result

async def change_active(user_id: uuid, active: bool, db: db_dependency):
    stmt = update(User).where(User.id == user_id).values(is_active=active)
    result = await db.execute(stmt)
    await db.commit()
    await db.close()
    return result
