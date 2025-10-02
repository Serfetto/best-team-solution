from datetime import datetime
import uuid
from fastapi import HTTPException
from src.configs.config import pwd_context
from sqlalchemy import and_, select, update
from src.models.database import async_session_maker
from fastadmin import SqlAlchemyModelAdmin, register
from src.models.models import User
from src.models.schemas import UserBase


@register(User, sqlalchemy_sessionmaker=async_session_maker)
class UserAdmin(SqlAlchemyModelAdmin):
    
    list_display = ("id", "email", "is_admin", "is_active", "created_at")
    list_display_links = ("id",)
    list_filter = ("id", "email", "is_admin", "is_active")
    search_fields = ("email",)
    readonly_fields = ("id", "created_at")
    # fieldsets = [
    #     (
    #         "Основная информация",
    #         {
    #             "fields": ["email", "hashed_password"]            
    #         }
    #     ),
    #     (
    #         "Настройки доступа",
    #         {
    #             "fields": ["is_admin", "is_active"]
    #         }
    #     )
    # ]

    async def authenticate(self, email: str, password: str) -> uuid.UUID | None:
        sessionmaker = self.get_sessionmaker()
        async with sessionmaker() as session:
            stmt = select(self.model_cls).where(and_(self.model_cls.email == email, self.model_cls.is_admin == True))
            db_history = await session.execute(statement=stmt)
            result = db_history.scalar_one_or_none()
            if not result or not pwd_context.verify(password, result.hashed_password):
                raise HTTPException(status_code=403, detail="Доступ запрещен")
            return result.id
    
    async def save_model(self, id: uuid.UUID | None, payload: dict) -> dict:
        if id:
            #Изменение пользователя
            created_at = datetime.strptime(payload["created_at"], "%Y-%m-%dT%H:%M:%S.%f")
            user_id = uuid.UUID(payload["id"])
            del payload["id"], payload["created_at"], payload["hashed_password"]
            sessionmaker = self.get_sessionmaker()
            async with sessionmaker() as session:
                query = update(self.model_cls).where(self.model_cls.id == user_id).values(**payload)
                await session.execute(query)
                await session.commit()
                await session.close()
            payload["id"] = user_id
            payload["created_at"] = created_at
            return UserBase.model_validate(payload)
        else:
            #Добавление нового пользователя
            sessionmaker = self.get_sessionmaker()
            async with sessionmaker() as session:
                is_active = payload.get("is_active", False)
                is_admin = payload.get("is_admin", False) 
                query = User(email=payload["email"], hashed_password=pwd_context.hash(payload["hashed_password"]), is_active=is_active, is_admin=is_admin)
                session.add(query)
                await session.commit()
                await session.refresh(query)
                await session.close()
            return UserBase.model_validate(query)