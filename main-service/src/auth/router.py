from datetime import timedelta
import os
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from src.configs.config import settings
from src.models.crud import create_user, get_user_by_email
from src.auth.utils import authenticate_user, create_token, get_current_user, get_password_hash
from src.models.schemas import UserAuth, UserBase
from src.models.database import db_dependency

router = APIRouter()

@router.post("/login/")
async def login(user_data: UserAuth, db: db_dependency = db_dependency):
    user = await authenticate_user(email=user_data.email, password=user_data.password, db=db)
    print(os.getenv("ADMIN_USER_MODEL"))
    if user is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail='Неверная почта или пароль')
    access_token = create_token(data={"sub": str(user.id)}, expire_time=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    response = JSONResponse(
        content={"message": f"Пользователь {user.email} успешно ввешел в систему"},
        status_code=status.HTTP_200_OK,
    )
    response.set_cookie(
        key="at", 
        value=access_token, 
        httponly=True,
    )
    return response

@router.post("/signup/")
async def signup(info_user: UserAuth, db: db_dependency = db_dependency):
    print(settings.BASE_DIR) 
    db_user = await get_user_by_email(str(info_user.email), db)
    if db_user:
        raise HTTPException(status_code=400, detail="Почта уже зарегестрирована")

    hashed_password = get_password_hash(info_user.password)
    db_user = await create_user(info_user.email, hashed_password, True, db)
    user = UserBase.model_validate(db_user)
    access_token = create_token(data={"sub": str(user.id)}, expire_time=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    response = JSONResponse(
        content={
            "message": f"Вы успешно зарегистрированы {user.email}",
            "access_token": access_token
        },
        status_code=status.HTTP_201_CREATED,
    )
    response.set_cookie(
        key="at",
        value=access_token,
        httponly=True,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        expires=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
    return response

@router.post("/logout/")
async def logout_user(user: UserBase = Depends(get_current_user), db: db_dependency = db_dependency):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Пользователь не авторизован")
    response = JSONResponse(
        content={"message": f"Пользователь {user.email} успешно вышел из системы"},
        status_code=status.HTTP_200_OK
    )
    response.delete_cookie(key="at")
    return response

@router.get("/me")
async def get_me(user: UserBase = Depends(get_current_user)):
    return user