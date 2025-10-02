import uuid
from fastapi import Depends, HTTPException, Request, status
from src.models.crud import get_user_by_id, get_user_by_email
from datetime import datetime, timedelta, timezone
from src.configs.config import settings, pwd_context
from jose import jwt, JWTError
from src.models.database import db_dependency
from src.models.schemas import UserBase


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_token(data: dict, expire_time: timedelta):
    to_encode = data.copy()
    to_encode.update({"exp": datetime.now(timezone.utc) + expire_time})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def authenticate_user(email: str, password: str, db: db_dependency):
    user = await get_user_by_email(email, db)
    if not user or verify_password(plain_password=password, hashed_password=user.hashed_password) is False:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Не правильно введен логин или пароль"
        )
    return user

def get_access_token(request: Request):
    return request.cookies.get('at') 

async def get_current_user(token: str = Depends(get_access_token), db: db_dependency = db_dependency):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, 
        detail="Не удалось аутентифицироваться",
    )
    if token is None:
        raise credentials_exception
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        
        expire = payload.get('exp')
        if datetime.fromtimestamp(int(expire), tz=timezone.utc) < datetime.now(timezone.utc):
            raise credentials_exception
        
        user_id = payload.get('sub')
        if not user_id:
            return credentials_exception
    except JWTError as e:
        raise credentials_exception
    
    user = await get_user_by_id(id=uuid.UUID(user_id), db=db)
    if user is None:
        return credentials_exception
    return UserBase.model_validate(user)

async def get_current_active_user(current_user: UserBase = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=404, detail="Inactive user")
    return current_user

async def get_current_admin_user(current_user: UserBase = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user