from pydantic_settings import BaseSettings, SettingsConfigDict
from passlib.context import CryptContext
from pathlib import Path, PosixPath, WindowsPath
from typing import Any, Dict

class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parents[1]
    # SQLite
    DATABASE_SQLITE: str = ""
    SQLITE_DB_NAME: str = ""
    
    #Url parsing
    PARSING_BANKI_FULL: str
    PARSING_BANKI_INCREMENTAL: str
    PARSING_SRAVNI_FULL: str
    PARSING_SRAVNI_INCREMENTAL: str

    #Url LLM model
    ANALIZE_BATCH_URL_MODEL: str
    ANALIZE_URL_MODEL: str
    REPORT_URL_MODEL: str
    INSIGHTS_PRODUCT_SUMMARY_URL_MODEL: str
    INSIGHTS_PRODUCTS_DESCRIPTION_URL_MODEL: str
    TAXONOMY_URL_MODEL: str
    EXPORT_COMBINED_URL_MODEL: str
    EXPORT_DATASET_URL_MODEL: str
    EXPORT_DERIVED_TAXONOMY_URL_MODEL: str

    Extract_URL_MODEL: str
    Process_URL_MODEL: str
    INSIGHTS_all_product_description_URL_MODEL: str
    INSIGHTS_all_product_summary_URL_MODEL: str
    TAXONOMY_EDIT_URL_MODEL: str
    TAXONOMY_SHUFFLE_URL_MODEL: str

    #Url dashboard
    DASHBORD_URL: str
    
    #Data
    PROCESSED_DATA_DIR: Path = Path(__file__).resolve().parents[3] / "model-service" / "data"
    TAXONOMY_FILE_PATH: Path = Path(__file__).resolve().parents[3] / "model-service" / "reviews" / "taxonomy.json"


    # Auth
    ALGORITHM: str = "HS256"
    SECRET_KEY: str = "1"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 0
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 0

    # CORS
    HOSTS: list = ["*"]
    METHODS: list = ["*"]
    HEADERS: list = ["*"]
    CREDENTIALS: bool = False
    LIMIT_ITEMS_PER_PAGE: int = 0

    TELEGRAM_BOT_NAME: str  = ""
    TELEGRAM_BOT_TOKEN: str  = ""
    TELEGRAM_LOGIN_REDIRECT_URL: str  = ""
    ADMIN_USER_MODEL: str = "User"
    ADMIN_USER_MODEL_USERNAME_FIELD: str = "email"
    ADMIN_SECRET_KEY: str = "secret"
    

    model_config = SettingsConfigDict(env_file="src/configs/.env")

settings = Settings()

ADMIN_PARAMS = {
    "ADMIN_SECRET_KEY": settings.ADMIN_SECRET_KEY,
    "ADMIN_USER_MODEL": settings.ADMIN_USER_MODEL,
    "ADMIN_USER_MODEL_USERNAME_FIELD": settings.ADMIN_USER_MODEL_USERNAME_FIELD
}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")