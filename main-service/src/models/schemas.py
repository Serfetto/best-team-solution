import enum
from pydantic import BaseModel, ConfigDict, EmailStr, RootModel, Field, field_validator, model_validator
from typing import List, Optional, Any, Literal, Union
from datetime import datetime, timezone
from uuid import UUID

class UserBase(BaseModel):
    id: UUID
    email: EmailStr
    is_active: bool
    is_admin: bool 
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BatchItem(BaseModel):
    id: Union[int, str]
    text: str

class BatchAnalysisRequest(BaseModel):
    data: List[BatchItem]

class InsightsItem(BaseModel):
    products: List[str]

class ProductSummaryRequest(BaseModel):
    product_name: str
    date_from: Optional[str] = None
    date_to: Optional[str] = None

class ProductAllDescriptionRequest(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None

class ProductSummaryResponse(BaseModel):
    product_name: str
    strengths: Optional[List[List[str]]]
    weaknesses: Optional[List[List[str]]]
    strengths_summary: str
    weaknesses_summary: str

class PredictModel(BaseModel):
    id: int
    topics: List[str]
    sentiments: List[str]
    model_config = ConfigDict(from_attributes=True)

class PredictResponse(BaseModel):
    predictions: List[PredictModel]
    model_config = ConfigDict(from_attributes=True)

class AnalysisRequest(BaseModel):
    date_from: datetime
    date_to: datetime
    mod: Literal['base', 'research', 'research_guided']
    save_excel: bool = True
    preview_rows: int = 5

class FullRequest(BaseModel):
    pages_per_run: Optional[int] = Field(default=None, ge=1)
    start_page: Optional[int] = Field(default=None, ge=1)

class IncrementalRequest(BaseModel):
    max_pages: Optional[int] = Field(default=None, ge=1)

class Banki_and_Sravni_Full_Result(BaseModel):
    processed_pages: Optional[int]
    added_reviews: Optional[int]
    last_page: Optional[int]
    has_more: Optional[bool]

class Banki_and_Sravni_Incremental_Result(BaseModel):
    added_reviews: Optional[int]
    reached_existing: Optional[bool]

class ProductDescriptionRequest(BaseModel):
    product_name: str
    date_from: Optional[str] = None
    date_to: Optional[str] = None

class ProductDescriptionResponse(BaseModel):
    product_name: str
    description: Optional[str] = None
    examples: Optional[List[str]] = None

class ProductAllDescriptionResponse(RootModel):
    root: List[ProductDescriptionResponse] 

class ProductDescriptionSummaryResponse(BaseModel):
    product_name: str
    description: Optional[str] = None
    examples: Optional[List[str]] = None
    strengths: Optional[List[List[str]]] = None
    weaknesses: Optional[List[List[str]]] = None
    strengths_summary: Optional[str] = None
    weaknesses_summary: Optional[str] = None

class ProductAllDescriptionSummaryResponse(RootModel):
    root: List[ProductDescriptionSummaryResponse] 




class Frequency(str, enum.Enum):
    real = "real"
    day = "day"
    week = "week"
    month = "month"
    year = "year"

class ParsingStartRequest(BaseModel):
    source: Literal["full", "banki", "sravni"]
    mode: Literal["incremental"]
    every: Optional[Literal['None', 'real', 'day', 'week', 'month', 'year']] = Field(default=None)

    @field_validator("every", mode="before")
    @classmethod
    def normalize_every(cls, v):
        if v in (None, "None", b""):
            return None
        if isinstance(v, Frequency):
            return v
        if isinstance(v, str):
            return Frequency(v.strip().lower())
        raise TypeError("every must be str or Frequency")


class TaxonomyUpdate(BaseModel):
    items: List[str]

    @field_validator("items", mode="before")
    @classmethod
    def _normalize_items(cls, value):
        if isinstance(value, str):
            value = [value]
        elif isinstance(value, set):
            value = list(value)
        if not isinstance(value, list):
            raise TypeError("items must be a list of strings")
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned



class TaxonomyShufflePayload(BaseModel):
    add: List[str] = Field(default_factory=list)
    delete: List[str] = Field(default_factory=list)

    @field_validator("add", "delete", mode="before")
    @classmethod
    def _normalize_lists(cls, value):
        if value is None or value == "":
            return []
        if isinstance(value, str):
            value = [value]
        elif isinstance(value, set):
            value = list(value)
        if not isinstance(value, list):
            raise TypeError("value must be a list of strings")
        return [str(item).strip() for item in value if str(item).strip()]

class TaxonomyEditPayload(BaseModel):
    show: List[str] = Field(default_factory=list)
    hide: List[str] = Field(default_factory=list)

    @field_validator("show", "hide", mode="before")
    @classmethod
    def _normalize_lists(cls, value):
        if value is None or value == "":
            return []
        if isinstance(value, str):
            value = [value]
        elif isinstance(value, set):
            value = list(value)
        if not isinstance(value, list):
            raise TypeError("value must be a list of strings")
        return [str(item).strip() for item in value if str(item).strip()]

class UserAuth(BaseModel):
    email: EmailStr
    password: str


