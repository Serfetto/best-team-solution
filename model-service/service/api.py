from __future__ import annotations

import os
import json
import time
import ast
import logging
import re
import tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
)

LLM_LOGGER_NAME = 'reviews.pipeline.llm'
_llm_logger = logging.getLogger(LLM_LOGGER_NAME)
_file_handler = None
for _existing in _llm_logger.handlers:
    if isinstance(_existing, logging.FileHandler):
        _file_handler = _existing
        break
if _file_handler is None:
    _file_handler = logging.FileHandler('pipeline-debug.log', encoding='utf-8')
    _file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s'))
    _file_handler.setLevel(logging.DEBUG)
    _llm_logger.addHandler(_file_handler)
_llm_logger.setLevel(logging.DEBUG)
_llm_logger.propagate = False

pipeline_logger = logging.getLogger('reviews.pipeline')
pipeline_logger.setLevel(logging.INFO)
pipeline_logger.propagate = False
if not any(handler is _file_handler for handler in pipeline_logger.handlers):
    pipeline_logger.addHandler(_file_handler)

for _name in ("httpx", "httpcore"):
    _http_logger = logging.getLogger(_name)
    _http_logger.setLevel(logging.WARNING)
    _http_logger.propagate = False

# Ensure project root is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local imports
import run_reviews as rr
import reviews.pipeline as pipeline


load_dotenv()

logger = logging.getLogger(__name__)

tags_metadata = [
    {
        "name": "analysis",
        "description": "Запуск анализа данных и получение агрегированного отчёта",
    },
    {
        "name": "taxonomy",
        "description": "Чтение и изменение таксономии (категорий продуктов)",
    },
    {
        "name": "export",
        "description": "Выгрузка артефактов анализа (XLSX, JSON)",
    },
]

app = FastAPI(
    title="Reviews Microservice",
    version="1.1.0",
    description=(
        "Сервис для обогащения отзывов и стандартизации продуктовых упоминаний.\n\n"
        "Swagger UI доступен по /docs, Redoc — по /redoc, OpenAPI JSON — по /openapi.json."
    ),
    openapi_tags=tags_metadata,
    contact={
        "name": "Reviews Team",
        "url": "http://127.0.0.1:8000/docs",
    },
)

DATA_DIR = Path(os.getenv("REVIEWS_DATA_DIR", ROOT / "data"))
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = PROCESSED_DIR / "reports"
TAXONOMY_PATH = Path(os.getenv("REVIEWS_TAXONOMY_PATH", ROOT / "reviews" / "taxonomy.json"))
ENRICHED_COMBINED_PATH = DATA_DIR / "enriched_combined.csv"
ENRICHED_PRODUCTS_PATH = DATA_DIR / "enriched_products_combined.csv"
TAXONOMY_BASE_PATH = DATA_DIR / "taxonomy_base.json"
TAXONOMY_API_PATH = DATA_DIR / "taxonomy_api.json"
TAXONOMY_FULL_PATH = DATA_DIR / "taxonomy_full.json"


SENTIMENT_RU_MAP = {
    "positive": "положительно",
    "negative": "отрицательно",
    "neutral": "нейтрально"
}

UNKNOWN_TOPIC = "Не определено"
UNKNOWN_SENTIMENT = "не определено"
QUOTE_CHARS = '"\''
_QUOTE_TRANSLATION = str.maketrans({char: "" for char in QUOTE_CHARS})

class BatchItem(BaseModel):
    id: Union[int, str]
    text: str


class BatchAnalysisRequest(BaseModel):
    data: List[BatchItem] = Field(default_factory=list)


class BatchPrediction(BaseModel):
    id: Union[int, str]
    topics: List[str]
    sentiments: List[str]


class BatchMetrics(BaseModel):

    duration_seconds: float

    processed: int

    llm_calls: int

    llm_failures: int

    cost_total: float





class BatchAnalysisResponse(BaseModel):

    predictions: List[BatchPrediction] = Field(default_factory=list)

    metrics: BatchMetrics







class ProcessResponse(BaseModel):
    new_reviews: int = 0
    processed_reviews: int = 0
    processed_products: int = 0
    combined_total: int = 0
    products_total: int = 0
    combined_path: str
    products_path: str
    elapsed_seconds: float = 0.0
    message: str = ""



class ExtractRequest(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None

    @field_validator("date_from", "date_to", mode="before")
    @classmethod
    def _clean_date(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class ExtractResponse(BaseModel):
    processed_rows: int = 0
    mapped_rows: int = 0
    added_categories: int = 0
    base_size: int = 0
    full_size: int = 0
    elapsed_seconds: float = 0.0
    message: str = ""
    was_processed: bool = False
    taxonomy_full_path: str
    taxonomy_base_path: Optional[str] = None



class ProductSummaryRequest(BaseModel):
    product_name: str
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class ProductDescriptionRequest(BaseModel):
    product_name: str
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class ProductDescriptionResponse(BaseModel):
    product_name: str
    description: str
    examples: List[str] = Field(default_factory=list)
    llm_cost: float = 0.0
    total_mentions: int = 0


class ProductSummaryResponse(BaseModel):
    product_name: str
    strengths: List[List[str]] = Field(default_factory=list)
    weaknesses: List[List[str]] = Field(default_factory=list)
    strengths_summary: str
    weaknesses_summary: str


class ProductListResponse(BaseModel):
    products: List[str] = Field(default_factory=list)


class TaxonomyEditRequest(BaseModel):
    show: List[str] = Field(default_factory=list)
    hide: List[str] = Field(default_factory=list)

    @field_validator("show", "hide", mode="before")
    @classmethod
    def _normalize_lists(cls, value: Any) -> List[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            result: List[str] = []
            for item in value:
                if item is None:
                    continue
                result.append(str(item))
            return result
        raise TypeError("show and hide must be provided as lists of strings")


class TaxonomyEditResponse(BaseModel):
    items: List[str]
    cleared_rows: int
    updated_rows: int = 0
    taxonomy_path: str
    products_path: str

class AllProductsRequest(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    was_processed: bool = False
    was_date_changed: bool = False

    @field_validator("date_from", "date_to", mode="before")
    @classmethod
    def _strip_dates(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value


class ProductDescriptionBatch(BaseModel):
    product: str
    description: str
    examples: List[str] = Field(default_factory=list)
    llm_cost: float = 0.0
    total_mentions: int = 0


class AllProductDescriptionResponse(BaseModel):
    processed: int
    items: List[ProductDescriptionBatch] = Field(default_factory=list)
    total_cost: float = 0.0
    elapsed_seconds: float = 0.0
    taxonomy_full_path: str


class ProductSummaryBatch(BaseModel):
    product: str
    strengths_summary: str
    weaknesses_summary: str
    strengths_examples: List[List[str]] = Field(default_factory=list)
    weaknesses_examples: List[List[str]] = Field(default_factory=list)


class AllProductSummaryResponse(BaseModel):
    processed: int
    items: List[ProductSummaryBatch] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    taxonomy_full_path: str


    products: List[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    date_from: Optional[str] = Field(None, description="ISO date inclusive start, e.g. 2025-01-01")
    date_to: Optional[str] = Field(None, description="ISO date inclusive end, e.g. 2025-01-31")
    mode: Optional[str] = Field(
        None,
        description="Analysis mode: base | research | research_guided",
    )
    save_excel: bool = Field(True, description="Save Excel outputs in processed dir")
    preview_rows: int = Field(5, ge=0, le=50)

    @field_validator("date_from", "date_to", mode="before")
    @classmethod
    def _nonempty(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped or None
        return v


class AnalyzeResponse(BaseModel):
    report: Dict[str, Any]
    report_path: str
    processed_dir: str


class TaxonomyUpdate(BaseModel):
    items: List[str] = Field(..., description="List of taxonomy category names (strings)")

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, v: List[str]) -> List[str]:
        if v is None:
            raise ValueError("taxonomy must contain at least one non-empty item")
        if not isinstance(v, (list, tuple)):
            raise TypeError("taxonomy must be provided as a list of strings")
        cleaned = [str(x).strip() for x in v if str(x).strip()]
        if not cleaned:
            raise ValueError("taxonomy must contain at least one non-empty item")
        return cleaned



def _load_llm_config() -> pipeline.LLMConfig:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")
    model_reviews = os.getenv("OPENROUTER_MODEL_REVIEWS") or os.getenv("OPENROUTER_MODEL", rr.DEFAULT_MODEL)
    model_mapping = os.getenv("OPENROUTER_MODEL_MAPPING") or model_reviews
    model_summary = os.getenv("OPENROUTER_MODEL_SUMMARY") or model_reviews
    base_url = os.getenv("OPENROUTER_BASE_URL", pipeline.OPENROUTER_URL)
    return pipeline.LLMConfig(
        api_key=api_key,
        model=model_reviews,
        base_url=base_url,
        mapping_model=model_mapping,
        summary_model=model_summary,
    )


def _set_env_date_filter(date_from: Optional[str], date_to: Optional[str]) -> None:
    if date_from:
        os.environ["REVIEWS_DATE_FROM"] = date_from
    else:
        os.environ.pop("REVIEWS_DATE_FROM", None)
    if date_to:
        os.environ["REVIEWS_DATE_TO"] = date_to
    else:
        os.environ.pop("REVIEWS_DATE_TO", None)


def _load_latest_report() -> Optional[Path]:
    if not REPORT_DIR.exists():
        return None
    candidates = sorted(REPORT_DIR.glob("run_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_or_init_csv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        empty = pd.DataFrame(columns=columns)
        empty.to_csv(path, index=False, encoding="utf-8")
        return empty.copy()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read {path.name}: {exc}")
    for col in columns:
        if col not in df.columns:
            df[col] = None
    ordered = list(columns) + [c for c in df.columns if c not in columns]
    return df[ordered]


def _align_dataframe_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in columns:
        if col not in aligned.columns:
            aligned[col] = None
    for col in ("product_extract", "product"):
        if col in aligned.columns:
            aligned[col] = aligned[col].astype(object)
    ordered = list(columns) + [c for c in aligned.columns if c not in columns]
    return aligned[ordered]


def _load_taxonomy_full_dict() -> Dict[str, Any]:
    if TAXONOMY_FULL_PATH.exists():
        try:
            data = json.loads(TAXONOMY_FULL_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read taxonomy_full.json: {exc}")
        if not isinstance(data, dict):
            raise HTTPException(status_code=500, detail="taxonomy_full.json must contain a JSON object")
        return data
    return {}


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text_value = str(value).strip()
    if not text_value:
        return None
    cleaned = text_value.translate(_QUOTE_TRANSLATION).strip()
    return cleaned or None


def _load_api_taxonomy() -> Tuple[str, ...]:
    try:
        if TAXONOMY_API_PATH.exists():
            data = json.loads(TAXONOMY_API_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                cleaned = [str(item).strip() for item in data if str(item).strip()]
                if cleaned:
                    taxonomy_tuple = tuple(dict.fromkeys(cleaned))
                    pipeline.PRODUCT_TAXONOMY = taxonomy_tuple
                    pipeline.PRODUCT_LOOKUP = {name.lower(): name for name in taxonomy_tuple}
                    return taxonomy_tuple
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read taxonomy_base.json: {exc}")
    taxonomy_tuple = pipeline.PRODUCT_TAXONOMY
    if isinstance(taxonomy_tuple, tuple):
        pipeline.PRODUCT_LOOKUP = {name.lower(): name for name in taxonomy_tuple}
    return taxonomy_tuple


def _reload_taxonomy_from_file(path: Path) -> None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            items = tuple(str(x).strip() for x in data if str(x).strip())
            if items:
                pipeline.PRODUCT_TAXONOMY = items  # type: ignore[attr-defined]
                pipeline.PRODUCT_LOOKUP = {name.lower(): name for name in items}  # type: ignore[attr-defined]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to reload taxonomy: {exc}")


def _parse_list_field(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, float):
        try:
            import math
            if math.isnan(value):
                return []
        except Exception:
            pass
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
        parts = [part.strip() for part in re.split(r"[;,|]", value) if part.strip()]
        if parts:
            return parts
        return [value]
    value_str = str(value).strip()
    return [value_str] if value_str else []



def _strip_product_quotes(df: pd.DataFrame, *, string_columns: Sequence[str], list_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Return a copy of df with quotes removed from specified columns."""
    if df.empty:
        return df

    cleaned = df.copy()

    def _clean_string(value: Any) -> Any:
        if isinstance(value, str):
            return value.translate(_QUOTE_TRANSLATION)
        return value

    def _clean_sequence(value: Any) -> Any:
        if isinstance(value, list):
            return [_clean_string(item) for item in value]
        if isinstance(value, tuple):
            return [_clean_string(item) for item in value]
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except Exception:
                return _clean_string(value)
            if isinstance(parsed, list):
                return [_clean_string(item) for item in parsed]
        return value

    for column in string_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].apply(_clean_string)

    for column in list_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].apply(_clean_sequence)

    return cleaned


def _safe_isoformat(value: Any) -> Optional[str]:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def _normalize_llm_items(raw: Any) -> List[str]:
    items: List[str] = []
    if isinstance(raw, list):
        for value in raw:
            if isinstance(value, str):
                candidate = value.strip()
            elif isinstance(value, dict):
                candidate = str(value.get("text", "")).strip()
            else:
                candidate = str(value).strip()
            candidate = candidate.strip().lstrip("-*•–—· 	")
            if candidate:
                items.append(candidate)
    elif isinstance(raw, str):
        for chunk in re.split(r"[\n;,]+", raw):
            candidate = chunk.strip().lstrip("-*•–—· 	")
            if candidate:
                items.append(candidate)
    seen: Set[str] = set()
    normalized: List[str] = []
    for item in items:
        if item not in seen:
            normalized.append(item)
            seen.add(item)
    return normalized


def _extract_json_dict(text_value: str) -> Dict[str, Any]:
    if not text_value:
        return {}
    try:
        parsed = json.loads(text_value)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", text_value, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


COMBINED_DATA_PATH = ENRICHED_COMBINED_PATH
COMBINED_PRODUCTS_DATA_PATH = ENRICHED_PRODUCTS_PATH


def _load_combined_dataframe() -> pd.DataFrame:
    if not COMBINED_DATA_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Combined CSV not found: {COMBINED_DATA_PATH}")
    try:
        df = pd.read_csv(COMBINED_DATA_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read combined CSV: {exc}")
    if "posted_at" in df.columns:
        df["posted_at_dt"] = pd.to_datetime(df["posted_at"], errors="coerce")
    else:
        df["posted_at_dt"] = pd.NaT
    df.sort_values("posted_at_dt", ascending=False, inplace=True)
    return df

def _load_products_dataframe() -> pd.DataFrame:
    path = ENRICHED_PRODUCTS_PATH
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Products CSV not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read products CSV: {exc}")

    if "product" not in df.columns:
        raise HTTPException(status_code=500, detail="Products CSV missing mandatory 'product' column")

    if "posted_at" in df.columns:
        df["posted_at_dt"] = pd.to_datetime(df["posted_at"], errors="coerce")
    else:
        df["posted_at_dt"] = pd.NaT
    df["product_norm"] = df["product"].astype(str).str.strip()
    df.sort_values("posted_at_dt", ascending=False, inplace=True)
    return df






WORD_LIMIT = 5_000  # per bucket (strengths/weaknesses)
EMPTY_STRENGTH_MESSAGE = "??? ?????? ??? ??????????? ??????? ??????."
EMPTY_WEAKNESS_MESSAGE = "??? ?????? ??? ??????????? ?????? ??????."



JOINT_SUMMARY_PROMPT = """Analyze customer feedback for banking product "{product}".

Strengths: {strengths}
Weaknesses: {weaknesses}

Tasks:
1. Create clean lists of strengths and weaknesses, merging similar feedback and making statements concise.
2. Resolve contradictions: if the same aspect appears in both categories, keep it only where it appears more frequently.
3. Write brief summaries (2-4 sentences) for strengths and weaknesses in professional Russian style.

Return strictly valid JSON:
{{
  "strengths_summary": "summary of strengths in Russian",
  "weaknesses_summary": "summary of weaknesses in Russian"
}}

Use only the provided data, do not add facts."""



def _generate_joint_summary(
    client: pipeline.OpenAI,
    model: str,
    product: str,
    strengths: Dict[str, Any],
    weaknesses: Dict[str, Any],
) -> Dict[str, Any]:

    prompt = JOINT_SUMMARY_PROMPT.format(
        product=product,
        strengths=strengths,
        weaknesses=weaknesses
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a banking products analyst. Analyze customer feedback professionally and objectively. Always respond in Russian."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        timeout=120,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content if response.choices else ""
    data = _extract_json_dict(content)

    strengths_summary = (data.get("strengths_summary") or "").strip()
    weaknesses_summary = (data.get("weaknesses_summary") or "").strip()

    return {
        "strengths_summary": strengths_summary or EMPTY_STRENGTH_MESSAGE,
        "weaknesses_summary": weaknesses_summary or EMPTY_WEAKNESS_MESSAGE,
    }

@app.get(
    "/insights/products",
    response_model=ProductListResponse,
    tags=["analysis"],
    summary="List products available for insights",
)
def list_insights_products() -> ProductListResponse:
    df = _load_products_dataframe()

    products = df['product_norm'].unique().tolist()

    return ProductListResponse(products=products)


@app.post(
    "/insights/product-description",
    response_model=ProductDescriptionResponse,
    tags=["analysis"],
    summary="Generate product description",
)
def product_description(request: ProductDescriptionRequest) -> ProductDescriptionResponse:
    product_name = (request.product_name or "").strip()
    if not product_name:
        raise HTTPException(status_code=400, detail="product_name is required")

    df = _load_products_dataframe()

    date_from_dt = None
    date_to_dt = None
    if request.date_from:
        try:
            date_from_dt = pd.to_datetime(request.date_from)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date_from format, expected YYYY-MM-DD")
    if request.date_to:
        try:
            date_to_dt = pd.to_datetime(request.date_to)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date_to format, expected YYYY-MM-DD")

    if date_from_dt is not None:
        df = df[df["posted_at_dt"] >= date_from_dt].copy()
    if date_to_dt is not None:
        df = df[df["posted_at_dt"] <= date_to_dt].copy()

    if "product_norm" in df.columns:
        df = df[df["product_norm"].fillna("") == product_name].copy()
    else:
        df = df[df["product"].astype(str).str.strip() == product_name].copy()

    if df.empty:
        return ProductDescriptionResponse(product_name=product_name, description="", examples=[], llm_cost=0.0, total_mentions=0)

    if "product_raw" not in df.columns:
        raise HTTPException(status_code=500, detail="Products Excel missing 'product_raw' column")

    raw_values = df["product_raw"].tolist()
    mentions: List[str] = []
    seen: Set[str] = set()
    for value in raw_values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            candidates = [str(item).strip() for item in value if str(item).strip()]
        else:
            text = str(value).strip()
            candidates = [text] if text else []
        for item in candidates:
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            mentions.append(item)

    total_mentions = len(mentions)
    if not mentions:
        return ProductDescriptionResponse(product_name=product_name, description="", examples=[], llm_cost=0.0, total_mentions=0)

    llm_config = _load_llm_config()
    description, examples, cost = pipeline.describe_product_from_mentions(product_name, mentions, llm_config)

    return ProductDescriptionResponse(
        product_name=product_name,
        description=description,
        examples=examples,
        llm_cost=cost,
        total_mentions=total_mentions,
    )


@app.post(
    "/insights/product-summary",
    response_model=ProductSummaryResponse,
    tags=["analysis"],
    summary="Summarize strengths and weaknesses for a product",
)
def product_summary(request: ProductSummaryRequest) -> ProductSummaryResponse:
    product_name = (request.product_name or "").strip()
    if not product_name:
        raise HTTPException(status_code=400, detail="product_name is required")

    df = _load_products_dataframe()

    date_from = None
    date_to = None
    if request.date_from:
        try:
            date_from = pd.to_datetime(request.date_from)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date_from format, expected YYYY-MM-DD")
    if request.date_to:
        try:
            date_to = pd.to_datetime(request.date_to)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date_to format, expected YYYY-MM-DD")

    if date_from is not None:
        df = df[df["posted_at_dt"] >= date_from].copy()
    if date_to is not None:
        df = df[df["posted_at_dt"] <= date_to].copy()

    if "product_norm" in df.columns:
        df = df[df["product_norm"].fillna("") == product_name].copy()
    else:
        df = df[df["product"].fillna("") == product_name].copy()

    df.sort_values("posted_at_dt", ascending=False, inplace=True)

    def flatten_to_semicolon_string(data):
        return '; '.join(item for sublist in data for item in sublist if item)

    strengths, weaknesses = [], []
    strengths.extend(map(_parse_list_field, df['strengths'].tolist()))
    weaknesses.extend(map(_parse_list_field, df['weaknesses'].tolist()))
    strengths_str = flatten_to_semicolon_string(strengths)
    weaknesses_str = flatten_to_semicolon_string(weaknesses)

    llm_config = _load_llm_config()
    client = pipeline.OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)

    summary_model = llm_config.summary_model or llm_config.model

    try:
        summaries = _generate_joint_summary(client, summary_model, product_name, strengths_str, weaknesses_str)
    except Exception:
        logger.exception("Failed to generate product summary for %s", product_name)
        summaries = {
            "strengths": [],
            "weaknesses": [],
            "strengths_summary": EMPTY_STRENGTH_MESSAGE,
            "weaknesses_summary": EMPTY_WEAKNESS_MESSAGE,
        }

    strengths_summary = summaries.get("strengths_summary") or EMPTY_STRENGTH_MESSAGE
    weaknesses_summary = summaries.get("weaknesses_summary") or EMPTY_WEAKNESS_MESSAGE

    return ProductSummaryResponse(
        product_name=product_name,
        strengths=strengths,
        weaknesses=weaknesses,
        strengths_summary=strengths_summary,
        weaknesses_summary=weaknesses_summary,
    )




@app.post(
    "/insights/all-product-description",
    response_model=AllProductDescriptionResponse,
    tags=["analysis"],
    summary="Generate descriptions for all products",
)
def all_product_description(request: AllProductsRequest) -> AllProductDescriptionResponse:
    start = time.perf_counter()

    date_from_dt = None
    date_to_dt = None
    if request.date_from:
        try:
            date_from_dt = pd.to_datetime(request.date_from)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date_from format, expected YYYY-MM-DD")
    if request.date_to:
        try:
            date_to_dt = pd.to_datetime(request.date_to)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date_to format, expected YYYY-MM-DD")

    if not (request.was_processed or request.was_date_changed):
        elapsed = float(time.perf_counter() - start)
        return AllProductDescriptionResponse(
            processed=0,
            items=[],
            total_cost=0.0,
            elapsed_seconds=elapsed,
            taxonomy_full_path=str(TAXONOMY_FULL_PATH),
        )

    df = _load_products_dataframe()

    if date_from_dt is not None:
        df = df[df["posted_at_dt"] >= date_from_dt].copy()
    if date_to_dt is not None:
        df = df[df["posted_at_dt"] <= date_to_dt].copy()

    if "product_extract" not in df.columns:
        raise HTTPException(status_code=500, detail="Products Excel missing 'product_extract' column")
    if "product_raw" not in df.columns:
        raise HTTPException(status_code=500, detail="Products Excel missing 'product_raw' column")

    def _normalize_raw_values(series: pd.Series) -> List[str]:
        seen: Set[str] = set()
        result: List[str] = []
        for value in series.tolist():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                candidates = [str(item).strip() for item in value if str(item).strip()]
            else:
                text = str(value).strip()
                candidates = [text] if text else []
            for item in candidates:
                lowered = item.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                result.append(item)
        return result

    grouped_entries: List[Tuple[str, List[str]]] = []
    for product_name, group in df.groupby("product_extract"):
        if product_name is None:
            continue
        try:
            if pd.isna(product_name):
                continue
        except Exception:
            pass
        product_clean = str(product_name).strip()
        if not product_clean:
            continue
        mentions = _normalize_raw_values(group["product_raw"])
        if not mentions:
            continue
        grouped_entries.append((product_clean, mentions))

    if not grouped_entries:
        elapsed_empty = float(time.perf_counter() - start)
        return AllProductDescriptionResponse(
            processed=0,
            items=[],
            total_cost=0.0,
            elapsed_seconds=elapsed_empty,
            taxonomy_full_path=str(TAXONOMY_FULL_PATH),
        )

    llm_config = _load_llm_config()
    workers = max(1, int(os.getenv("REVIEWS_WORKERS", str(pipeline.DEFAULT_WORKERS))))

    def _run_description(payload: Tuple[str, List[str]]) -> Tuple[str, str, List[str], float, int]:
        product_name, mentions = payload
        try:
            description, examples, cost = pipeline.describe_product_from_mentions(product_name, mentions, llm_config)
        except Exception:
            logger.exception("Failed to generate description for %s", product_name)
            description = ""
            examples = mentions[: min(20, len(mentions))]
            cost = 0.0
        if not examples:
            examples = mentions[: min(20, len(mentions))]
        return product_name, description, examples, float(cost or 0.0), len(mentions)

    results: List[Tuple[str, str, List[str], float, int]] = []
    total_cost = 0.0
    grouped_entries.sort(key=lambda item: item[0].lower())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_description, entry): entry[0] for entry in grouped_entries}
        for fut in as_completed(futures):
            try:
                product_name, description, examples, cost, total_mentions = fut.result()
                results.append((product_name, description, examples, cost, total_mentions))
                total_cost += cost
            except Exception as exc:
                logger.exception("Description worker failed: %s", exc)

    results.sort(key=lambda item: item[0].lower())

    taxonomy_full = _load_taxonomy_full_dict()
    for product_name, description, examples, _, _ in results:
        entry = taxonomy_full.setdefault(product_name, {})
        entry["description"] = description
        entry["description_examples"] = examples

    try:
        TAXONOMY_FULL_PATH.parent.mkdir(parents=True, exist_ok=True)
        TAXONOMY_FULL_PATH.write_text(json.dumps(taxonomy_full, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write taxonomy_full.json: {exc}")

    elapsed = float(time.perf_counter() - start)
    response_items = [
        ProductDescriptionBatch(
            product=product_name,
            description=description,
            examples=examples,
            llm_cost=cost,
            total_mentions=total_mentions,
        )
        for product_name, description, examples, cost, total_mentions in results
    ]

    return AllProductDescriptionResponse(
        processed=len(response_items),
        items=response_items,
        total_cost=total_cost,
        elapsed_seconds=elapsed,
        taxonomy_full_path=str(TAXONOMY_FULL_PATH),
    )


@app.post(
    "/insights/all-product-summary",
    response_model=AllProductSummaryResponse,
    tags=["analysis"],
    summary="Generate summaries for all products",
)
def all_product_summary(request: AllProductsRequest) -> AllProductSummaryResponse:
    start = time.perf_counter()

    date_from_dt = None
    date_to_dt = None
    if request.date_from:
        try:
            date_from_dt = pd.to_datetime(request.date_from)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date_from format, expected YYYY-MM-DD")
    if request.date_to:
        try:
            date_to_dt = pd.to_datetime(request.date_to)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date_to format, expected YYYY-MM-DD")

    if not (request.was_processed or request.was_date_changed):
        elapsed = float(time.perf_counter() - start)
        return AllProductSummaryResponse(
            processed=0,
            items=[],
            elapsed_seconds=elapsed,
            taxonomy_full_path=str(TAXONOMY_FULL_PATH),
        )

    df = _load_products_dataframe()

    if date_from_dt is not None:
        df = df[df["posted_at_dt"] >= date_from_dt].copy()
    if date_to_dt is not None:
        df = df[df["posted_at_dt"] <= date_to_dt].copy()

    for column in ["product_extract", "strengths", "weaknesses"]:
        if column not in df.columns:
            raise HTTPException(status_code=500, detail=f"Products Excel missing '{column}' column")

    def _collect_lists(values: List[Any]) -> List[List[str]]:
        collected: List[List[str]] = []
        for value in values:
            items = _parse_list_field(value)
            if items:
                collected.append(items)
        return collected

    def _flatten_to_string(groups: List[List[str]]) -> str:
        flat: List[str] = []
        for group in groups:
            flat.extend([item for item in group if item])
        return "; ".join(flat)

    grouped_entries: List[Tuple[str, List[List[str]], List[List[str]]]] = []
    for product_name, group in df.groupby("product_extract"):
        if product_name is None:
            continue
        try:
            if pd.isna(product_name):
                continue
        except Exception:
            pass
        product_clean = str(product_name).strip()
        if not product_clean:
            continue
        strengths_groups = _collect_lists(group["strengths"].tolist())
        weaknesses_groups = _collect_lists(group["weaknesses"].tolist())
        if not strengths_groups and not weaknesses_groups:
            continue
        grouped_entries.append((product_clean, strengths_groups, weaknesses_groups))

    if not grouped_entries:
        elapsed_empty = float(time.perf_counter() - start)
        return AllProductSummaryResponse(
            processed=0,
            items=[],
            elapsed_seconds=elapsed_empty,
            taxonomy_full_path=str(TAXONOMY_FULL_PATH),
        )

    llm_config = _load_llm_config()
    summary_model = llm_config.summary_model or llm_config.model
    workers = max(1, int(os.getenv("REVIEWS_WORKERS", str(pipeline.DEFAULT_WORKERS))))

    def _run_summary(payload: Tuple[str, List[List[str]], List[List[str]]]) -> Tuple[str, str, str, List[List[str]], List[List[str]]]:
        product_name, strengths_groups, weaknesses_groups = payload
        strengths_text = _flatten_to_string(strengths_groups)
        weaknesses_text = _flatten_to_string(weaknesses_groups)
        client = pipeline.OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
        try:
            summaries = _generate_joint_summary(client, summary_model, product_name, strengths_text, weaknesses_text)
        except Exception:
            logger.exception("Failed to generate summary for %s", product_name)
            summaries = {
                "strengths_summary": EMPTY_STRENGTH_MESSAGE,
                "weaknesses_summary": EMPTY_WEAKNESS_MESSAGE,
            }
        strengths_summary = summaries.get("strengths_summary") or EMPTY_STRENGTH_MESSAGE
        weaknesses_summary = summaries.get("weaknesses_summary") or EMPTY_WEAKNESS_MESSAGE
        return product_name, strengths_summary, weaknesses_summary, strengths_groups, weaknesses_groups

    results: List[Tuple[str, str, str, List[List[str]], List[List[str]]]] = []
    grouped_entries.sort(key=lambda item: item[0].lower())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_summary, entry): entry[0] for entry in grouped_entries}
        for fut in as_completed(futures):
            try:
                result = fut.result()
                results.append(result)
            except Exception as exc:
                logger.exception("Summary worker failed: %s", exc)

    results.sort(key=lambda item: item[0].lower())

    taxonomy_full = _load_taxonomy_full_dict()
    for product_name, strengths_summary, weaknesses_summary, strengths_groups, weaknesses_groups in results:
        entry = taxonomy_full.setdefault(product_name, {})
        entry["strengths_summary"] = strengths_summary
        entry["weaknesses_summary"] = weaknesses_summary
        entry["strengths_examples"] = strengths_groups
        entry["weaknesses_examples"] = weaknesses_groups

    try:
        TAXONOMY_FULL_PATH.parent.mkdir(parents=True, exist_ok=True)
        TAXONOMY_FULL_PATH.write_text(json.dumps(taxonomy_full, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write taxonomy_full.json: {exc}")

    elapsed = float(time.perf_counter() - start)
    response_items = [
        ProductSummaryBatch(
            product=product_name,
            strengths_summary=strengths_summary,
            weaknesses_summary=weaknesses_summary,
            strengths_examples=strengths_groups,
            weaknesses_examples=weaknesses_groups,
        )
        for product_name, strengths_summary, weaknesses_summary, strengths_groups, weaknesses_groups in results
    ]

    return AllProductSummaryResponse(
        processed=len(response_items),
        items=response_items,
        elapsed_seconds=elapsed,
        taxonomy_full_path=str(TAXONOMY_FULL_PATH),
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["analysis"],
    summary="Запустить анализ",
    description=(
        "Запускает анализ за указанный период.\n\n"
        "Режимы: base — с таксономией; research — свободная форма; research_guided — свободная форма с выравниванием к известной таксономии на этапе маппинга."
    ),
)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    # Configure runtime flags
    if req.mode:
        os.environ["REVIEWS_ANALYSIS_MODE"] = req.mode
    else:
        os.environ.pop("REVIEWS_ANALYSIS_MODE", None)
    _set_env_date_filter(req.date_from, req.date_to)

    # Run pipeline
    rr.run_reviews(save_excel=req.save_excel, preview_rows=req.preview_rows, configure_logging=False)

    # Load the latest JSON report
    report_path = _load_latest_report()
    if not report_path or not report_path.exists():
        raise HTTPException(status_code=500, detail="Run report not found")
    try:
        with open(report_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read report: {exc}")

    return AnalyzeResponse(report=report, report_path=str(report_path), processed_dir=str(PROCESSED_DIR))



@app.post(
    "/analyze/batch",
    tags=["analysis"],
    summary="Batch analyze reviews using base taxonomy",
)
def analyze_batch(request: BatchAnalysisRequest) -> Dict[str, Any]:
    if not request.data:
        return {"predictions": []}

    llm_config = _load_llm_config()
    client = pipeline.OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)

    taxonomy_tuple = _load_api_taxonomy()
    taxonomy_lookup = {item.lower(): item for item in taxonomy_tuple} if taxonomy_tuple else {}

    predictions: List[Optional[BatchPrediction]] = [None] * len(request.data)

    try:
        workers = max(1, int(os.getenv("REVIEWS_WORKERS", str(pipeline.DEFAULT_WORKERS))))
    except Exception:
        workers = pipeline.DEFAULT_WORKERS

    def _normalize_topic(value: Any) -> Optional[str]:
        clean = _normalize_text(value)
        if not clean:
            return None
        lowered = clean.lower()
        if taxonomy_lookup:
            return taxonomy_lookup.get(lowered)
        return clean

    def _process(index_and_item: Tuple[int, BatchItem]):
        idx, item = index_and_item
        text_raw = (item.text or "").strip()
        if not text_raw:
            prediction = BatchPrediction(id=item.id, topics=[UNKNOWN_TOPIC], sentiments=[UNKNOWN_SENTIMENT])
            return idx, prediction

        payload = {
            "dataset": "batch",
            "id": item.id,
            "title": None,
            "text_clean": pipeline.html_to_text(text_raw),
            "text_raw": text_raw,
            "rating": None,
            "agent_answer_text": "",
            "user_name": None,
        }

        try:
            llm_data, _ = pipeline._call_llm(client, llm_config.model, payload, taxonomy_tuple or None)
        except Exception:
            llm_data = None

        normalized = pipeline._normalize_enrichment(llm_data, None, taxonomy=taxonomy_tuple or None)

        raw_topics = normalized.get("product_list") or []
        topics: List[str] = []
        seen_topics: Set[str] = set()
        for raw_topic in raw_topics:
            normalized_topic = _normalize_topic(raw_topic)
            if not normalized_topic:
                continue
            display = normalized_topic
            if display not in seen_topics:
                seen_topics.add(display)
                topics.append(display)
        if not topics:
            normalized_topic = _normalize_topic(normalized.get("product"))
            if normalized_topic:
                topics = [normalized_topic]
        if not topics:
            topics = [UNKNOWN_TOPIC]

        raw_sentiments = normalized.get("sentiment_list") or []
        sentiments_en: List[str] = [
            str(s).strip().lower() for s in raw_sentiments if isinstance(s, str) and str(s).strip()
        ]
        if not sentiments_en:
            sentiment = normalized.get("sentiment")
            if isinstance(sentiment, str) and sentiment.strip():
                sentiments_en = [sentiment.strip().lower()]
        if not sentiments_en:
            sentiments_en = ["neutral"] * len(topics)
        if len(sentiments_en) < len(topics):
            sentiments_en.extend([sentiments_en[-1]] * (len(topics) - len(sentiments_en)))

        sentiments_ru = [SENTIMENT_RU_MAP.get(s, UNKNOWN_SENTIMENT) for s in sentiments_en[: len(topics)]]
        prediction = BatchPrediction(id=item.id, topics=topics, sentiments=sentiments_ru)
        return idx, prediction

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process, entry) for entry in enumerate(request.data)]
        for fut in as_completed(futures):
            idx, prediction = fut.result()
            predictions[idx] = prediction

    ordered_predictions = [p for p in predictions if p is not None]
    output = []
    for prediction in ordered_predictions:
        output.append({
            "id": prediction.id,
            "topics": prediction.topics,
            "sentiments": prediction.sentiments,
        })

    return {"predictions": output}

@app.post(
    "/extract",
    response_model=ExtractResponse,
    tags=["analysis"],
    summary="Extract taxonomy from raw products",
    description="Map raw product names to clean taxonomy using optional date filters.",
)
def extract_products(request: ExtractRequest) -> ExtractResponse:
    start = time.perf_counter()

    products_df = _load_or_init_csv(ENRICHED_PRODUCTS_PATH, pipeline.PRODUCT_COLUMNS)
    products_df = _strip_product_quotes(products_df, string_columns=("product", "product_extract"))
    for column in ("product_extract", "product"):
        if column not in products_df.columns:
            products_df[column] = None
        products_df[column] = products_df[column].astype(object)

    def _parse_date(value: Optional[str]) -> Optional[datetime]:
        if value is None:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {value}. Use YYYY-MM-DD")

    date_from_obj = _parse_date(request.date_from)
    date_to_obj = _parse_date(request.date_to)

    if "posted_at" not in products_df.columns:
        raise HTTPException(status_code=500, detail="Column 'posted_at' missing in enriched products file")

    posted_dt = pd.to_datetime(products_df["posted_at"], errors="coerce")
    mask = pd.Series(True, index=products_df.index)
    if date_from_obj is not None:
        mask &= posted_dt >= pd.Timestamp(date_from_obj)
    if date_to_obj is not None:
        mask &= posted_dt <= pd.Timestamp(date_to_obj) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    filtered = products_df.loc[mask].copy()
    if filtered.empty:
        elapsed_empty = float(time.perf_counter() - start)
        return ExtractResponse(
            processed_rows=0,
            mapped_rows=0,
            added_categories=0,
            base_size=0,
            full_size=0,
            elapsed_seconds=elapsed_empty,
            message="No products found in the selected date range",
            was_processed=False,
            taxonomy_full_path=str(TAXONOMY_FULL_PATH),
            taxonomy_base_path=str(TAXONOMY_BASE_PATH),
        )

    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        try:
            if pd.isna(value):
                return True
        except Exception:
            pass
        if isinstance(value, str) and not value.strip():
            return True
        return False

    if "product" not in filtered.columns:
        filtered["product"] = None

    missing_mask = filtered["product"].apply(_is_missing)
    pending_df = filtered.loc[missing_mask].copy()
    if pending_df.empty:
        elapsed_none = float(time.perf_counter() - start)
        return ExtractResponse(
            processed_rows=0,
            mapped_rows=0,
            added_categories=0,
            base_size=0,
            full_size=0,
            elapsed_seconds=elapsed_none,
            message="All products already mapped in the selected date range",
            was_processed=False,
            taxonomy_full_path=str(TAXONOMY_FULL_PATH),
            taxonomy_base_path=str(TAXONOMY_BASE_PATH),
        )

    if "product_raw" not in pending_df.columns:
        raise HTTPException(status_code=500, detail="Column 'product_raw' missing in enriched products file")

    raw_candidates = pipeline._extract_unique_products(pending_df["product_raw"].tolist())
    if not raw_candidates:
        elapsed_zero = float(time.perf_counter() - start)
        return ExtractResponse(
            processed_rows=int(len(pending_df)),
            mapped_rows=0,
            added_categories=0,
            base_size=0,
            full_size=0,
            elapsed_seconds=elapsed_zero,
            message="No raw product values available for mapping",
            was_processed=False,
            taxonomy_full_path=str(TAXONOMY_FULL_PATH),
            taxonomy_base_path=str(TAXONOMY_BASE_PATH),
        )

    base_path = TAXONOMY_BASE_PATH
    full_path = TAXONOMY_FULL_PATH

    try:
        if base_path.exists():
            base_data = json.loads(base_path.read_text(encoding="utf-8"))
            if base_data is None:
                base_data = []
            if not isinstance(base_data, list):
                raise ValueError("Base taxonomy must be a JSON list")
        else:
            base_data = []
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read taxonomy_base.json: {exc}")

    def _clean_taxonomy(values: Sequence[Any]) -> List[str]:
        cleaned: List[str] = []
        seen_items: Set[str] = set()
        for item in values:
            text = _normalize_text(item)
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen_items:
                continue
            seen_items.add(lowered)
            cleaned.append(text)
        return cleaned

    base_taxonomy = _clean_taxonomy(base_data)

    try:
        if full_path.exists():
            full_data_raw = json.loads(full_path.read_text(encoding="utf-8"))
            if isinstance(full_data_raw, dict):
                full_values = list(full_data_raw.keys())
            elif isinstance(full_data_raw, list):
                full_values = full_data_raw
            elif full_data_raw is None:
                full_values = []
            else:
                raise ValueError("taxonomy_full.json must contain a JSON object or list")
        else:
            full_values = []
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read taxonomy_full.json: {exc}")

    initial_taxonomy = _clean_taxonomy(full_values)

    new_taxonomy: List[str] = list(initial_taxonomy)
    seen_categories: Dict[str, str] = {item.lower(): item for item in new_taxonomy}

    llm_config = _load_llm_config()
    client = pipeline.OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
    mapping_model = llm_config.mapping_model or llm_config.model

    mapping: Dict[str, str] = {}
    mapping_batches = 0
    total_cost = 0.0
    seen_products: Set[str] = set()

    for batch in pipeline._chunked(raw_candidates, max(1, pipeline.GUIDED_BATCH_SIZE)):
        to_map: List[str] = []
        for value in batch:
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen_products:
                continue
            seen_products.add(lowered)
            to_map.append(text)
        if not to_map:
            continue
        mapping_batches += 1
        batch_mapping, batch_cost = pipeline._standardize_products_with_llm(
            client,
            mapping_model,
            to_map,
            known_categories=new_taxonomy,
        )
        if batch_mapping:
            for orig, target in batch_mapping.items():
                if not isinstance(orig, str) or not isinstance(target, str):
                    continue
                orig_clean = orig.strip()
                target_clean = target.strip()
                if not orig_clean or not target_clean:
                    continue
                target_normalized = _normalize_text(target_clean) or target_clean
                if not target_normalized:
                    continue
                mapping[orig_clean] = target_normalized
                lowered_target = target_normalized.lower()
                if lowered_target not in seen_categories:
                    seen_categories[lowered_target] = target_normalized
                    new_taxonomy.append(target_normalized)
        if batch_cost is not None:
            try:
                total_cost += float(batch_cost)
            except Exception:
                pass

    mapping_lookup: Dict[str, str] = {}
    for key_raw, value_raw in mapping.items():
        key = str(key_raw).strip().lower()
        if not key:
            continue
        if isinstance(value_raw, str):
            value = _normalize_text(value_raw) or value_raw.strip()
        else:
            value = _normalize_text(value_raw) or str(value_raw).strip()
        if not value:
            continue
        mapping_lookup[key] = value

    def _map_value(raw_value: Any) -> Optional[str]:
        if raw_value is None:
            return None
        try:
            if pd.isna(raw_value):
                return None
        except Exception:
            pass
        text = str(raw_value).strip()
        if not text:
            return None
        lowered = text.lower()
        result = mapping_lookup.get(lowered)
        if result:
            return result
        return None
    processed_indices = pending_df.index
    mapped_pending = pending_df["product_raw"].apply(_map_value)

    # Retry once for only unmapped raw values, as an extra LLM batch
    try:
        pending_total = int(len(pending_df))
        initially_mapped = int(mapped_pending.notna().sum())
    except Exception:
        pending_total = len(pending_df)
        initially_mapped = mapped_pending.notna().sum()

    if pending_total > initially_mapped:
        retry_values = pending_df.loc[mapped_pending.isna(), "product_raw"].tolist()
        retry_candidates = pipeline._extract_unique_products(retry_values)
        if retry_candidates:
            for retry_batch in pipeline._chunked(retry_candidates, max(1, pipeline.GUIDED_BATCH_SIZE)):
                if not retry_batch:
                    continue
                mapping_batches += 1
                retry_mapping, retry_cost = pipeline._standardize_products_with_llm(
                    client,
                    mapping_model,
                    retry_batch,
                    known_categories=new_taxonomy,
                )
                if retry_mapping:
                    for orig, target in retry_mapping.items():
                        if not isinstance(orig, str) or not isinstance(target, str):
                            continue
                        orig_clean = orig.strip()
                        target_clean = target.strip()
                        if not orig_clean or not target_clean:
                            continue
                        target_normalized = _normalize_text(target_clean) or target_clean
                        if not target_normalized:
                            continue
                        mapping[orig_clean] = target_normalized
                        lowered_target = target_normalized.lower()
                        if lowered_target not in seen_categories:
                            seen_categories[lowered_target] = target_normalized
                            new_taxonomy.append(target_normalized)
                if retry_cost is not None:
                    try:
                        total_cost += float(retry_cost)
                    except Exception:
                        pass
            # Rebuild lookup and recompute mapped_pending after retry
            mapping_lookup = {}
            for key_raw, value_raw in mapping.items():
                key = str(key_raw).strip().lower()
                if not key:
                    continue
                if isinstance(value_raw, str):
                    value = _normalize_text(value_raw) or value_raw.strip()
                else:
                    value = _normalize_text(value_raw) or str(value_raw).strip()
                if not value:
                    continue
                mapping_lookup[key] = value
            mapped_pending = pending_df["product_raw"].apply(_map_value)

    # Apply mapping results only to the pending subset
    mapped_rows = int(mapped_pending.notna().sum())
    mapped_non_null = mapped_pending.dropna()
    if not mapped_non_null.empty:
        normalized_series = mapped_non_null.apply(lambda v: _normalize_text(v) or str(v).strip())
        products_df.loc[normalized_series.index, "product_extract"] = normalized_series

    # Refresh taxonomy based on existing product_extract values
    final_taxonomy: List[str] = list(new_taxonomy)
    seen_final: Set[str] = {item.lower() for item in final_taxonomy}
    for value in products_df["product_extract"].tolist():
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if not text:
            continue
        normalized_value = _normalize_text(text) or text
        lowered = normalized_value.lower()
        if lowered in seen_final:
            continue
        seen_final.add(lowered)
        final_taxonomy.append(normalized_value)

    base_size_before = len(base_taxonomy)
    base_updated_size = base_size_before

    if not base_taxonomy:
        not_missing_extract_mask = products_df["product_extract"].apply(lambda v: not _is_missing(v))
        products_df.loc[not_missing_extract_mask, "product"] = products_df.loc[not_missing_extract_mask, "product_extract"]
        base_updated_size = len(final_taxonomy)
        try:
            base_path.parent.mkdir(parents=True, exist_ok=True)
            base_path.write_text(json.dumps(final_taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to write taxonomy_base.json: {exc}")
    else:
        base_lookup = {item.lower(): item for item in base_taxonomy}
        product_missing_mask = products_df["product"].apply(_is_missing)
        for idx in products_df.index[product_missing_mask]:
            extract_value = products_df.at[idx, "product_extract"]
            if _is_missing(extract_value):
                continue
            text = str(extract_value).strip()
            lowered = text.lower()
            if lowered in base_lookup:
                products_df.at[idx, "product"] = base_lookup[lowered]

        extract_lower = products_df["product_extract"].apply(
            lambda v: str(v).strip().lower() if not _is_missing(v) and str(v).strip() else None
        )
        known_categories = set(base_lookup.keys())
        non_base_mask = extract_lower.notna() & ~extract_lower.isin(list(known_categories))
        if non_base_mask.any():
            products_df.loc[non_base_mask, "product"] = "другой"

    # Within selected range, mark any unresolved items as "другой"
    for idx in processed_indices:
        if _is_missing(products_df.at[idx, "product"]) and _is_missing(products_df.at[idx, "product_extract"]):
            products_df.at[idx, "product"] = "другой"


    taxonomy_full_dict = {name: {} for name in final_taxonomy if isinstance(name, str) and name.strip()}
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(json.dumps(taxonomy_full_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write taxonomy_full.json: {exc}")

    try:
        products_df = _strip_product_quotes(products_df, string_columns=("product", "product_extract"))
        products_df.to_csv(ENRICHED_PRODUCTS_PATH, index=False, encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update enriched products file: {exc}")

    elapsed = float(time.perf_counter() - start)
    processed_count = int(len(processed_indices))
    message_parts = [
        f"Processed {processed_count} rows",
        f"mapped {mapped_rows} values",
    ]
    if mapping_batches:
        message_parts.append(f"LLM batches: {mapping_batches}")
    message = "; ".join(message_parts)

    return ExtractResponse(
        processed_rows=processed_count,
        mapped_rows=mapped_rows,
        added_categories=max(0, len(final_taxonomy) - base_size_before),
        base_size=base_updated_size,
        full_size=len(final_taxonomy),
        elapsed_seconds=elapsed,
        message=message,
        was_processed=True,
        taxonomy_full_path=str(full_path),
        taxonomy_base_path=str(base_path),
    )

@app.post(
    "/process",
    response_model=ProcessResponse,
    tags=["analysis"],
    summary="Process new reviews",
    description="Process only reviews whose d_id values are missing in enriched outputs.",
)
def process_new_reviews() -> ProcessResponse:
    start = time.perf_counter()
    combined_df = _load_or_init_csv(ENRICHED_COMBINED_PATH, pipeline.REVIEW_COLUMNS)
    products_df = _load_or_init_csv(ENRICHED_PRODUCTS_PATH, pipeline.PRODUCT_COLUMNS)

    def _clean_identifier(value: Any) -> Optional[str]:
        if value is None:
            return None
        value_str = str(value).strip()
        if not value_str or value_str.lower() == "nan":
            return None
        return value_str

    existing_ids: Set[str] = set()
    if "d_id" in combined_df.columns:
        for value in combined_df["d_id"].tolist():
            cleaned = _clean_identifier(value)
            if cleaned:
                existing_ids.add(cleaned)

    sources: List[Tuple[str, Path]] = [
        ("banki.ru", DATA_DIR / "banki_ru_full.csv"),
        ("sravni.ru", DATA_DIR / "sravni_ru_full.csv"),
    ]

    datasets_to_process: List[Tuple[str, pd.DataFrame]] = []
    total_candidates = 0
    for dataset_name, csv_path in sources:
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {csv_path}")
        try:
            df_src = pd.read_csv(csv_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read {csv_path.name}: {exc}")
        if "d_id" not in df_src.columns:
            raise HTTPException(status_code=500, detail=f"Column d_id missing in {csv_path.name}")
        df_src["d_id"] = df_src["d_id"].apply(_clean_identifier)
        df_src = df_src[df_src["d_id"].notna()]
        new_df = df_src[~df_src["d_id"].isin(existing_ids)].copy()
        total_candidates += int(len(new_df))
        if not new_df.empty:
            datasets_to_process.append((dataset_name, new_df))

    if not datasets_to_process:
        elapsed = float(time.perf_counter() - start)
        return ProcessResponse(
            new_reviews=0,
            processed_reviews=0,
            processed_products=0,
            combined_total=int(len(combined_df)),
            products_total=int(len(products_df)),
            combined_path=str(ENRICHED_COMBINED_PATH),
            products_path=str(ENRICHED_PRODUCTS_PATH),
            elapsed_seconds=elapsed,
            message="No new reviews found",
        )

    llm_config = _load_llm_config()
    prev_mode = os.environ.get("REVIEWS_ANALYSIS_MODE")
    prev_date_from = os.environ.get("REVIEWS_DATE_FROM")
    prev_date_to = os.environ.get("REVIEWS_DATE_TO")
    os.environ["REVIEWS_ANALYSIS_MODE"] = "research"
    _set_env_date_filter(None, None)

    combined_new = pd.DataFrame(columns=pipeline.REVIEW_COLUMNS)
    results: Dict[str, pipeline.DatasetResult] = {}

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_specs: List[pipeline.DatasetSpec] = []
            tmp_root = Path(tmp_dir)
            for dataset_name, frame in datasets_to_process:
                safe_name = dataset_name.replace("/", "_").replace(".", "_")
                tmp_path = tmp_root / f"{safe_name}.csv"
                frame.to_csv(tmp_path, index=False)
                dataset_specs.append(pipeline.DatasetSpec(name=dataset_name, path=str(tmp_path)))
            combined_new, results = rr.run_reviews(
                datasets=dataset_specs,
                llm_config=llm_config,
                data_dir=DATA_DIR,
                save_excel=False,
                preview_rows=0,
                configure_logging=False,
            )
    finally:
        if prev_mode is None:
            os.environ.pop("REVIEWS_ANALYSIS_MODE", None)
        else:
            os.environ["REVIEWS_ANALYSIS_MODE"] = prev_mode
        _set_env_date_filter(prev_date_from, prev_date_to)

    combined_existing = _align_dataframe_columns(combined_df, pipeline.REVIEW_COLUMNS)
    combined_existing = _strip_product_quotes(combined_existing, string_columns=("product",), list_columns=("product_list",))
    combined_existing["d_id"] = combined_existing["d_id"].apply(_clean_identifier)

    combined_new = _align_dataframe_columns(combined_new, pipeline.REVIEW_COLUMNS)
    combined_new = _strip_product_quotes(combined_new, string_columns=("product",), list_columns=("product_list",))
    combined_new["d_id"] = combined_new["d_id"].apply(_clean_identifier)

    processed_reviews = int(len(combined_new))
    if processed_reviews:
        combined_updated = pd.concat([combined_existing, combined_new], ignore_index=True)
        combined_updated = _align_dataframe_columns(combined_updated, pipeline.REVIEW_COLUMNS)
        combined_updated = _strip_product_quotes(combined_updated, string_columns=("product",), list_columns=("product_list",))
        combined_updated.to_csv(ENRICHED_COMBINED_PATH, index=False, encoding="utf-8")
    else:
        combined_updated = combined_existing

    combined_result = results.get("combined") if isinstance(results, dict) else None
    products_new = combined_result.products if combined_result else pd.DataFrame(columns=pipeline.PRODUCT_COLUMNS)
    products_existing = _align_dataframe_columns(products_df, pipeline.PRODUCT_COLUMNS)
    products_existing = _strip_product_quotes(products_existing, string_columns=("product", "product_extract"))
    products_existing["d_id"] = products_existing["d_id"].apply(_clean_identifier)
    products_new = _align_dataframe_columns(products_new, pipeline.PRODUCT_COLUMNS)
    products_new = _strip_product_quotes(products_new, string_columns=("product", "product_extract"))
    products_new["d_id"] = products_new["d_id"].apply(_clean_identifier)
    if "product_extract" in products_new.columns:
        products_new["product_extract"] = products_new["product_extract"].astype(object)

    processed_products = int(len(products_new))
    if processed_products:
        products_updated = pd.concat([products_existing, products_new], ignore_index=True)
        products_updated = _align_dataframe_columns(products_updated, pipeline.PRODUCT_COLUMNS)
        products_updated = _strip_product_quotes(products_updated, string_columns=("product", "product_extract"))
        products_updated.to_csv(ENRICHED_PRODUCTS_PATH, index=False, encoding="utf-8")
    else:
        products_updated = products_existing

    elapsed = float(time.perf_counter() - start)
    message = (
        f"Processed {processed_reviews} new reviews"
        if processed_reviews
        else "No new reviews processed"
    )
    return ProcessResponse(
        new_reviews=total_candidates,
        processed_reviews=processed_reviews,
        processed_products=processed_products,
        combined_total=int(len(combined_updated)),
        products_total=int(len(products_updated)),
        combined_path=str(ENRICHED_COMBINED_PATH),
        products_path=str(ENRICHED_PRODUCTS_PATH),
        elapsed_seconds=elapsed,
        message=message,
    )








@app.post(
    "/taxonomy/edit",
    response_model=TaxonomyEditResponse,
    tags=["taxonomy"],
    summary="Update taxonomy and optionally remap products",
)
def taxonomy_edit(request: TaxonomyEditRequest) -> TaxonomyEditResponse:
    base_path = TAXONOMY_BASE_PATH
    full_path = TAXONOMY_FULL_PATH
    products_path = ENRICHED_PRODUCTS_PATH

    def _clean_unique(values: Sequence[Any]) -> List[str]:
        cleaned: List[str] = []
        seen_lower: Set[str] = set()
        for value in values or []:
            text_value = _normalize_text(value)
            if not text_value:
                continue
            lowered = text_value.lower()
            if lowered in seen_lower:
                continue
            seen_lower.add(lowered)
            cleaned.append(text_value)
        return cleaned

    show_clean = _clean_unique(request.show)
    hide_clean = _clean_unique(request.hide)
    full_clean = _clean_unique(show_clean + hide_clean)

    try:
        if full_path.exists():
            full_raw = json.loads(full_path.read_text(encoding="utf-8"))
            if isinstance(full_raw, dict):
                existing_full_items = list(full_raw.keys())
            elif isinstance(full_raw, list):
                existing_full_items = full_raw
            elif full_raw is None:
                existing_full_items = []
            else:
                raise ValueError("taxonomy_full.json must contain a JSON object or list")
        else:
            existing_full_items = []
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read taxonomy_full.json: {exc}")

    existing_full_set: Set[str] = set()
    for value in existing_full_items:
        normalized = _normalize_text(value)
        if not normalized:
            continue
        existing_full_set.add(normalized.lower())

    full_set = {item.lower() for item in full_clean}
    full_matches = full_set == existing_full_set and len(full_set) == len(existing_full_set)

    if full_matches:
        cleaned_items = show_clean
        try:
            base_path.parent.mkdir(parents=True, exist_ok=True)
            base_path.write_text(json.dumps(cleaned_items, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to write taxonomy_base.json: {exc}")

        products_df = _load_or_init_csv(products_path, pipeline.PRODUCT_COLUMNS)
        products_df = _align_dataframe_columns(products_df, pipeline.PRODUCT_COLUMNS)
        products_df = _strip_product_quotes(products_df, string_columns=("product", "product_extract"))
        for column in ("product_extract", "product"):
            if column not in products_df.columns:
                products_df[column] = None
            products_df[column] = products_df[column].astype(object)

        base_lookup = {item.lower(): item for item in cleaned_items}
        extract_lower = products_df["product_extract"].apply(_normalize_text)
        extract_lower = extract_lower.apply(lambda v: v.lower() if v else None)

        products_df["product"] = None

        match_mask = extract_lower.notna() & extract_lower.isin(list(base_lookup.keys()))
        if match_mask.any():
            products_df.loc[match_mask, "product"] = extract_lower[match_mask].map(base_lookup)

        non_base_mask = extract_lower.notna() & ~extract_lower.isin(list(base_lookup.keys()))
        if non_base_mask.any():
            products_df.loc[non_base_mask, "product"] = "другой"

        try:
            products_df = _strip_product_quotes(products_df, string_columns=("product", "product_extract"))
            products_df.to_csv(products_path, index=False, encoding="utf-8")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to write enriched products file: {exc}")

        return TaxonomyEditResponse(
            items=cleaned_items,
            cleared_rows=0,
            updated_rows=int(len(products_df)),
            taxonomy_path=str(base_path),
            products_path=str(products_path),
        )

    cleaned_items = show_clean
    try:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps(cleaned_items, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write taxonomy_base.json: {exc}")

    full_dict = {name: {} for name in full_clean}
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(json.dumps(full_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write taxonomy_full.json: {exc}")

    products_df = _load_or_init_csv(products_path, pipeline.PRODUCT_COLUMNS)
    products_df = _align_dataframe_columns(products_df, pipeline.PRODUCT_COLUMNS)
    products_df = _strip_product_quotes(products_df, string_columns=("product", "product_extract"))
    for column in ("product_extract", "product"):
        if column not in products_df.columns:
            products_df[column] = None
        products_df[column] = products_df[column].astype(object)
        products_df[column] = pd.NA

    try:
        products_df = _strip_product_quotes(products_df, string_columns=("product", "product_extract"))
        products_df.to_csv(products_path, index=False, encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write enriched products file: {exc}")

    return TaxonomyEditResponse(
        items=cleaned_items,
        cleared_rows=int(len(products_df)),
        updated_rows=0,
        taxonomy_path=str(base_path),
        products_path=str(products_path),
    )

@app.get(
    "/taxonomy",
    response_model=Dict[str, Any],
    tags=["taxonomy"],
    summary="Получить таксономию",
)
def get_taxonomy() -> Dict[str, Any]:
    try:
        if not TAXONOMY_PATH.exists():
            raise FileNotFoundError(str(TAXONOMY_PATH))
        with open(TAXONOMY_PATH, "r", encoding="utf-8") as fh:
            items = json.load(fh)
        if not isinstance(items, list):
            raise ValueError("taxonomy file must be a JSON array")
        return {"path": str(TAXONOMY_PATH), "items": items}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load taxonomy: {exc}")


@app.put(
    "/taxonomy",
    response_model=Dict[str, Any],
    tags=["taxonomy"],
    summary="Обновить таксономию",
)
def update_taxonomy(update: TaxonomyUpdate) -> Dict[str, Any]:
    try:
        TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TAXONOMY_PATH, "w", encoding="utf-8") as fh:
            json.dump(update.items, fh, ensure_ascii=False, indent=2)
        # Hot-reload into pipeline module so next analyze uses updated taxonomy
        _reload_taxonomy_from_file(TAXONOMY_PATH)
        return {"path": str(TAXONOMY_PATH), "items": update.items}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update taxonomy: {exc}")


class ReportPath(BaseModel):
    report_path: str
    processed_dir: str


@app.get(
    "/report",
    response_model=ReportPath,
    tags=["analysis"],
    summary="Путь к последнему отчёту",
)
def get_report_path() -> ReportPath:
    path = _load_latest_report()
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return ReportPath(report_path=str(path), processed_dir=str(PROCESSED_DIR))


def _combined_csv_path() -> Path:
    return ENRICHED_COMBINED_PATH


@app.get("/export/combined", tags=["export"], summary="?????? ??? CSV")
def export_combined() -> FileResponse:
    path = _combined_csv_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Combined CSV not found. Run analysis first.")
    return FileResponse(
        path,
        media_type="text/csv",
        filename=path.name,
    )



@app.get("/export/dataset/{name}", tags=["export"], summary="?????? CSV ?? ??????")
def export_dataset(name: str) -> FileResponse:
    safe = name.replace("/", "_").replace(" ", "_")
    path = PROCESSED_DIR / f"enriched_{safe}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset CSV not found: {path.name}")
    return FileResponse(
        path,
        media_type="text/csv",
        filename=path.name,
    )



@app.get(
    "/export/derived-taxonomy",
    tags=["export"],
    summary="Скачать новую таксономию (research)",
)
def export_derived_taxonomy() -> FileResponse:
    path = PROCESSED_DIR / "derived_taxonomy_combined.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Derived taxonomy not found. Run research/research_guided first.")
    return FileResponse(
        path,
        media_type="application/json",
        filename=path.name,
    )


@app.get("/", tags=["analysis"], summary="Root")
def root() -> Dict[str, str]:
    return {
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "analyze": "/analyze",
        "taxonomy_get": "/taxonomy",
        "taxonomy_put": "/taxonomy",
        "export_combined": "/export/combined",
        "export_derived_taxonomy": "/export/derived-taxonomy",
        "report": "/report",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("service.api:app", host="0.0.0.0", port=8002, reload=False)

