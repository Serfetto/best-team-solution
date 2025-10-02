"""Pipeline utilities for enriching financial reviews with LLM annotations."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from openai import OpenAI
from reviews.prompts import (
    render_review_prompt_taxonomy,
    render_review_prompt_freeform,
    render_mapping_prompt,
    render_taxonomy_description_prompt,
)
try:
    # Progress bar for record-level processing
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    # Fallback if tqdm is not installed; keep runtime functional
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

LOGGER = logging.getLogger(__name__)
LLM_LOGGER = logging.getLogger("reviews.pipeline.llm")

OPENROUTER_URL = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT = 60
MAX_RETRIES = 1
RETRY_BACKOFF_SEC = 3
DEFAULT_SOURCE = "unknown"
DEFAULT_WORKERS = 4
GUIDED_BATCH_SIZE = 100

VALID_SENTIMENTS = {"positive", "neutral", "negative"}

PRODUCT_TAXONOMY: Tuple[str, ...] = ()

PRODUCT_LOOKUP = {name.lower(): name for name in PRODUCT_TAXONOMY}

def _filter_by_date(df: pd.DataFrame, from_str: str, to_str: str) -> pd.DataFrame:
    """Filter dataframe by inclusive date range using column 'date'.

    - Converts both series and bounds to UTC to avoid tz-aware vs naive errors.
    - Builds an index-aligned boolean mask to avoid reindex warnings.
    """
    if "date" not in df.columns or (not from_str and not to_str):
        return df
    try:
        dates = pd.to_datetime(df["date"], utc=True, errors="coerce")
        mask = pd.Series(True, index=df.index)
        if from_str:
            start = pd.to_datetime(from_str, utc=True, errors="coerce")
            if pd.notna(start):
                mask &= dates >= start
        if to_str:
            end = pd.to_datetime(to_str, utc=True, errors="coerce")
            if pd.notna(end):
                mask &= dates <= end
        return df.loc[mask]
    except Exception as exc:
        LOGGER.warning("Failed to apply date filter: %s", exc)
        return df

# Optionally override taxonomy from a JSON file
try:
    _tx_path = os.getenv("REVIEWS_TAXONOMY_PATH", "").strip()
    if not _tx_path:
        _tx_path = str(Path(__file__).with_name("taxonomy.json"))
    _tx_p = Path(_tx_path)
    if _tx_p.exists():
        with open(_tx_p, "r", encoding="utf-8") as _fh:
            _tx_data = json.load(_fh)
        if isinstance(_tx_data, list):
            _items = [str(x).strip() for x in _tx_data if str(x).strip()]
            if _items:
                PRODUCT_TAXONOMY = tuple(_items)  # type: ignore
                PRODUCT_LOOKUP = {name.lower(): name for name in PRODUCT_TAXONOMY}  # type: ignore
except Exception as _exc:
    LOGGER.warning("Failed to override taxonomy from JSON: %s", _exc)

def _extract_cost(response: Any) -> Optional[float]:
    """Extract monetary cost from an OpenRouter/OpenAI response if present.

    Handles multiple shapes: plain numbers, numeric strings, nested dicts.
    """
    def _as_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            # Extract first numeric like 0.00123 from string "$0.00123 USD"
            m = re.search(r"[-+]?[0-9]*\.?[0-9]+", v.replace(",", "."))
            if m:
                try:
                    return float(m.group(0))
                except Exception:
                    return None
        if isinstance(v, dict):
            # Common nested keys
            for k in ("usd", "amount", "value", "total", "price"):
                if k in v:
                    f = _as_float(v[k])
                    if f is not None:
                        return f
        return None

    try:
        # 1) Attribute-style usage
        usage = getattr(response, "usage", None)
        if usage is not None:
            for key in ("cost", "total_cost", "total_cost_usd", "cost_usd"):
                val = getattr(usage, key, None)
                f = _as_float(val)
                if f is not None:
                    return f
            if hasattr(usage, "to_dict"):
                u = usage.to_dict()
                if isinstance(u, dict):
                    for key in ("cost", "total_cost", "total_cost_usd", "cost_usd"):
                        if key in u:
                            f = _as_float(u.get(key))
                            if f is not None:
                                return f
        # 1b) completion.usage.cost path (OpenRouter may nest usage under completion)
        completion = getattr(response, "completion", None)
        if completion is not None:
            comp_usage = getattr(completion, "usage", None)
            if comp_usage is not None:
                val = getattr(comp_usage, "cost", None)
                f = _as_float(val)
                if f is not None:
                    return f
                if hasattr(comp_usage, "to_dict"):
                    u2 = comp_usage.to_dict()
                    if isinstance(u2, dict):
                        f = _as_float(u2.get("cost"))
                        if f is not None:
                            return f
        # 2) Dict dump of entire response
        data = None
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        elif hasattr(response, "to_dict"):
            data = response.to_dict()
        if isinstance(data, dict):
            u = data.get("usage")
            if isinstance(u, dict):
                for key in ("cost", "total_cost", "total_cost_usd", "cost_usd"):
                    if key in u:
                        f = _as_float(u.get(key))
                        if f is not None:
                            return f
            # completion.usage.cost in dict form
            comp = data.get("completion")
            if isinstance(comp, dict):
                cu = comp.get("usage")
                if isinstance(cu, dict):
                    f = _as_float(cu.get("cost"))
                    if f is not None:
                        return f
            meta = data.get("meta")
            if isinstance(meta, dict):
                for key in ("cost", "total_cost"):
                    if key in meta:
                        f = _as_float(meta.get(key))
                        if f is not None:
                            return f
    except Exception:
        return None
    return None
@dataclass
class DatasetSpec:
    """Input dataset description."""

    name: str
    path: str
    limit: Optional[int] = None


@dataclass
class LLMConfig:
    """Configuration required to call OpenRouter-compatible models.

    model: used for per-review enrichment.
    mapping_model: optional model used for product mapping standardization; defaults to `model` if None.
    summary_model: optional model used for textual summaries; defaults to `model` if None.
    """

    api_key: str
    model: str
    base_url: str = OPENROUTER_URL
    mapping_model: Optional[str] = None
    summary_model: Optional[str] = None


@dataclass
class DatasetResult:
    """Holds processed reviews, product rows and accompanying metrics for a dataset."""

    name: str
    dataframe: pd.DataFrame
    products: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers without LLM
# ---------------------------------------------------------------------------
def html_to_text(html: Any) -> str:
    if html is None or (isinstance(html, float) and pd.isna(html)):
        return ""
    soup = BeautifulSoup(str(html), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\\n".join(lines)



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        parsed = dtparser.parse(str(value))
    except (ValueError, TypeError, OverflowError) as exc:
        LOGGER.debug("Failed to parse datetime %s: %s", value, exc)
        return None
    return parsed.isoformat()


def _normalize_rating(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _sentiment_from_rating(rating: Optional[float]) -> str:
    if rating is None:
        return "neutral"
    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return "neutral"


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        tokens = [token.strip() for token in re.split(r"[;,\n]+", value) if token.strip()]
        return tokens
    text = str(value).strip()
    return [text] if text else []

def _ensure_sequence(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, float) and pd.isna(value):
        return []
    return [value]

REVIEW_COLUMNS = [
    "dataset",
    "id",
    "d_id",
    "title",
    "text_clean",
    "posted_at",
    "grade_extracted",
    "product_raw_list",
    "product_list",
    "sentiment_list",
    "score_service_list",
    "score_tariffs_list",
    "score_reliability_list",
    "strengths_list",
    "weaknesses_list",
    "emotional_tags_list",
    "cost",
    "iter_seconds",
]

REVIEW_LIST_COLUMNS = [
    "product_raw_list",
    "product_list",
    "sentiment_list",
    "score_service_list",
    "score_tariffs_list",
    "score_reliability_list",
    "strengths_list",
    "weaknesses_list",
    "emotional_tags_list",
]

PRODUCT_COLUMNS = [
    "dataset",
    "d_id",
    "posted_at",
    "grade_extracted",
    "product_raw",
    "product_extract",
    "product",
    "sentiment",
    "score_service",
    "score_tariffs",
    "score_reliability",
    "strengths",
    "weaknesses",
    "emotional_tags",
]

PRODUCT_LIST_COLUMNS = ["strengths", "weaknesses", "emotional_tags"]

def _prepare_review_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    length = len(df)
    for col in REVIEW_COLUMNS:
        if col not in df.columns:
            if col in REVIEW_LIST_COLUMNS:
                df[col] = [[] for _ in range(length)]
            else:
                df[col] = None
    for col in REVIEW_LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_ensure_sequence)
    if "text_clean" in df.columns:
        df["text_clean"] = df["text_clean"].fillna("")
    return df[REVIEW_COLUMNS]

def _prepare_product_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=PRODUCT_COLUMNS)
    length = len(df)
    for col in PRODUCT_COLUMNS:
        if col not in df.columns:
            if col in PRODUCT_LIST_COLUMNS:
                df[col] = [[] for _ in range(length)]
            else:
                df[col] = None
    for col in PRODUCT_LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_ensure_sequence)
    return df[PRODUCT_COLUMNS]

def _build_product_rows(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        items = rec.get("__product_items") or []
        dataset = rec.get("dataset")
        posted_at = rec.get("posted_at")
        grade = rec.get("grade_extracted")
        for item in items:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "d_id": rec.get("d_id"),
                    "posted_at": posted_at,
                    "grade_extracted": grade,
                    "product_raw": item.get("product_raw"),
                    "product_extract": None,
                    "product": item.get("product"),
                    "sentiment": item.get("sentiment"),
                    "score_service": item.get("score_service"),
                    "score_tariffs": item.get("score_tariffs"),
                    "score_reliability": item.get("score_reliability"),
                    "strengths": item.get("strengths") or [],
                    "weaknesses": item.get("weaknesses") or [],
                    "emotional_tags": item.get("emotional_tags") or [],
                }
            )
    return rows


def _normalize_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        return float(value)
    if isinstance(value, str) and value.strip():
        cleaned = value.replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _default_enrichment(rating: Optional[float]) -> Dict[str, Any]:
    return {
        "sentiment": _sentiment_from_rating(rating),
        "product": PRODUCT_TAXONOMY[-1],
        "score_service": None,
        "score_tariffs": None,
        "score_reliability": None,
        "product_strengths": [],
        "product_weaknesses": [],
        "product_details": "[]",
        "product_list": [],
        "product_raw_list": [],
        "sentiment_list": [],
        "score_service_list": [],
        "score_tariffs_list": [],
        "score_reliability_list": [],
        "strengths_list": [],
        "weaknesses_list": [],
        "emotional_tags_list": [],
        "__product_items": [],
    }

def _parse_llm_json(content: str) -> Dict[str, Any]:
    if not content:
        raise ValueError("Empty response from LLM")
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"{.*}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise ValueError("Failed to parse JSON from LLM response") from exc
        raise ValueError("Failed to parse JSON from LLM response")


def _normalize_enrichment(
    data: Optional[Dict[str, Any]],
    rating: Optional[float],
    *,
    taxonomy: Optional[Sequence[str]] = PRODUCT_TAXONOMY,
) -> Dict[str, Any]:
    result = _default_enrichment(rating)
    if not isinstance(data, dict):
        return result

    raw_items = data.get("items")
    normalized_items: List[Dict[str, Any]] = []
    if isinstance(raw_items, list):
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            product_value = raw.get("product")
            product_raw = str(product_value).strip() if isinstance(product_value, str) else ""
            if taxonomy:
                product_name = (
                    PRODUCT_LOOKUP.get(product_raw.lower(), PRODUCT_TAXONOMY[-1])
                    if product_raw
                    else PRODUCT_TAXONOMY[-1]
                )
            else:
                product_name = product_raw or ""

            sentiment_value = raw.get("sentiment")
            if isinstance(sentiment_value, str) and sentiment_value.strip().lower() in VALID_SENTIMENTS:
                sentiment_norm = sentiment_value.strip().lower()
            else:
                sentiment_norm = _sentiment_from_rating(rating)

            score_service = _normalize_score(raw.get("score_service"))
            score_tariffs = _normalize_score(raw.get("score_tariffs"))
            score_reliability = _normalize_score(raw.get("score_reliability"))

            strengths = _ensure_list(raw.get("strengths") or raw.get("product_strengths"))
            weaknesses = _ensure_list(raw.get("weaknesses") or raw.get("product_weaknesses"))
            emotional_tags = _ensure_list(raw.get("emotional_tags", []))

            normalized_items.append(
                {
                    "product_raw": product_raw or product_name,
                    "product": product_name,
                    "sentiment": sentiment_norm,
                    "score_service": score_service,
                    "score_tariffs": score_tariffs,
                    "score_reliability": score_reliability,
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                    "emotional_tags": emotional_tags,
                }
            )

    if normalized_items:
        primary = normalized_items[0]
        result["product"] = primary.get("product") or (PRODUCT_TAXONOMY[-1] if taxonomy else "")
        result["sentiment"] = primary.get("sentiment") or _sentiment_from_rating(rating)
        result["score_service"] = primary.get("score_service")
        result["score_tariffs"] = primary.get("score_tariffs")
        result["score_reliability"] = primary.get("score_reliability")
        result["product_strengths"] = primary.get("strengths") or []
        result["product_weaknesses"] = primary.get("weaknesses") or []

        result["product_raw_list"] = [item.get("product_raw") for item in normalized_items]
        result["product_list"] = [item.get("product") for item in normalized_items]
        result["sentiment_list"] = [item.get("sentiment") for item in normalized_items]
        result["score_service_list"] = [item.get("score_service") for item in normalized_items]
        result["score_tariffs_list"] = [item.get("score_tariffs") for item in normalized_items]
        result["score_reliability_list"] = [item.get("score_reliability") for item in normalized_items]
        result["strengths_list"] = [item.get("strengths") for item in normalized_items]
        result["weaknesses_list"] = [item.get("weaknesses") for item in normalized_items]
        result["emotional_tags_list"] = [item.get("emotional_tags") for item in normalized_items]
        result["product_details"] = json.dumps(normalized_items, ensure_ascii=False)
        result["__product_items"] = normalized_items
        return result

    sentiment = data.get("sentiment")
    if isinstance(sentiment, str) and sentiment.strip():
        candidate = sentiment.strip().lower()
        if candidate in VALID_SENTIMENTS:
            result["sentiment"] = candidate

    product = data.get("product")
    if taxonomy:
        candidate = product if isinstance(product, str) and product.strip() else result["product"]
        if isinstance(candidate, str):
            result["product"] = PRODUCT_LOOKUP.get(candidate.lower(), PRODUCT_TAXONOMY[-1])
        else:
            result["product"] = PRODUCT_TAXONOMY[-1]
        if isinstance(product, str) and product.strip():
            result["product_raw_list"] = [product.strip()]
    else:
        items: List[str] = []
        if isinstance(product, (list, tuple)):
            items = [str(x).strip() for x in product if str(x).strip()]
        elif isinstance(product, str):
            items = [t.strip() for t in re.split(r"[;,\n]+", product) if t.strip()]
        items = list(dict.fromkeys(items))
        if items:
            result["product"] = items[0]
            result["product_list"] = items
            result["product_raw_list"] = items

    for key in ("score_service", "score_tariffs", "score_reliability"):
        result[key] = _normalize_score(data.get(key))

    result["product_strengths"] = _ensure_list(data.get("product_strengths"))
    result["product_weaknesses"] = _ensure_list(data.get("product_weaknesses"))

    if not result["product_list"]:
        result["product_list"] = [result["product"]]
    if not result["product_raw_list"]:
        result["product_raw_list"] = list(result["product_list"])
    if not result["sentiment_list"]:
        result["sentiment_list"] = [result["sentiment"]]
    if not result["score_service_list"]:
        result["score_service_list"] = [result.get("score_service")]
    if not result["score_tariffs_list"]:
        result["score_tariffs_list"] = [result.get("score_tariffs")]
    if not result["score_reliability_list"]:
        result["score_reliability_list"] = [result.get("score_reliability")]
    if not result["strengths_list"]:
        result["strengths_list"] = [result.get("product_strengths")]
    if not result["weaknesses_list"]:
        result["weaknesses_list"] = [result.get("product_weaknesses")]
    if not result["emotional_tags_list"]:
        result["emotional_tags_list"] = [[]]

    legacy_item = {
        "product_raw": result["product_raw_list"][0] if result["product_raw_list"] else result.get("product"),
        "product": result.get("product"),
        "sentiment": result.get("sentiment"),
        "score_service": result.get("score_service"),
        "score_tariffs": result.get("score_tariffs"),
        "score_reliability": result.get("score_reliability"),
        "strengths": result.get("product_strengths", []),
        "weaknesses": result.get("product_weaknesses", []),
        "emotional_tags": [],
    }
    result["product_details"] = json.dumps([legacy_item], ensure_ascii=False)
    result["__product_items"] = [legacy_item]
    return result


def _build_prompt(payload: Dict[str, Any]) -> str:
    pieces: List[str] = []
    title = payload.get("title")
    if isinstance(title, str) and title.strip():
        pieces.append(f"Title: {title.strip()}")

    text_clean = payload.get("text_clean") or ""
    if text_clean:
        pieces.append(f"Review: {text_clean}")
    else:
        text_raw = payload.get("text_raw") or ""
        if text_raw:
            pieces.append(f"Review (raw): {text_raw}")

    rating = payload.get("rating")
    if rating is not None:
        pieces.append(f"Rating: {rating}")

    user_name = payload.get("user_name") or ""
    if user_name:
        pieces.append(f"Author: {user_name}")

    context = "\n".join(pieces)
    taxonomy_text = "\n".join(f"- {item}" for item in PRODUCT_TAXONOMY)
    return render_review_prompt_taxonomy(taxonomy_text, context)


def _build_prompt_freeform(payload: Dict[str, Any]) -> str:
    pieces: List[str] = []
    title = payload.get("title")
    if isinstance(title, str) and title.strip():
        pieces.append(f"Title: {title.strip()}")

    text_clean = payload.get("text_clean") or ""
    if text_clean:
        pieces.append(f"Review: {text_clean}")
    else:
        text_raw = payload.get("text_raw") or ""
        if text_raw:
            pieces.append(f"Review (raw): {text_raw}")

    rating = payload.get("rating")
    if rating is not None:
        pieces.append(f"Rating: {rating}")

    user_name = payload.get("user_name") or ""
    if user_name:
        pieces.append(f"Author: {user_name}")

    context = "\n".join(pieces)
    return render_review_prompt_freeform(context)

def _standardize_products_with_llm(client: OpenAI, model: str, mentions: List[str], known_categories: Optional[Sequence[str]] = None) -> Tuple[Dict[str, str], Optional[float]]:
    # Prepare a stable, deduplicated list to reduce token usage
    unique = list(dict.fromkeys([m.strip() for m in mentions if isinstance(m, str) and m.strip()]))
    if not unique:
        return {}, None

    sample_text = "\n".join(f"- {m}" for m in unique)
    prompt = render_mapping_prompt(sample_text, list(known_categories) if known_categories else None)

    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Ответь строго валидным JSON. Верни только объект 'mapping'. Значения mapping ДОЛЖНЫ быть строкой названия категории на русском языке. Не добавляй дополнительные поля (например, 'sentiment')."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                timeout=REQUEST_TIMEOUT,
                response_format={"type": "json_object"},
                extra_body={
                    # "provider": {"sort": "price"},  # optional provider hint
                    "usage": {"include": True},
                    "response_format": {"type": "json_object"},
                },
            )
            choice = response.choices[0]
            content = getattr(choice.message, "content", None)
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if content is None:
                raise ValueError("LLM response contains no content")

            raw_text = content if isinstance(content, str) else str(content)
            LLM_LOGGER.debug("LLM mapping raw response (chars=%s)\n%s", len(raw_text), raw_text)
            try:
                data = _parse_llm_json(raw_text)
            except ValueError:
                LLM_LOGGER.error("Failed to parse LLM mapping response (chars=%s)\n%s", len(raw_text), raw_text)
                LOGGER.error("Failed to parse LLM mapping response; see pipeline-debug.log for raw payload")
                raise
            mapping_raw = data.get("mapping") if isinstance(data, dict) else {}
            if not isinstance(mapping_raw, dict):
                mapping_raw = {}
            try:
                usage = getattr(response, "usage", None)
                raw_cost = getattr(usage, "cost", None) if usage is not None else None
                if raw_cost is None:
                    completion = getattr(response, "completion", None)
                    comp_usage = getattr(completion, "usage", None) if completion is not None else None
                    raw_cost = getattr(comp_usage, "cost", None) if comp_usage is not None else None
                cost: Optional[float] = float(raw_cost) if raw_cost is not None else None
            except Exception:
                cost = None

            def _norm_cat(value: Any) -> Optional[str]:
                if isinstance(value, dict):
                    for key in ("category", "��⥣���", "name", "��������"):
                        val = value.get(key)
                        if isinstance(val, str) and val.strip():
                            return val.strip()
                    return None
                if isinstance(value, str):
                    s = value.strip()
                    if not s:
                        return None
                    try:
                        import json as _json
                        s_try = s
                        if s_try.startswith("{") and "category" in s_try and "'" in s_try:
                            s_try = s_try.replace("'", '"')
                        obj = _json.loads(s_try)
                        if isinstance(obj, dict):
                            return _norm_cat(obj)
                    except Exception:
                        pass
                    return s
                try:
                    return str(value).strip() or None
                except Exception:
                    return None

            def _extract_strings(value: Any) -> List[str]:
                items: List[str] = []

                def _walk(node: Any) -> None:
                    if node is None:
                        return
                    if isinstance(node, (list, tuple, set)):
                        for child in node:
                            _walk(child)
                        return
                    if isinstance(node, dict):
                        for key in ("original", "phrase", "text", "value", "raw", "name"):
                            if key in node:
                                _walk(node.get(key))
                        return
                    try:
                        text_value = str(node).strip()
                    except Exception:
                        return
                    if text_value:
                        items.append(text_value)

                _walk(value)
                return items

            mapping: Dict[str, str] = {}
            seen_originals: Set[str] = set()
            for cat_key, originals in mapping_raw.items():
                cat = _norm_cat(cat_key)
                if not cat:
                    continue
                for original_text in _extract_strings(originals):
                    lowered_original = original_text.lower()
                    if lowered_original in seen_originals:
                        continue
                    seen_originals.add(lowered_original)
                    mapping[original_text] = cat
            return mapping, cost
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
                continue
            raise
    if last_error:
        raise last_error
    return {}, None



def _chunked(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _load_combined_reviews(processed_dir: Path) -> Tuple[pd.DataFrame, Path]:
    path = processed_dir / "enriched_combined.csv"
    if not path.exists():
        return pd.DataFrame(columns=REVIEW_COLUMNS), path
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        LOGGER.warning("Failed to read combined reviews file %s: %s", path, exc)
        df = pd.DataFrame(columns=REVIEW_COLUMNS)
    return df, path


def _load_products_for_guided_mapping(processed_dir: Path) -> Tuple[pd.DataFrame, Path]:
    path = processed_dir / "enriched_products_combined.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Combined products file not found: {path}. Run research mode to generate it first."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read combined products file {path}: {exc}") from exc
    return df, path


def _filter_products_by_date_range(df: pd.DataFrame, date_from: Optional[str], date_to: Optional[str]) -> pd.DataFrame:
    if "posted_at" not in df.columns or (not date_from and not date_to):
        return df
    posted = pd.to_datetime(df["posted_at"], utc=True, errors="coerce")
    mask = pd.Series(True, index=df.index)
    if date_from:
        start = pd.to_datetime(date_from, utc=True, errors="coerce")
        if pd.notna(start):
            mask &= posted >= start
    if date_to:
        end = pd.to_datetime(date_to, utc=True, errors="coerce")
        if pd.notna(end):
            mask &= posted <= end
    return df.loc[mask]


def _extract_unique_products(values: Iterable[Any]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
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
        low = text.lower()
        if low in seen:
            continue
        seen.add(low)
        result.append(text)
    return result


def run_guided_mapping_only(
    llm_config: LLMConfig,
    *,
    data_root: Path,
    mapping_guidance: Optional[Sequence[str]] = None,
    batch_size: int = GUIDED_BATCH_SIZE,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, DatasetResult]]:
    processed_dir = Path(data_root).resolve() / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    combined_df, _ = _load_combined_reviews(processed_dir)
    products_df, products_path = _load_products_for_guided_mapping(processed_dir)

    if "product_raw" not in products_df.columns:
        raise ValueError(f"Column 'product_raw' not found in {products_path}")
    if "product" not in products_df.columns:
        products_df["product"] = None

    filtered = _filter_products_by_date_range(products_df, date_from, date_to)
    product_values = _extract_unique_products(filtered["product_raw"].tolist()) if not filtered.empty else []

    client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
    mapping_model = llm_config.mapping_model or llm_config.model

    categories_seed = list(mapping_guidance) if mapping_guidance else []
    mapping: Dict[str, str] = {}
    total_cost = 0.0
    mapping_batches = 0
    seen_products: Set[str] = set()

    for batch in _chunked(product_values, max(1, batch_size)):
        to_map: List[str] = []
        for value in batch:
            norm = value.strip().lower()
            if norm and norm not in seen_products:
                seen_products.add(norm)
                to_map.append(value)
        if not to_map:
            continue
        mapping_batches += 1
        batch_mapping, batch_cost = _standardize_products_with_llm(
            client, mapping_model, to_map, known_categories=categories_seed
        )
        if batch_mapping:
            mapping.update(batch_mapping)
            categories_seed = list(
                dict.fromkeys(
                    categories_seed
                    + [
                        str(v).strip()
                        for v in batch_mapping.values()
                        if isinstance(v, str) and str(v).strip()
                    ]
                )
            )
        if batch_cost:
            try:
                total_cost += float(batch_cost)
            except Exception:
                pass

    lowered = {
        str(k).strip().lower(): str(v).strip()
        for k, v in mapping.items()
        if isinstance(k, str) and isinstance(v, str) and str(k).strip() and str(v).strip()
    }

    def _map_single(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        text = str(value).strip()
        if not text:
            return None
        return lowered.get(text.lower()) if lowered else None

    def _map_list(values: Any) -> List[str]:
        items = _ensure_sequence(values)
        mapped: List[str] = []
        seen_local: Set[str] = set()
        for item in items:
            candidate = _map_single(item)
            if candidate and candidate not in seen_local:
                mapped.append(candidate)
                seen_local.add(candidate)
        return mapped

    products_df["product"] = products_df["product_raw"].apply(_map_single)

    if not combined_df.empty and "product_raw_list" in combined_df.columns:
        combined_df["product_list"] = combined_df["product_raw_list"].apply(_map_list)
        combined_df["product"] = combined_df["product_list"].apply(lambda lst: lst[0] if lst else None)

    try:
        products_df.to_csv(products_path, index=False, encoding='utf-8')
    except Exception as exc:
        LOGGER.warning("Failed to write combined products file %s: %s", products_path, exc)

    categories: Dict[str, List[str]] = {}
    for orig, target in mapping.items():
        if not isinstance(orig, str) or not isinstance(target, str):
            continue
        orig_clean = orig.strip()
        target_clean = target.strip()
        if not orig_clean or not target_clean:
            continue
        categories.setdefault(target_clean, [])
        if orig_clean not in categories[target_clean]:
            categories[target_clean].append(orig_clean)

    derived_entries: List[Dict[str, Any]] = []
    for name in sorted(categories.keys()):
        examples = categories[name][: min(50, len(categories[name]))]
        derived_entries.append({"name": name, "description": "", "examples": examples})

    metrics = {
        "total_rows": int(len(combined_df)),
        "processed_rows": int(len(combined_df)),
        "product_rows": int(len(products_df)),
        "llm_calls": 0,
        "llm_failures": 0,
        "llm_elapsed_seconds": 0.0,
        "llm_cost_total": 0.0,
        "product_mapping": categories,
        "product_mapping_lookup": mapping,
        "product_mapping_size": len(categories),
        "product_mapping_cost": float(total_cost),
        "product_mapping_batches": mapping_batches,
        "product_descriptions": {},
        "product_descriptions_size": 0,
        "product_descriptions_cost": 0.0,
        "derived_taxonomy": derived_entries,
        "source_path": str(products_path),
    }

    results = {
        "combined": DatasetResult(
            name="combined", dataframe=combined_df, products=products_df, metrics=metrics
        )
    }

    return combined_df, results



def describe_product_from_mentions(
    product_name: str,
    mentions: Sequence[str],
    llm_config: LLMConfig,
    *,
    max_examples: int = 20,
) -> Tuple[str, List[str], float]:
    """Generate a concise description for a product using raw mention strings."""
    clean_name = (product_name or "").strip()
    if not clean_name:
        return "", [], 0.0

    cleaned: List[str] = []
    seen: Set[str] = set()
    for value in mentions:
        if value is None:
            continue
        if isinstance(value, float):
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
        text = str(value).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(text)
    if not cleaned:
        return "", [], 0.0

    client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
    model = llm_config.mapping_model or llm_config.model

    try:
        descriptions, cost = _describe_products_with_llm(
            client,
            model,
            {clean_name: cleaned},
            max_examples=max_examples,
        )
    except Exception:
        return "", cleaned[: max_examples], 0.0

    description = ""
    if isinstance(descriptions, dict):
        description = str(descriptions.get(clean_name, "") or "").strip()

    final_cost = 0.0
    if cost is not None:
        try:
            final_cost = float(cost)
        except Exception:
            final_cost = 0.0

    return description, cleaned[: max_examples], final_cost



def _describe_products_with_llm(
    client: OpenAI,
    model: str,
    categories: Dict[str, List[str]],
    max_examples: int = 50,
) -> Tuple[Dict[str, str], Optional[float]]:
    descriptions: Dict[str, str] = {}
    total_cost: float = 0.0
    last_error: Optional[Exception] = None

    for name, mentions in categories.items():
        unique: List[str] = []
        seen = set()
        for item in mentions or []:
            if not isinstance(item, str):
                continue
            stripped = item.strip()
            lowered = stripped.lower() if stripped else ''
            if not stripped or lowered in seen:
                continue
            unique.append(stripped)
            seen.add(lowered)
            if len(unique) >= max_examples:
                break
        if not unique:
            descriptions[name] = ''
            continue

        prompt = render_taxonomy_description_prompt(name, unique)
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {'role': 'system', 'content': 'Ответь строго валидным JSON.'},
                        {'role': 'user', 'content': prompt},
                    ],
                    temperature=0.1,
                    timeout=REQUEST_TIMEOUT,
                    response_format={'type': 'json_object'},
                    extra_body={'usage': {'include': True}},
                )
                choice = response.choices[0]
                content = getattr(choice.message, 'content', None)
                if isinstance(content, list):
                    content = ''.join(part.get('text', '') for part in content if isinstance(part, dict))
                if content is None:
                    raise ValueError('LLM response contains no content')
                data = _parse_llm_json(content)
                description = ''
                if isinstance(data, dict):
                    desc_val = data.get('description')
                    if isinstance(desc_val, str):
                        description = desc_val.strip()
                if not description:
                    description = '; '.join(unique[: min(3, len(unique))])
                descriptions[name] = description
                try:
                    usage = getattr(response, 'usage', None)
                    raw_cost = getattr(usage, 'cost', None) if usage is not None else None
                    if raw_cost is None:
                        completion = getattr(response, 'completion', None)
                        comp_usage = getattr(completion, 'usage', None) if completion is not None else None
                        raw_cost = getattr(comp_usage, 'cost', None) if comp_usage is not None else None
                    if raw_cost is not None:
                        total_cost += float(raw_cost)
                except Exception:
                    pass
                break
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
                    continue
                if name not in descriptions:
                    descriptions[name] = ''
    if last_error and not descriptions:
        raise last_error
    return descriptions, (total_cost if total_cost else None)



def _call_llm(
    client: OpenAI,
    model: str,
    payload: Dict[str, Any],
    taxonomy: Optional[Sequence[str]] = PRODUCT_TAXONOMY,
) -> Tuple[Dict[str, Any], Optional[float]]:
    prompt = _build_prompt(payload) if taxonomy else _build_prompt_freeform(payload)
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Ответь строго валидным JSON. Используй русский язык для всех текстовых значений и названий категорий. Поле 'sentiment' должно быть одним из: positive | neutral | negative (англ.)."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                timeout=REQUEST_TIMEOUT,
                response_format={"type": "json_object"},
                extra_body={
                    # "provider": {"sort": "price"},  # optional provider hint
                    "usage": {"include": True},
                    "response_format": {"type": "json_object"},
                },
            )
            choice = response.choices[0]
            content = getattr(choice.message, "content", None)
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if content is None:
                raise ValueError("LLM response contains no content")

            data = _parse_llm_json(content)
            # Extract monetary cost via response.usage.cost or completion.usage.cost
            try:
                usage = getattr(response, "usage", None)
                raw_cost = getattr(usage, "cost", None) if usage is not None else None
                if raw_cost is None:
                    completion = getattr(response, "completion", None)
                    comp_usage = getattr(completion, "usage", None) if completion is not None else None
                    raw_cost = getattr(comp_usage, "cost", None) if comp_usage is not None else None
                cost: Optional[float] = float(raw_cost) if raw_cost is not None else None
            except Exception:
                cost = None
            return data, cost
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "LLM request failed for dataset=%s id=%s (attempt %s/%s): %s",
                payload.get("dataset"),
                payload.get("id"),
                attempt + 1,
                tries,
                exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("LLM request failed without exception")


def _prepare_record(row: Dict[str, Any], spec: DatasetSpec) -> Dict[str, Any]:
    source_dataset = row.get("__dataset_name") or spec.name
    record: Dict[str, Any] = {
        "dataset": source_dataset,
        "id": row.get("id"),
        "title": row.get("title"),
        "text_raw": row.get("text"),
        "text_clean": html_to_text(row.get("text")),
        "agent_answer_text": html_to_text(row.get("agentAnswerText")) if "agentAnswerText" in row else "",
        "posted_at": _normalize_datetime(row.get("date")),
        "grade_extracted": _normalize_rating(row.get("rating")),
        "rating": _normalize_rating(row.get("rating")),
        "source_extracted": row.get("userName") or DEFAULT_SOURCE,
        "user_name": row.get("userName"),
    }
    d_id_value = row.get("d_id")
    if isinstance(d_id_value, str):
        d_id_value = d_id_value.strip() or None
    elif d_id_value is not None:
        d_id_value = str(d_id_value).strip() or None
    record["d_id"] = d_id_value
    record["text_clean"] = record["text_clean"] or ""
    record["agent_answer_text"] = record["agent_answer_text"] or ""
    return record


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_reviews_dataset(
    specs: Sequence[DatasetSpec],
    llm_config: LLMConfig,
    taxonomy: Optional[Sequence[str]] = PRODUCT_TAXONOMY,
    mapping_guidance: Optional[Sequence[str]] = None,
    *,
    standardize_freeform: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, DatasetResult]]:
    if not specs:
        raise ValueError("At least one dataset specification is required")

    freeform_mode = taxonomy is None
    apply_standardization = freeform_mode and standardize_freeform

    client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)

    combined_frames: List[pd.DataFrame] = []
    combined_product_frames: List[pd.DataFrame] = []
    results: Dict[str, DatasetResult] = {}

    # Collect mentions across datasets when taxonomy is not provided (free-form mode)
    global_mentions: List[str] = [] if taxonomy is None else []

    for spec in specs:
        dataset_start = time.time()
        try:
            df = pd.read_csv(spec.path)
        except Exception as exc:
            LOGGER.error("Failed to load dataset %s: %s", spec.path, exc)
            raise

        # Optional date filtering from environment
        df_date_from = os.getenv("REVIEWS_DATE_FROM", "").strip()
        df_date_to = os.getenv("REVIEWS_DATE_TO", "").strip()
        if df_date_from or df_date_to:
            df = _filter_by_date(df, df_date_from, df_date_to)

        if spec.limit is not None:
            df = df.head(spec.limit)

        if not df.empty:
            df = df.copy()
            df["__dataset_name"] = spec.name

        metrics: Dict[str, Any] = {
            "dataset": spec.name,
            "source_path": spec.path,
            "limit": spec.limit,
            "total_rows": int(len(df)),
            "processed_rows": 0,
            "llm_calls": 0,
            "llm_failures": 0,
            "llm_elapsed_seconds": 0.0,
            "llm_cost_total": 0.0,
        }

        records: List[Dict[str, Any]] = []

        if not df.empty:
            rows = df.to_dict(orient="records")

            # Prepare indices, records and payloads
            prepared: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
            for idx, row in enumerate(rows):
                record = _prepare_record(row, spec)
                payload = {
                    "dataset": spec.name,
                    "id": record.get("id"),
                    "title": record.get("title"),
                    "text_clean": record.get("text_clean"),
                    "text_raw": record.get("text_raw"),
                    "rating": record.get("rating"),
                    "agent_answer_text": record.get("agent_answer_text"),
                    "user_name": record.get("user_name"),
                }
                prepared.append((idx, record, payload))

            def _process_one(item: Tuple[int, Dict[str, Any], Dict[str, Any]]):
                idx, record, payload = item
                iter_start = time.perf_counter()
                llm_data: Optional[Dict[str, Any]] = None
                llm_cost: Optional[float] = None
                llm_calls = 0
                llm_failures = 0
                llm_elapsed = 0.0
                if record.get("text_clean"):
                    try:
                        llm_start = time.time()
                        llm_data, llm_cost = _call_llm(client, llm_config.model, payload, taxonomy)
                        llm_elapsed += time.time() - llm_start
                        llm_calls += 1
                    except Exception as exc:
                        llm_failures += 1
                        LOGGER.warning(
                            "Falling back to heuristic enrichment for dataset=%s id=%s: %s",
                            spec.name,
                            record.get("id"),
                            exc,
                        )
                enrichment = _normalize_enrichment(llm_data, record.get("rating"), taxonomy=taxonomy)
                record.update(enrichment)
                if freeform_mode and not apply_standardization:
                    record["product"] = None
                    record["product_list"] = []
                    items = record.get("__product_items") or []
                    cleaned_items = []
                    for item in items:
                        if isinstance(item, dict):
                            item["product"] = None
                            cleaned_items.append(item)
                    record["__product_items"] = cleaned_items
                    if cleaned_items:
                        record["product_details"] = json.dumps(cleaned_items, ensure_ascii=False)
                    else:
                        record["product_details"] = "[]"
                record["cost"] = float(llm_cost) if llm_cost is not None else None
                iter_elapsed = time.perf_counter() - iter_start
                record["iter_seconds"] = float(iter_elapsed)
                return (
                    idx,
                    record,
                    {
                        "llm_calls": llm_calls,
                        "llm_failures": llm_failures,
                        "llm_elapsed_seconds": llm_elapsed,
                        "llm_cost_total": float(llm_cost) if llm_cost is not None else 0.0,
                    },
                )

            try:
                workers = max(1, int(os.getenv("REVIEWS_WORKERS", str(DEFAULT_WORKERS))))
            except Exception:
                workers = DEFAULT_WORKERS
            # Save worker count into metrics
            metrics["workers"] = workers

            results_buffer: List[Optional[Dict[str, Any]]] = [None] * len(prepared)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_process_one, it) for it in prepared]
                pbar = tqdm(total=len(futures), desc=f"{spec.name}", unit="row", leave=False)
                for fut in as_completed(futures):
                    idx, rec, inc = fut.result()
                    results_buffer[idx] = rec
                    metrics["llm_calls"] += inc.get("llm_calls", 0)
                    metrics["llm_failures"] += inc.get("llm_failures", 0)
                    metrics["llm_elapsed_seconds"] += inc.get("llm_elapsed_seconds", 0.0)
                    metrics["llm_cost_total"] += inc.get("llm_cost_total", 0.0)
                    pbar.update(1)
                pbar.close()

            records = [r for r in results_buffer if r is not None]

        product_rows = _build_product_rows(records)
        for rec in records:
            rec.pop("__product_items", None)

        dataset_df = pd.DataFrame.from_records(records)
        product_df = pd.DataFrame.from_records(product_rows)

        dataset_df = _prepare_review_dataframe(dataset_df)
        product_df = _prepare_product_dataframe(product_df)

        metrics["processed_rows"] = int(len(dataset_df))
        metrics["product_rows"] = int(len(product_df))
        metrics["processing_seconds"] = time.time() - dataset_start
        if metrics["processed_rows"] > 0 and metrics["processing_seconds"] > 0:
            metrics["avg_iter_seconds"] = metrics["processing_seconds"] / metrics["processed_rows"]

        if not dataset_df.empty and "text_clean" in dataset_df:
            avg_length = dataset_df["text_clean"].str.len().mean()
            if isinstance(avg_length, float) and avg_length == avg_length:
                metrics["avg_text_length"] = float(avg_length)

        # In free-form mode, accumulate product mentions for global standardization later
        if apply_standardization and not dataset_df.empty:
            if "product_list" in dataset_df:
                for entry in dataset_df["product_list"].dropna().tolist():
                    if isinstance(entry, list):
                        for item in entry:
                            item_str = str(item).strip()
                            if item_str:
                                global_mentions.append(item_str)
                    else:
                        global_mentions.extend(_ensure_list(entry))

        result = DatasetResult(name=spec.name, dataframe=dataset_df, products=product_df, metrics=metrics)
        results[spec.name] = result

        if not result.dataframe.empty:
            combined_frames.append(result.dataframe)
        if not result.products.empty:
            combined_product_frames.append(result.products)

    combined_df = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame(columns=REVIEW_COLUMNS)
    combined_products_df = pd.concat(combined_product_frames, ignore_index=True) if combined_product_frames else pd.DataFrame(columns=PRODUCT_COLUMNS)

    combined_metrics: Dict[str, Any] = {
        "total_rows": int(len(combined_df)),
        "processed_rows": int(len(combined_df)),
        "product_rows": int(len(combined_products_df)),
    }

    # If taxonomy is not provided, standardize products globally across all datasets
    if apply_standardization and not combined_df.empty and "product_list" in combined_df.columns:
        mapping_model = llm_config.mapping_model or llm_config.model
        mapping, map_cost = _standardize_products_with_llm(
            client, mapping_model, global_mentions, known_categories=mapping_guidance
        )

        lowered = {str(k).strip().lower(): str(v).strip() for k, v in (mapping or {}).items() if str(k).strip()}

        def _map_single(value: Any) -> str:
            key = str(value).strip().lower()
            return lowered.get(key, str(value).strip())

        def _map_list(values: Any) -> List[str]:
            items = _ensure_sequence(values)
            mapped: List[str] = []
            seen: Set[str] = set()
            for item in items:
                candidate = _map_single(item)
                if candidate and candidate not in seen:
                    mapped.append(candidate)
                    seen.add(candidate)
            return mapped

        for res in results.values():
            df = res.dataframe
            if not df.empty and "product_list" in df.columns:
                df["product_list"] = df["product_list"].apply(_map_list)
        for res in results.values():
            prod_df = res.products
            if not prod_df.empty and "product" in prod_df.columns:
                prod_df["product"] = prod_df["product"].apply(_map_single)

        combined_df = pd.concat(
            [res.dataframe for res in results.values() if not res.dataframe.empty], ignore_index=True
        ) if results else combined_df
        combined_products_df = pd.concat(
            [res.products for res in results.values() if not res.products.empty], ignore_index=True
        ) if results else combined_products_df

        combined_metrics["total_rows"] = int(len(combined_df))
        combined_metrics["processed_rows"] = int(len(combined_df))
        combined_metrics["product_rows"] = int(len(combined_products_df))

        categories: Dict[str, List[str]] = {}
        for orig, cat in (mapping or {}).items():
            if not isinstance(orig, str) or not isinstance(cat, str):
                continue
            cat_clean = cat.strip()
            orig_clean = orig.strip()
            if not cat_clean or not orig_clean:
                continue
            try:
                low = cat_clean.lower()
                if low in {"счет", "счёт"} or re.search(r"(справк|выписк).*(сч[её]т)", low):
                    cat_clean = "счёт"
            except Exception:
                pass
            categories.setdefault(cat_clean, [])
            if orig_clean not in categories[cat_clean]:
                categories[cat_clean].append(orig_clean)

        descriptions: Dict[str, str] = {}
        desc_cost: Optional[float] = None
        if categories:
            try:
                descriptions, desc_cost = _describe_products_with_llm(
                    client,
                    mapping_model,
                    categories,
                )
            except Exception as exc:
                LOGGER.warning("Failed to generate product descriptions: %s", exc)
                descriptions = {name: "" for name in categories.keys()}

        derived_entries: List[Dict[str, Any]] = []
        for name in sorted(categories.keys()):
            examples = categories[name][: min(50, len(categories[name]))]
            derived_entries.append(
                {
                    "name": name,
                    "description": descriptions.get(name, ""),
                    "examples": examples,
                }
            )

        combined_metrics.update(
            {
                "product_mapping": categories,
                "product_mapping_size": int(len(categories)),
                "derived_taxonomy": derived_entries,
                "product_descriptions": descriptions,
            }
        )
        if map_cost is not None:
            combined_metrics["product_mapping_cost"] = float(map_cost)
        if desc_cost is not None:
            combined_metrics["product_descriptions_cost"] = float(desc_cost)
        if descriptions:
            combined_metrics["product_descriptions_size"] = len([d for d in descriptions.values() if d])

    results["combined"] = DatasetResult(
        name="combined",
        dataframe=combined_df,
        products=combined_products_df,
        metrics=combined_metrics,
    )


    return combined_df, results




