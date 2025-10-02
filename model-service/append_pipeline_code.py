from pathlib import Path
import textwrap

append_code = textwrap.dedent("""
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
        "summary": "",
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


def _normalize_enrichment(data: Optional[Dict[str, Any]], rating: Optional[float]) -> Dict[str, Any]:
    result = _default_enrichment(rating)
    if not isinstance(data, dict):
        return result

    sentiment = data.get("sentiment")
    if isinstance(sentiment, str) and sentiment.strip():
        candidate = sentiment.strip().lower()
        result["sentiment"] = candidate if candidate in VALID_SENTIMENTS else _sentiment_from_rating(rating)

    product = data.get("product")
    product_candidate = product if isinstance(product, str) and product.strip() else result["product"]
    if isinstance(product_candidate, str):
        result["product"] = PRODUCT_LOOKUP.get(product_candidate.lower(), PRODUCT_TAXONOMY[-1])
    else:
        result["product"] = PRODUCT_TAXONOMY[-1]

    for key in ("score_service", "score_tariffs", "score_reliability"):
        result[key] = _normalize_score(data.get(key))

    result["product_strengths"] = _ensure_list(data.get("product_strengths"))
    result["product_weaknesses"] = _ensure_list(data.get("product_weaknesses"))

    summary = data.get("summary")
    if isinstance(summary, str):
        result["summary"] = summary.strip()

    return result


def _build_prompt(payload: Dict[str, Any]) -> str:
    taxonomy_text = "\n".join(f"- {item}" for item in PRODUCT_TAXONOMY)
    pieces: List[str] = []
    title = payload.get("title")
    if isinstance(title, str) and title.strip():
        pieces.append(f"Заголовок: {title.strip()}")

    text_clean = payload.get("text_clean") or ""
    if text_clean:
        pieces.append(f"Отзыв: {text_clean}")

    text_raw = payload.get("text_raw") or ""
    if not text_clean and text_raw:
        pieces.append(f"Отзыв (raw): {text_raw}")

    rating = payload.get("rating")
    if rating is not None:
        pieces.append(f"Оценка пользователя: {rating}")

    agent_answer = payload.get("agent_answer_text") or ""
    if agent_answer:
        pieces.append(f"Ответ банка: {agent_answer}")

    user_name = payload.get("user_name") or ""
    if user_name:
        pieces.append(f"Автор: {user_name}")

    context = "\n".join(pieces)
    prompt = (
        "Ты аналитик службы качества банка. Проанализируй отзыв клиента о финансовом продукте.\n"
        "Категории продуктов (выбери одну из них):\n"
        f"{taxonomy_text}\n\n"
        "Верни только валидный JSON со следующими полями:\n"
        "{\n"
        '  "sentiment": "positive|neutral|negative",\n'
        '  "product": "<одна категория>",\n'
        '  "score_service": <целое 0-10 или null>,\n'
        '  "score_tariffs": <целое 0-10 или null>,\n'
        '  "score_reliability": <целое 0-10 или null>,\n'
        '  "product_strengths": ["краткая сильная сторона", ...],\n'
        '  "product_weaknesses": ["краткий недостаток", ...],\n'
        '  "summary": "1-2 предложения вывода"\n'
        "}\n"
        "Используй null или пустые списки, если информации нет. Не добавляй пояснений вне JSON.\n\n"
        f"Данные отзыва:\n{context}"
    )
    return prompt


def _call_llm(client: OpenAI, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _build_prompt(payload)
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Отвечай только валидным JSON без пояснений."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                timeout=REQUEST_TIMEOUT,
            )
            choice = response.choices[0]
            content = getattr(choice.message, "content", None)
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if content is None:
                raise ValueError("LLM response contains no content")
            return _parse_llm_json(content)
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "LLM request failed for dataset=%s id=%s (attempt %s/%s): %s",
                payload.get("dataset"),
                payload.get("id"),
                attempt + 1,
                MAX_RETRIES + 1,
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
    record: Dict[str, Any] = {
        "dataset": spec.name,
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
    record["text_clean"] = record["text_clean"] or ""
    record["agent_answer_text"] = record["agent_answer_text"] or ""
    return record


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_reviews_dataset(
    specs: Sequence[DatasetSpec],
    llm_config: LLMConfig,
) -> Tuple[pd.DataFrame, Dict[str, DatasetResult]]:
    if not specs:
        raise ValueError("At least one dataset specification is required")

    client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)

    combined_frames: List[pd.DataFrame] = []
    results: Dict[str, DatasetResult] = {}

    for spec in specs:
        dataset_start = time.time()
        try:
            df = pd.read_csv(spec.path)
        except Exception as exc:
            LOGGER.error("Failed to load dataset %s: %s", spec.path, exc)
            raise

        if spec.limit is not None:
            df = df.head(spec.limit)

        metrics: Dict[str, Any] = {
            "dataset": spec.name,
            "source_path": spec.path,
            "limit": spec.limit,
            "total_rows": int(len(df)),
            "processed_rows": 0,
            "llm_calls": 0,
            "llm_failures": 0,
            "llm_elapsed_seconds": 0.0,
        }

        records: List[Dict[str, Any]] = []

        if not df.empty:
            for row in df.to_dict(orient="records"):
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

                llm_data: Optional[Dict[str, Any]] = None
                if record["text_clean"]:
                    try:
                        llm_start = time.time()
                        llm_data = _call_llm(client, llm_config.model, payload)
                        metrics["llm_elapsed_seconds"] += time.time() - llm_start
                        metrics["llm_calls"] += 1
                    except Exception as exc:
                        metrics["llm_failures"] += 1
                        LOGGER.warning(
                            "Falling back to heuristic enrichment for dataset=%s id=%s: %s",
                            spec.name,
                            record.get("id"),
                            exc,
                        )
                enrichment = _normalize_enrichment(llm_data, record.get("rating"))
                record.update(enrichment)
                records.append(record)

        dataset_df = pd.DataFrame.from_records(records)
        metrics["processed_rows"] = int(len(dataset_df))
        metrics["processing_seconds"] = time.time() - dataset_start

        if not dataset_df.empty and "text_clean" in dataset_df:
            avg_length = dataset_df["text_clean"].str.len().mean()
            if isinstance(avg_length, float) and avg_length == avg_length:
                metrics["avg_text_length"] = float(avg_length)

        result = DatasetResult(name=spec.name, dataframe=dataset_df, metrics=metrics)
        results[spec.name] = result

        if not dataset_df.empty:
            combined_frames.append(dataset_df)

    combined_df = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
    return combined_df, results
""")

path = Path('reviews/pipeline.py')
path.write_text(path.read_text() + append_code)
