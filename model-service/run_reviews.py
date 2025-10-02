#!/usr/bin/env python3
"""Entry point and public API for the review enrichment workflow."""

from __future__ import annotations

import logging
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv

from reviews.pipeline import DatasetResult, DatasetSpec, LLMConfig, build_reviews_dataset
import reviews.pipeline as rpipe

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324"
DEFAULT_DATA_DIR = Path("data")
DEFAULT_LIMIT = 10


def _run_mode_base(specs, llm_config):
    return build_reviews_dataset(specs, llm_config)


def _run_mode_research(specs, llm_config):
    return build_reviews_dataset(specs, llm_config, taxonomy=None, standardize_freeform=False)


def _run_mode_research_guided(
    llm_config,
    *,
    data_root: Path,
    mapping_guidance,
    batch_size: int,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    return rpipe.run_guided_mapping_only(
        llm_config,
        data_root=data_root,
        mapping_guidance=mapping_guidance,
        batch_size=batch_size,
        date_from=date_from,
        date_to=date_to,
    )


def _parse_limit(raw: Optional[str], label: str) -> Optional[int]:
    if raw in (None, "", "None"):
        return None
    try:
        return int(raw)
    except ValueError:
        LOGGER.warning("Invalid %s value %s, falling back to %s", label, raw, DEFAULT_LIMIT)
        return DEFAULT_LIMIT


def _parse_bool(raw: Optional[str], default: bool = True) -> bool:
    if raw is None:
        return default
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return default


def load_runtime_from_env() -> Tuple[List[DatasetSpec], LLMConfig, Path]:
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        LOGGER.error("OPENROUTER_API_KEY is required")
        raise SystemExit(1)

    # Models: reviews (enrichment) and mapping (standardization)
    model_reviews = os.getenv("OPENROUTER_MODEL_REVIEWS") or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    model_mapping = os.getenv("OPENROUTER_MODEL_MAPPING") or model_reviews
    model_summary = os.getenv("OPENROUTER_MODEL_SUMMARY") or model_reviews
    data_root = Path(os.getenv("REVIEWS_DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()
    if not data_root.exists():
        LOGGER.error("Data directory not found: %s", data_root)
        raise SystemExit(1)

    datasets: List[DatasetSpec] = []

    banki_path = Path(os.getenv("BANKI_RU_DATASET", data_root / "banki_ru_full.csv"))
    if banki_path.exists():
        banki_limit = _parse_limit(os.getenv("BANKI_RU_LIMIT"), "BANKI_RU_LIMIT")
        datasets.append(DatasetSpec(name="banki.ru", path=str(banki_path), limit=banki_limit))
    else:
        LOGGER.warning("banki.ru dataset not found: %s", banki_path)

    sravni_path = Path(os.getenv("SRAVNI_RU_DATASET", data_root / "sravni_ru_full.csv"))
    if sravni_path.exists():
        sravni_limit = _parse_limit(os.getenv("SRAVNI_RU_LIMIT"), "SRAVNI_RU_LIMIT")
        datasets.append(DatasetSpec(name="sravni.ru", path=str(sravni_path), limit=sravni_limit))
    else:
        LOGGER.warning("sravni.ru dataset not found: %s", sravni_path)

    if not datasets:
        LOGGER.error("No datasets specified for enrichment")
        raise SystemExit(1)

    llm_config = LLMConfig(api_key=api_key, model=model_reviews, mapping_model=model_mapping, summary_model=model_summary)
    return datasets, llm_config, data_root


def _build_specs_from_paths(paths: Iterable[str]) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        specs.append(DatasetSpec(name=p.stem, path=str(p.resolve())))
    return specs


def run_reviews(
    dataset_paths: Optional[Iterable[str]] = None,
    *,
    datasets: Optional[Sequence[DatasetSpec]] = None,
    llm_config: Optional[LLMConfig] = None,
    data_dir: Optional[Path | str] = None,
    save_excel: bool = True,
    preview_rows: int = 5,
    configure_logging: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, DatasetResult]]:
    """Run the enrichment pipeline and return the combined dataframe."""

    if configure_logging:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        # Suppress noisy HTTP client logs (e.g., httpx/httpcore request lines)
        for noisy_logger in ("httpx", "httpcore", "openai", "urllib3"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)
            logging.getLogger(noisy_logger).propagate = False

    load_dotenv()

    specs: List[DatasetSpec] = list(datasets or [])
    if dataset_paths:
        specs.extend(_build_specs_from_paths(dataset_paths))

    if llm_config is None:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY must be provided")
        model_reviews = os.getenv("OPENROUTER_MODEL_REVIEWS") or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
        model_mapping = os.getenv("OPENROUTER_MODEL_MAPPING") or model_reviews
        model_summary = os.getenv("OPENROUTER_MODEL_SUMMARY") or model_reviews
        llm_config = LLMConfig(api_key=api_key, model=model_reviews, mapping_model=model_mapping, summary_model=model_summary)

    if data_dir is None:
        data_root = Path(os.getenv("REVIEWS_DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()
    else:
        data_root = Path(data_dir).resolve()

    if not specs:
        env_specs, env_llm, env_root = load_runtime_from_env()
        specs = env_specs
        data_root = env_root
        llm_config = llm_config or env_llm

    mode = os.getenv("REVIEWS_ANALYSIS_MODE", "").strip().lower() or "base"
    requires_datasets = mode in {"base", "research"}
    if requires_datasets and not specs:
        raise ValueError("No datasets provided for processing")

    processed_dir = data_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.time()

    if mode == "base":
        combined_df, results = _run_mode_base(specs, llm_config)
    elif mode == "research":
        combined_df, results = _run_mode_research(specs, llm_config)
    elif mode == "research_guided":
        taxonomy_seed = getattr(rpipe, "PRODUCT_TAXONOMY", None)
        date_from_env = os.getenv("REVIEWS_DATE_FROM", "").strip() or None
        date_to_env = os.getenv("REVIEWS_DATE_TO", "").strip() or None
        combined_df, results = _run_mode_research_guided(
            llm_config,
            data_root=data_root,
            mapping_guidance=taxonomy_seed,
            batch_size=rpipe.GUIDED_BATCH_SIZE,
            date_from=date_from_env,
            date_to=date_to_env,
        )
    else:
        raise ValueError(f"Unknown analysis mode: {mode}")

    preview_columns = [
        "dataset",
        "id",
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
        "cost",
        "iter_seconds",
    ]

    if save_excel:
        for name, result in results.items():
            if name == "combined":
                # skip individual save; combined is saved below
                continue
            safe_name = name.replace("/", "_").replace(" ", "_")
            output_path = processed_dir / f"enriched_{safe_name}.csv"
            result.dataframe.to_csv(output_path, index=False, encoding='utf-8')
            if not result.products.empty:
                products_path = processed_dir / f"enriched_products_{safe_name}.csv"
                result.products.to_csv(products_path, index=False, encoding='utf-8')
                LOGGER.info("Сохранён продуктовый результат: %s", products_path)
            # Save product mapping (free-form taxonomy mode)
            try:
                mapping = result.metrics.get("product_mapping") if isinstance(result.metrics, dict) else None
                if isinstance(mapping, dict):
                    mapping_path = processed_dir / f"product_mapping_{safe_name}.json"
                    with open(mapping_path, "w", encoding="utf-8") as fh:
                        json.dump(mapping, fh, ensure_ascii=False, indent=2)
                    try:
                        LOGGER.info("Словарь категорий сохранён: %s", mapping_path)
                        if mapping:
                            LOGGER.info("Словарь категорий (%d): %s", len(mapping), json.dumps(mapping, ensure_ascii=False))
                    except Exception:
                        LOGGER.info("Product mapping saved: %s", mapping_path)
            except Exception as exc:
                LOGGER.warning("Failed to save product mapping for %s: %s", name, exc)
            LOGGER.info("Сохранён результат: %s", output_path)

        if not combined_df.empty:
            combined_path = processed_dir / "enriched_combined.csv"
            combined_df.to_csv(combined_path, index=False, encoding='utf-8')
            # Save a single combined product mapping (category -> [mentions]) if available
            try:
                combined_res = results.get("combined")
                if combined_res and not combined_res.products.empty:
                    combined_products_path = processed_dir / "enriched_products_combined.csv"
                    combined_res.products.to_csv(combined_products_path, index=False, encoding='utf-8')
                    LOGGER.info("Сохранён объединённый продуктовый результат: %s", combined_products_path)
                mapping = combined_res.metrics.get("product_mapping") if combined_res else None
                if isinstance(mapping, dict):
                    mapping_path = processed_dir / "product_mapping_combined.json"
                    with open(mapping_path, "w", encoding="utf-8") as fh:
                        json.dump(mapping, fh, ensure_ascii=False, indent=2)
                    try:
                        LOGGER.info("Словарь категорий (общий) сохранён: %s", mapping_path)
                    except Exception:
                        LOGGER.info("Product mapping (combined) saved: %s", mapping_path)
                # Save derived taxonomy (for research/research_guided)
                derived = combined_res.metrics.get("derived_taxonomy") if combined_res else None
                if isinstance(derived, list) and derived:
                    derived_path = processed_dir / "derived_taxonomy_combined.json"
                    with open(derived_path, "w", encoding="utf-8") as fh:
                        json.dump(derived, fh, ensure_ascii=False, indent=2)
                    try:
                        LOGGER.info("Новая таксономия (derived) сохранена: %s", derived_path)
                    except Exception:
                        LOGGER.info("Derived taxonomy saved: %s", derived_path)
            except Exception as exc:
                LOGGER.warning("Failed to save combined product mapping: %s", exc)
            LOGGER.info("Сохранён объединённый результат: %s", combined_path)

    for name, result in results.items():
        existing_cols = [col for col in preview_columns if col in result.dataframe.columns]
        if False and existing_cols:
            print(f"--- {name} ---")
            print(result.dataframe[existing_cols].head(preview_rows).to_json(orient="records", force_ascii=False, indent=2))

    if not combined_df.empty:
        existing_cols = [col for col in preview_columns if col in combined_df.columns]
        if False and existing_cols:
            print("=== Комбинированный набор ===")
            print(combined_df[existing_cols].head(preview_rows).to_json(orient="records", force_ascii=False, indent=2))

    total_seconds = time.time() - run_start
    try:
        LOGGER.info("Общее время обработки: %.2f сек", total_seconds)
    except Exception:
        LOGGER.info("Total processing time: %.2f sec", total_seconds)

    # Build and save JSON report for this run
    models = {
        "reviews": getattr(llm_config, "model", None),
        "mapping": getattr(llm_config, "mapping_model", None) or getattr(llm_config, "model", None),
    }

    dataset_entries: List[Dict[str, object]] = []
    total_rows = 0
    processed_rows = 0
    product_rows_total = 0
    llm_calls = 0
    llm_failures = 0
    llm_cost_total = 0.0
    for name, res in results.items():
        if name == "combined":
            continue
        m = res.metrics or {}
        entry = {
            "dataset": name,
            "source_path": m.get("source_path"),
            "limit": m.get("limit"),
            "total_rows": m.get("total_rows"),
            "processed_rows": m.get("processed_rows"),
            "product_rows": m.get("product_rows"),
            "llm_calls": m.get("llm_calls"),
            "llm_failures": m.get("llm_failures"),
            "llm_elapsed_seconds": m.get("llm_elapsed_seconds"),
            "llm_cost_total": m.get("llm_cost_total", 0.0),
            "processing_seconds": m.get("processing_seconds"),
            "avg_iter_seconds": m.get("avg_iter_seconds"),
            "workers": m.get("workers"),
        }
        dataset_entries.append(entry)
        total_rows += int(m.get("total_rows", 0) or 0)
        processed_rows += int(m.get("processed_rows", 0) or 0)
        product_rows_total += int(m.get("product_rows", 0) or 0)
        llm_calls += int(m.get("llm_calls", 0) or 0)
        llm_failures += int(m.get("llm_failures", 0) or 0)
        try:
            llm_cost_total += float(m.get("llm_cost_total", 0.0) or 0.0)
        except Exception:
            pass

    mapping_info: Dict[str, object] = {}
    combined_res = results.get("combined")
    if combined_res and isinstance(combined_res.metrics, dict):
        mm = combined_res.metrics
        mapping_info = {
            "mapping_size": mm.get("product_mapping_size"),
            "mapping_cost": mm.get("product_mapping_cost"),
            "mapping_batches": mm.get("product_mapping_batches"),
            "mapping_lookup": mm.get("product_mapping_lookup"),
            "descriptions_size": mm.get("product_descriptions_size"),
            "descriptions_cost": mm.get("product_descriptions_cost"),
            "model": models.get("mapping"),
            "derived_taxonomy": mm.get("derived_taxonomy"),
            "product_descriptions": mm.get("product_descriptions"),
        }
        try:
            if mm.get("product_mapping_cost"):
                llm_cost_total += float(mm.get("product_mapping_cost"))
            if mm.get("product_descriptions_cost"):
                llm_cost_total += float(mm.get("product_descriptions_cost"))
        except Exception:
            pass

    # If we glued and processed as a single combined dataset, fill summary from combined metrics
    if not dataset_entries and combined_res and isinstance(combined_res.metrics, dict):
        cm = combined_res.metrics
        total_rows = int(cm.get("total_rows", 0) or 0)
        processed_rows = int(cm.get("processed_rows", 0) or 0)
        product_rows_total = int(cm.get("product_rows", 0) or 0)
        llm_calls = int(cm.get("llm_calls", 0) or 0)
        llm_failures = int(cm.get("llm_failures", 0) or 0)
        try:
            llm_cost_total += float(cm.get("llm_cost_total", 0.0) or 0.0)
        except Exception:
            pass
        # Optionally include combined as a single dataset entry for UI display
        dataset_entries.append({
            "dataset": "combined",
            "source_path": cm.get("source_path"),
            "limit": cm.get("limit"),
            "total_rows": total_rows,
            "processed_rows": processed_rows,
            "product_rows": product_rows_total,
            "llm_calls": llm_calls,
            "llm_failures": llm_failures,
            "llm_elapsed_seconds": cm.get("llm_elapsed_seconds"),
            "llm_cost_total": cm.get("llm_cost_total", 0.0),
            "processing_seconds": cm.get("processing_seconds"),
            "avg_iter_seconds": cm.get("avg_iter_seconds"),
            "workers": cm.get("workers"),
        })

    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_start))
    end_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_start + total_seconds))
    mapping_batches = int(mapping_info.get("mapping_batches") or 0)
    description_calls = int(mapping_info.get("descriptions_size") or 0)

    report = {
        "started_at": start_iso,
        "finished_at": end_iso,
        "duration_seconds": round(float(total_seconds), 3),
        "mode": mode,
        "models": models,
        "summary": {
            "datasets": max(1 if combined_res else 0, len([k for k in results.keys() if k != "combined"])) ,
            "total_rows": total_rows,
            "processed_rows": processed_rows,
            "product_rows": product_rows_total,
            "llm_calls": llm_calls + mapping_batches + description_calls,
            "llm_failures": llm_failures,
            "cost_total": round(float(llm_cost_total), 6),
        },
        "by_dataset": dataset_entries,
        "mapping": mapping_info,
    }

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(run_start))
    report_path = processed_dir / f"run_report_{ts}.json"
    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        LOGGER.info("JSON-отчёт сохранён: %s", report_path)
    except Exception as exc:
        LOGGER.warning("Не удалось сохранить JSON-отчёт: %s", exc)

    return combined_df, results


def _cli(argv: Iterable[str] | None = None) -> int:
    _ = argv  # reserved for future CLI args
    run_reviews(configure_logging=True)
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
