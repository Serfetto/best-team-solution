from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Reviews Microservice UI", layout="wide")

DEFAULT_API_URL = os.getenv("REVIEWS_API_URL", "http://127.0.0.1:8000")


def api_get(base: str, path: str) -> requests.Response:
    url = f"{base.rstrip('/')}{path}"
    return requests.get(url, timeout=60)


def api_put(base: str, path: str, payload: Dict[str, Any]) -> requests.Response:
    url = f"{base.rstrip('/')}{path}"
    return requests.put(url, json=payload, timeout=60)


def api_post(base: str, path: str, payload: Dict[str, Any], timeout: Optional[float] = None) -> requests.Response:
    url = f"{base.rstrip('/')}{path}"
    return requests.post(url, json=payload, timeout=timeout)


st.title("Reviews Microservice - Test UI")

with st.sidebar:
    st.header("Server")
    api_base = st.text_input("API base URL", value=DEFAULT_API_URL, help="FastAPI base URL (e.g., http://127.0.0.1:8000)")
    st.caption("Endpoints: /process, /extract, /insights/all-product-description, /insights/all-product-summary, /analyze, /taxonomy, /report")



st.header("Process New Reviews")
process_btn = st.button("Run /process", type="primary")
if process_btn:
    try:
        with st.spinner("Processing new reviews..."):
            resp = api_post(api_base, "/process", {})
        if resp.ok:
            data = resp.json()
            st.success(data.get("message") or "Process completed")
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("New reviews", data.get("new_reviews"))
                st.metric("Processed reviews", data.get("processed_reviews"))
            with metrics_cols[1]:
                st.metric("Processed products", data.get("processed_products"))
                st.metric("Total reviews", data.get("combined_total"))
            with metrics_cols[2]:
                st.metric("Total products", data.get("products_total"))
                st.metric("Elapsed (sec)", data.get("elapsed_seconds"))
            st.caption(f"Combined file: {data.get('combined_path')}")
            st.caption(f"Products file: {data.get('products_path')}")
        else:
            st.error(f"Process failed: {resp.status_code} {resp.text}")
    except Exception as exc:
        st.exception(exc)



st.header("Extract Taxonomy")
col_extract_from, col_extract_to = st.columns(2)
with col_extract_from:
    extract_date_from = st.text_input("Date from (YYYY-MM-DD)", value="", key="extract_date_from")
with col_extract_to:
    extract_date_to = st.text_input("Date to (YYYY-MM-DD)", value="", key="extract_date_to")
run_extract = st.button("Run /extract", type="primary")
if run_extract:
    payload = {
        "date_from": extract_date_from or None,
        "date_to": extract_date_to or None,
    }
    try:
        with st.spinner("Running extract..."):
            resp = api_post(api_base, "/extract", payload)
        if resp.ok:
            data = resp.json()
            st.success(data.get("message") or "Extract completed")
            metrics_extract = st.columns(3)
            with metrics_extract[0]:
                st.metric("Rows processed", data.get("processed_rows"))
                st.metric("Rows mapped", data.get("mapped_rows"))
            with metrics_extract[1]:
                st.metric("Added categories", data.get("added_categories"))
                st.metric("Full taxonomy size", data.get("full_size"))
            with metrics_extract[2]:
                st.metric("Base taxonomy size", data.get("base_size"))
                st.metric("Elapsed (sec)", data.get("elapsed_seconds"))
            st.caption(f"taxonomy_full: {data.get('taxonomy_full_path')}")
            base_path = data.get("taxonomy_base_path")
            if base_path:
                st.caption(f"taxonomy_base: {base_path}")
        else:
            st.error(f"Extract failed: {resp.status_code} {resp.text}")
    except Exception as exc:
        st.exception(exc)

st.header("Run Analysis")

col_mode, col_save, col_preview, col_caption = st.columns([1, 1, 1, 2])
with col_mode:
    mode = st.selectbox(
        "Mode",
        options=["base", "research", "research_guided"],
        help="base: taxonomy; research: freeform; research_guided: freeform + guided mapping",
    )
with col_save:
    save_excel = st.checkbox("Save Excel", value=True)
with col_preview:
    preview_rows = st.number_input("Preview rows", min_value=0, max_value=50, value=5, step=1)
with col_caption:
    st.caption("base - ����������; research - ��� ����������; research_guided - ������������ �� ���������� ������ �� ����� ��������")

run_btn = st.button("Run analyze", type="primary")

if run_btn:
    payload = {
        "mode": mode,
        "save_excel": bool(save_excel),
        "preview_rows": int(preview_rows),
    }
    try:
        with st.spinner("Running analysis..."):
            resp = api_post(api_base, "/analyze", payload, timeout=None)
        if resp.ok:
            data = resp.json()
            st.success("Analysis complete")

            report = data.get("report", {})
            st.subheader("Summary")
            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Duration (sec)", report.get("duration_seconds"))
                st.metric("Datasets", report.get("summary", {}).get("datasets"))
            with colB:
                st.metric("Processed rows", report.get("summary", {}).get("processed_rows"))
                st.metric("LLM calls", report.get("summary", {}).get("llm_calls"))
            with colC:
                st.metric("Cost total", report.get("summary", {}).get("cost_total"))
                st.write("Models:")
                st.json(report.get("models", {}))

            st.subheader("By dataset")
            ds = report.get("by_dataset", [])
            if isinstance(ds, list) and ds:
                st.dataframe(ds, use_container_width=True)
            else:
                st.info("No dataset entries")

            # Show derived taxonomy for research modes
            derived = report.get("mapping", {}).get("derived_taxonomy")
            if derived:
                st.subheader("Derived taxonomy (research)")
                if isinstance(derived, list) and derived and isinstance(derived[0], dict):
                    df_derived = pd.DataFrame(derived)
                    st.dataframe(df_derived, use_container_width=True)
                else:
                    st.json(derived)

            st.subheader("Report file")
            st.code(data.get("report_path"), language="text")
            st.caption(f"Processed dir: {data.get('processed_dir')}")
        else:
            st.error(f"Analyze failed: {resp.status_code} {resp.text}")
    except Exception as exc:
        st.exception(exc)




st.header("Product Insights")
products_cached = st.session_state.get("products_cached")
if products_cached is None:
    try:
        resp = api_get(api_base, "/insights/products")
        if resp.ok:
            data = resp.json()
            products_cached = data.get("products", [])
            st.session_state["products_cached"] = products_cached
        else:
            products_cached = []
    except Exception:
        products_cached = []

if not products_cached:
    st.info("Запустите анализ, чтобы сформировать продуктовую таблицу и получить инсайты.")
else:
    col_ps1, col_ps2 = st.columns(2)
    with col_ps1:
        selected_product = st.selectbox("Product", products_cached, key="insights_product")
    with col_ps2:
        date_range_col1, date_range_col2 = st.columns(2)
    with date_range_col1:
        summary_date_from = st.text_input("Date from (YYYY-MM-DD)", value="", key="insights_date_from")
    with date_range_col2:
        summary_date_to = st.text_input("Date to (YYYY-MM-DD)", value="", key="insights_date_to")
    run_summary = st.button("Generate product summary", type="primary", key="run_product_summary")
    if run_summary:
        payload = {
            "product_name": selected_product,
            "date_from": summary_date_from or None,
            "date_to": summary_date_to or None,
        }
        summary_resp = None
        description_resp = None
        desc_error = None
        try:
            with st.spinner("Generating product insights..."):
                summary_resp = api_post(api_base, "/insights/product-summary", payload, timeout=None)
                if summary_resp.ok:
                    try:
                        description_resp = api_post(api_base, "/insights/product-description", payload, timeout=None)
                        if not description_resp.ok:
                            desc_error = f"{description_resp.status_code} {description_resp.text}"
                    except Exception as desc_exc:
                        desc_error = str(desc_exc)
        except Exception as exc:
            st.exception(exc)
            summary_resp = None

        if summary_resp is None:
            pass
        elif not summary_resp.ok:
            st.error(f"Insights request failed: {summary_resp.status_code} {summary_resp.text}")
        else:
            result = summary_resp.json()
            description_data = None
            if description_resp is not None and description_resp.ok:
                description_data = description_resp.json()
            elif description_resp is not None and not description_resp.ok and not desc_error:
                desc_error = f"{description_resp.status_code} {description_resp.text}"

            st.success("Summary generated")

            if description_data:
                st.subheader("Product description")
                desc_text = description_data.get("description") or "No description available"
                st.write(desc_text)
                examples = description_data.get("examples") or []
                if examples:
                    with st.expander("Product raw examples", expanded=False):
                        st.write("\n".join(f"- {item}" for item in examples))
                mentions_total = description_data.get("total_mentions")
                cost_value = description_data.get("llm_cost")
                caption_parts = []
                if isinstance(mentions_total, int) and mentions_total > 0:
                    caption_parts.append(f"{mentions_total} unique raw mentions")
                try:
                    if cost_value:
                        caption_parts.append(f"LLM cost: {float(cost_value):.6f}")
                except Exception:
                    pass
                if caption_parts:
                    st.caption(" | ".join(caption_parts))
            elif desc_error:
                st.warning(f"Description unavailable: {desc_error}")

            summary_cols = st.columns(2)
            with summary_cols[0]:
                st.subheader("Strengths summary")
                st.write(result.get("strengths_summary") or "No data available")
            with summary_cols[1]:
                st.subheader("Weaknesses summary")
                st.write(result.get("weaknesses_summary") or "No data available")

            st.subheader("Details")

            product_label = result.get("product_name") or selected_product
            if product_label:
                st.caption(f"Product: {product_label}")

            strengths_nested = result.get("strengths") or []
            weaknesses_nested = result.get("weaknesses") or []

            def normalize_groups(data):
                normalized = []
                for bucket in data:
                    if isinstance(bucket, (list, tuple, set)):
                        items = [str(item).strip() for item in bucket if str(item).strip()]
                    elif bucket is None:
                        items = []
                    else:
                        value = str(bucket).strip()
                        items = [value] if value else []
                    normalized.append(items)
                return normalized

            strengths_normalized = normalize_groups(strengths_nested)
            weaknesses_normalized = normalize_groups(weaknesses_nested)

            strengths_flat = [item for group in strengths_normalized for item in group]
            weaknesses_flat = [item for group in weaknesses_normalized for item in group]

            strength_groups = [group for group in strengths_normalized if group]
            weakness_groups = [group for group in weaknesses_normalized if group]

            metrics_cols = st.columns(2)
            with metrics_cols[0]:
                st.metric("Reviews with strengths", len(strength_groups))
                st.metric("Strength mentions", len(strengths_flat))
            with metrics_cols[1]:
                st.metric("Reviews with weaknesses", len(weakness_groups))
                st.metric("Weakness mentions", len(weaknesses_flat))

            display_cols = st.columns(2)
            with display_cols[0]:
                with st.expander("Strength mentions", expanded=False):
                    if strengths_flat:
                        st.write("\n".join(f"- {item}" for item in strengths_flat))
                    else:
                        st.write("No data")
            with display_cols[1]:
                with st.expander("Weakness mentions", expanded=False):
                    if weaknesses_flat:
                        st.write("\n".join(f"- {item}" for item in weaknesses_flat))
                    else:
                        st.write("No data")

            preview_limit = 200
            strengths_formatted = ["; ".join(group) for group in strength_groups]
            weaknesses_formatted = ["; ".join(group) for group in weakness_groups]

            raw_cols = st.columns(2)
            with raw_cols[0]:
                with st.expander("Strengths by review (newest first)", expanded=False):
                    if strengths_formatted:
                        preview_items = strengths_formatted[:preview_limit]
                        st.write("\n".join(f"- {item}" for item in preview_items))
                        if len(strengths_formatted) > preview_limit:
                            st.caption(f"Showing first {preview_limit} of {len(strengths_formatted)} entries")
                    else:
                        st.write("No data")
            with raw_cols[1]:
                with st.expander("Weaknesses by review (newest first)", expanded=False):
                    if weaknesses_formatted:
                        preview_items = weaknesses_formatted[:preview_limit]
                        st.write("\n".join(f"- {item}" for item in preview_items))
                        if len(weaknesses_formatted) > preview_limit:
                            st.caption(f"Showing first {preview_limit} of {len(weaknesses_formatted)} entries")
                    else:
                        st.write("No data")

            st.download_button(
                "Download JSON",
                data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="product_summary.json",
                mime="application/json",
                use_container_width=True,
            )


st.header("All Product Insights")
col_all_from, col_all_to = st.columns(2)
with col_all_from:
    all_date_from = st.text_input("Date from (YYYY-MM-DD)", value="", key="all_products_date_from")
with col_all_to:
    all_date_to = st.text_input("Date to (YYYY-MM-DD)", value="", key="all_products_date_to")

col_all_desc, col_all_summary = st.columns(2)
with col_all_desc:
    run_all_desc = st.button("Run all descriptions", use_container_width=True)
with col_all_summary:
    run_all_summary = st.button("Run all summaries", use_container_width=True)

if run_all_desc:
    payload = {
        "date_from": all_date_from or None,
        "date_to": all_date_to or None,
        "was_processed": True,
        "was_date_changed": True,
    }
    try:
        with st.spinner("Generating descriptions..."):
            resp = api_post(api_base, "/insights/all-product-description", payload, timeout=None)
        if resp.ok:
            data = resp.json()
            st.success(f"Descriptions generated for {data.get('processed', 0)} products")
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("Processed", data.get("processed"))
            with metrics_cols[1]:
                st.metric("Total cost", data.get("total_cost"))
            with metrics_cols[2]:
                st.metric("Elapsed (sec)", data.get("elapsed_seconds"))
            st.caption(f"taxonomy_full: {data.get('taxonomy_full_path')}")
            with st.expander("Description payload", expanded=False):
                st.json(data.get("items", []))
        else:
            st.error(f"All descriptions failed: {resp.status_code} {resp.text}")
    except Exception as exc:
        st.exception(exc)

if run_all_summary:
    payload = {
        "date_from": all_date_from or None,
        "date_to": all_date_to or None,
        "was_processed": True,
        "was_date_changed": True,
    }
    try:
        with st.spinner("Generating summaries..."):
            resp = api_post(api_base, "/insights/all-product-summary", payload, timeout=None)
        if resp.ok:
            data = resp.json()
            st.success(f"Summaries generated for {data.get('processed', 0)} products")
            metrics_cols = st.columns(2)
            with metrics_cols[0]:
                st.metric("Processed", data.get("processed"))
            with metrics_cols[1]:
                st.metric("Elapsed (sec)", data.get("elapsed_seconds"))
            st.caption(f"taxonomy_full: {data.get('taxonomy_full_path')}")
            with st.expander("Summary payload", expanded=False):
                st.json(data.get("items", []))
        else:
            st.error(f"All summaries failed: {resp.status_code} {resp.text}")
    except Exception as exc:
        st.exception(exc)

st.header("Batch JSON Analysis")
with st.expander("Input format", expanded=False):
    st.markdown(
        """Expected JSON payload:
```json
{
  "data": [
    {"id": 1, "text": "Review text..."},
    {"id": 2, "text": "Another review..."}
  ]
}
```
"""
    )

upload_col, sample_col = st.columns([3, 1])
with upload_col:
    uploaded_file = st.file_uploader("Upload JSON", type=["json"], key="batch_upload")
with sample_col:
    use_sample = st.button("Use sample", use_container_width=True)

if use_sample:
    sample_path = Path(__file__).resolve().parents[1] / "samples" / "batch_reviews.json"
    try:
        sample_payload = json.loads(sample_path.read_text(encoding="utf-8"))
        st.session_state["batch_payload"] = sample_payload
        st.success("Loaded sample from samples/batch_reviews.json")
    except Exception as exc:
        st.error(f"Failed to read sample: {exc}")

if uploaded_file is not None:
    try:
        payload = json.loads(uploaded_file.getvalue().decode("utf-8"))
        st.session_state["batch_payload"] = payload
        st.success("File uploaded")
    except Exception as exc:
        st.error(f"Failed to parse file: {exc}")

batch_payload = st.session_state.get("batch_payload")
run_batch = st.button("Run batch analysis", type="primary")

if run_batch:
    if not isinstance(batch_payload, dict) or "data" not in batch_payload:
        st.error("Upload a valid JSON before running")
    else:
        try:
            with st.spinner("Calling API..."):
                resp = api_post(api_base, "/analyze/batch", batch_payload, timeout=None)
            if resp.ok:
                result = resp.json()
                st.success("Batch analysis completed")
                metrics = result.get("metrics", {})
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Duration (sec)", metrics.get("duration_seconds"))
                    st.metric("Processed", metrics.get("processed"))
                with col_m2:
                    st.metric("LLM calls", metrics.get("llm_calls"))
                    st.metric("LLM failures", metrics.get("llm_failures"))
                with col_m3:
                    st.metric("Cost total", metrics.get("cost_total"))

                predictions = result.get("predictions", [])
                if predictions:
                    st.subheader("Predictions")
                    st.dataframe(predictions, use_container_width=True)
                    st.download_button(
                        "Download result",
                        data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name="batch_predictions.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                else:
                    st.info("Response does not contain predictions")
            else:
                st.error(f"Batch analysis failed: {resp.status_code} {resp.text}")
        except Exception as exc:
            st.exception(exc)



st.header("Taxonomy Editor")
tax_actions = st.columns([1, 1, 2])
with tax_actions[0]:
    if st.button("Load from server", use_container_width=True):
        try:
            resp = api_get(api_base, "/taxonomy")
            if resp.ok:
                tj = resp.json()
                st.session_state["taxonomy_items"] = tj.get("items", [])
                st.session_state["taxonomy_path"] = tj.get("path")
                st.success("Taxonomy loaded")
            else:
                st.error(f"Failed to load taxonomy: {resp.status_code} {resp.text}")
        except Exception as exc:
            st.exception(exc)
with tax_actions[1]:
    if st.button("Refresh", use_container_width=True):
        try:
            resp = api_get(api_base, "/taxonomy")
            if resp.ok:
                tj = resp.json()
                st.session_state["taxonomy_items"] = tj.get("items", [])
                st.session_state["taxonomy_path"] = tj.get("path")
            else:
                st.error(f"Failed to refresh taxonomy: {resp.status_code} {resp.text}")
        except Exception as exc:
            st.exception(exc)

# Auto-load once on first run
if "taxonomy_items" not in st.session_state:
    try:
        resp = api_get(api_base, "/taxonomy")
        if resp.ok:
            tj = resp.json()
            st.session_state["taxonomy_items"] = tj.get("items", [])
            st.session_state["taxonomy_path"] = tj.get("path")
    except Exception:
        pass

items: List[str] = st.session_state.get("taxonomy_items", [])
path_info = st.session_state.get("taxonomy_path")
if path_info:
    st.caption(f"Taxonomy path: {path_info}")

# Show as editable table for convenience
df_tax = pd.DataFrame({"Категория": items}) if items else pd.DataFrame({"Категория": []})
edited = st.data_editor(
    df_tax,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key="tax_editor",
)

col_save1, col_save2 = st.columns([1, 1])
with col_save1:
    if st.button("Save to server", type="primary", use_container_width=True):
        new_items = [str(v).strip() for v in edited["Категория"].tolist() if str(v).strip()]
        if not new_items:
            st.warning("Provide at least one taxonomy item")
        else:
            try:
                resp = api_put(api_base, "/taxonomy", {"items": new_items})
                if resp.ok:
                    tj = resp.json()
                    st.session_state["taxonomy_items"] = tj.get("items", new_items)
                    st.session_state["taxonomy_path"] = tj.get("path")
                    st.success("Taxonomy updated on server")
                else:
                    st.error(f"Failed to update taxonomy: {resp.status_code} {resp.text}")
            except Exception as exc:
                st.exception(exc)
with col_save2:
    if st.button("Reload from server", use_container_width=True):
        try:
            resp = api_get(api_base, "/taxonomy")
            if resp.ok:
                tj = resp.json()
                st.session_state["taxonomy_items"] = tj.get("items", [])
                st.session_state["taxonomy_path"] = tj.get("path")
                st.experimental_rerun()
            else:
                st.error(f"Failed to reload taxonomy: {resp.status_code} {resp.text}")
        except Exception as exc:
            st.exception(exc)

with st.expander("Show taxonomy JSON"):
    st.json(st.session_state.get("taxonomy_items", []), expanded=True)


st.header("Latest Report Path")
if st.button("Get latest report path"):
    try:
        resp = api_get(api_base, "/report")
        if resp.ok:
            rj = resp.json()
            st.code(rj.get("report_path"), language="text")
            st.caption(f"Processed dir: {rj.get('processed_dir')}")
        else:
            st.error(f"No report found: {resp.status_code} {resp.text}")
    except Exception as exc:
        st.exception(exc)


st.header("Download Results")
dl_col1, dl_col2 = st.columns([1, 1])
with dl_col1:
    if st.button("Fetch combined CSV"):
        try:
            resp = api_get(api_base, "/export/combined")
            if resp.ok:
                st.session_state["combined_csv_bytes"] = resp.content
                st.success("Combined CSV fetched")
            else:
                st.error(f"Failed to fetch combined CSV: {resp.status_code} {resp.text}")
        except Exception as exc:
            st.exception(exc)

if "combined_csv_bytes" in st.session_state:
    st.download_button(
        label="Download combined.csv",
        data=st.session_state["combined_csv_bytes"],
        file_name="enriched_combined.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.subheader("Download per-dataset (optional)")
ds_name = st.text_input("Dataset name (e.g., banki.ru)", value="")
if st.button("Fetch dataset CSV"):
    if not ds_name.strip():
        st.warning("Enter dataset name")
    else:
        try:
            resp = api_get(api_base, f"/export/dataset/{ds_name.strip()}")
            if resp.ok:
                st.session_state["dataset_csv_bytes"] = resp.content
                st.session_state["dataset_csv_name"] = f"enriched_{ds_name.strip().replace('/', '_').replace(' ', '_')}.csv"
                st.success("Dataset CSV fetched")
            else:
                st.error(f"Failed to fetch dataset CSV: {resp.status_code} {resp.text}")
        except Exception as exc:
            st.exception(exc)

if "dataset_csv_bytes" in st.session_state:
    st.download_button(
        label="Download dataset.csv",
        data=st.session_state["dataset_csv_bytes"],
        file_name=st.session_state.get("dataset_csv_name", "dataset.csv"),
        mime="text/csv",
        use_container_width=True,
    )

st.subheader("Download derived taxonomy (research)")
if st.button("Fetch derived taxonomy"):
    try:
        resp = api_get(api_base, "/export/derived-taxonomy")
        if resp.ok:
            st.session_state["derived_taxonomy_bytes"] = resp.content
            st.success("Derived taxonomy fetched")
        else:
            st.error(f"Failed to fetch derived taxonomy: {resp.status_code} {resp.text}")
    except Exception as exc:
        st.exception(exc)

if "derived_taxonomy_bytes" in st.session_state:
    st.download_button(
        label="Download derived_taxonomy_combined.json",
        data=st.session_state["derived_taxonomy_bytes"],
        file_name="derived_taxonomy_combined.json",
        mime="application/json",
        use_container_width=True,
    )

