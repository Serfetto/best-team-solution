# -*- coding: utf-8 -*-
from __future__ import annotations

from ast import literal_eval
import ast
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Tuple

import json
import logging
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html
from wordcloud import WordCloud

LOGGER = logging.getLogger(__name__)

# === CONFIG ===
INTERVAL = 60_000

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE_MODEL = BASE_DIR.parents[1] / "model-service" / "data"

DATA_PATH_NEW = DATA_FILE_MODEL / "enriched_products_combined.csv"
DATA_PATH_OLD = DATA_FILE_MODEL / "enriched_combined.csv"

ASSETS_FOLDER = str(BASE_DIR / "assets")

CONFIG_FIG = {
    "displayModeBar": True,
    "responsive": True,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "zoomIn2d", "zoomOut2d"],
    "displaylogo": False,
    "watermark": False,
}

REQUIRED_REVIEW_COLUMNS = [
    "posted_at",
    "product",
    "sentiment",
    "strengths",
    "weaknesses",
    "grade_extracted",
]


def _empty_reviews_dataframe() -> pd.DataFrame:
    """Return an empty dataframe with the expected review columns."""
    return pd.DataFrame(columns=REQUIRED_REVIEW_COLUMNS)

API_SERVICE_BASE_URL = os.getenv("REVIEWS_API_BASE_URL", "http://localhost:8002").rstrip("/")
ALL_PRODUCT_SUMMARY = f"{API_SERVICE_BASE_URL}/insights/all-product-summary"
ALL_PRODUCT_DESCRIPTION = f"{API_SERVICE_BASE_URL}/insights/all-product-description"
EXTRACT = f"{API_SERVICE_BASE_URL}/extract"

UNICODE_SPACE = [
    "\u0020", "\u00A0", "\u2000", "\u2001", "\u2002", "\u2003",
    "\u2004", "\u2005", "\u2006", "\u2007", "\u2008", "\u2009",
    "\u200A", "\u2028", "\u205F", "\u3000",
]

# === HELPERS ===
def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def to_pct(x: float) -> float:
    """Преобразует долю в проценты (0.0–1.0 → 0–100)."""
    return x * 100 if 0.0 <= x <= 1.0 else x


def pick_share(normalized_shares: dict[str, float], *aliases: str) -> float:
    """Выбирает значение по одному из возможных ключей."""
    for alias in aliases:
        if alias in normalized_shares:
            return normalized_shares[alias]
    return 0.0


def _safe_literal_eval(value: Any) -> list[str]:
    """Аккуратно парсит строки в списки (например, strengths/weaknesses)."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str) and not value.strip():
        return []
    try:
        parsed = literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return []
    if isinstance(parsed, list):
        return parsed
    return [parsed]

# === PLOTTING HELPERS ===
def _sentiment_item(label: str, value_pct: float, color_hex: str) -> html.Div:
    """Элемент для отображения процента тональности с полоской."""
    bar_bg_style = {
        "width": "100%", "height": "8px", "borderRadius": "9999px",
        "background": "rgba(0,0,0,0.06)", "overflow": "hidden", "marginTop": "6px",
    }
    bar_fill_style = {
        "width": f"{max(0.0, min(100.0, value_pct))}%",
        "height": "100%", "backgroundColor": color_hex,
        "borderRadius": "9999px", "transition": "width 300ms ease",
    }
    row_style = {
        "display": "flex", "justifyContent": "space-between",
        "alignItems": "baseline", "gap": "8px",
    }
    return html.Div(
        [
            html.Div(
                [html.Span(label, className="sentiment-caption"),
                 html.Span(f"{value_pct:.1f}%", className="sentiment-value")],
                style=row_style,
            ),
            html.Div(html.Div(style=bar_fill_style), style=bar_bg_style),
        ],
        className="sentiment-item",
    )


def _empty_figure(message: str) -> go.Figure:
    """Пустая заглушка для графика."""
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=24, r=16, t=32, b=24),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(text=message, x=0.5, y=0.5,
                          xref="paper", yref="paper", showarrow=False)],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1f2937"),
        height=320,
    )
    return fig


def _time_series_figure(
    series_map: Mapping[str, dict[str, float]] | dict[str, float],
    title: str,
    y_title: str,
    uirevision: str,
    hovertemplate: str,
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    """Строим таймсериз для одной или нескольких линий."""
    if not series_map:
        return _empty_figure("Данные не найдены")

    if isinstance(series_map, dict) and series_map and not any(isinstance(v, dict) for v in series_map.values()):
        series_map = {"Общий тренд": series_map}  # type: ignore[assignment]

    records: list[pd.DataFrame] = []
    for series_name, data_map in series_map.items():
        if not data_map:
            continue
        series_df = (
            pd.Series(data_map, dtype=float)
            .rename_axis("date")
            .reset_index(name="value")
            .assign(date=lambda frame: pd.to_datetime(frame["date"]))
            .sort_values("date")
        )
        series_df["series"] = series_name
        records.append(series_df)

    if not records:
        return _empty_figure("Данные не найдены")

    df = pd.concat(records, ignore_index=True)
    series_order = list(dict.fromkeys(df["series"]))

    fig = px.line(
        df,
        x="date",
        y="value",
        color="series",
        markers=True,
        title=title,
        color_discrete_map=color_map or {},
        category_orders={"series": series_order},
    )
    fig.update_traces(mode="lines+markers", hovertemplate=hovertemplate)
    if series_order:
        trace_by_name = {trace.name: trace for trace in fig.data}
        fig.data = tuple(trace_by_name[name] for name in series_order if name in trace_by_name)
    fig.update_layout(
        margin=dict(l=24, r=16, t=40, b=32),
        xaxis_title="Дата",
        yaxis_title=y_title,
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(255,255,255)",
        font=dict(color="#1f2937"),
        uirevision=uirevision,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(15, 23, 42, 0.08)")
    return fig

def _bar_figure(data_map: dict[str, float], title: str, x_title: str, uirevision: str) -> go.Figure:
    """Обычный bar-chart (вертикальный)."""
    if not data_map:
        return _empty_figure("Данные не найдены")

    df = (
        pd.Series(data_map)
        .sort_values(ascending=False)
        .rename_axis("label")
        .reset_index(name="value")
    )
    fig = px.bar(df, y="label", x="value", text="value", title=title, orientation="h")
    fig.update_traces(textposition="outside")
    fig.update_layout(
        margin=dict(l=24, r=16, t=40, b=40),
        xaxis_title="Count", yaxis_title=x_title,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(255,255,255)",
        font=dict(color="#1f2937"),
        uirevision=uirevision, height=320,
    )
    fig.update_yaxes(type="category")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(15, 23, 42, 0.08)")
    return fig

def generate_cloud(df: pd.DataFrame):
    df["parsed"] = df["emotional_tags"].apply(lambda x: ast.literal_eval(x))

    df_exploded = df.explode("parsed").dropna(subset=["parsed"])
    
    data = df_exploded["parsed"].value_counts()

    if not len(data):
        return None
    
    d = {word: count for word, count in data.items()}
    
    wc = WordCloud(
        width=1280,
        height=500,
        scale=1,
        background_color="#fff7fa"
    )
    wc.fit_words(d)
    img_array = wc.to_array()
    
    fig = go.Figure(go.Image(z=img_array, hoverinfo="skip"))
    fig.update_layout(
        title=dict(
            text="Облако слов",
            x=0.5,
            y=0.97,
            xanchor="center",
            font=dict(size=25),
        ),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        margin=dict(l=0, r=0, t=50, b=0),  # сверху оставляем место под заголовок
        hovermode=False,
        dragmode=False,
    )
    
    return fig


def _horizontal_bar(
    pairs: dict[str, float],
    title: str,
    x_title: str,
    uirevision: str,
    bar_color: str = "#3b82f6",
    selected_labels: list[str] | None = None,
    series_colors: Mapping[str, str] | None = None,
) -> go.Figure:
    """Интерактивный bar-chart с подсветкой выбранных продуктов."""
    if not pairs:
        return _empty_figure("Данные не найдены")

    df = pd.DataFrame({"label": list(pairs.keys()), "value": list(pairs.values())})
    labels = df["label"].astype(str)

    color_lookup: dict[str, str] = {}
    if series_colors:
        for raw_key, raw_color in series_colors.items():
            if not isinstance(raw_key, str) or not isinstance(raw_color, str):
                continue
            trimmed_key = raw_key.strip()
            if not trimmed_key:
                continue
            color_lookup.setdefault(trimmed_key, raw_color)
            color_lookup.setdefault(trimmed_key.lower(), raw_color)

    def resolve_color(label_value: str) -> str:
        if not isinstance(label_value, str):
            return bar_color
        trimmed = label_value.strip()
        if not trimmed:
            return bar_color
        return color_lookup.get(trimmed, color_lookup.get(trimmed.lower(), bar_color))

    selected_norms: set[str] = set()
    if selected_labels:
        selected_norms = {
            label.strip().lower()
            for label in selected_labels
            if isinstance(label, str) and label.strip()
        }

    if selected_norms:
        dim_color = "rgba(148, 163, 184, 0.35)"
        colors = [
            resolve_color(label) if label.strip().lower() in selected_norms else dim_color
            for label in labels
        ]
    else:
        colors = [resolve_color(label) for label in labels]

    fig = px.bar(df, x="value", y="label", orientation="h", text="value", title=title)
    fig.update_traces(
        textposition="auto",
        hovertemplate="Товар: %{y}<br>Кол-во отзывов: %{x}",
        marker=dict(color=colors),
    )
    fig.update_layout(
        margin=dict(l=160, r=16, t=50, b=32),
        xaxis_title="", yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(255,255,255)",
        font=dict(color="#1f2937"),
        uirevision=uirevision,
        autosize=False,
        height=max(6, len(df) / 2) * 100,
        width=1000,
    )
    fig.update_yaxes(automargin=True, ticklabelposition="outside left", ticklabelstandoff=10)
    fig.update_xaxes(visible=False, showgrid=False)
    return fig

# === DATA HELPERS ===
def load_reviews_dataframe(path: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed review datasets, tolerating missing files."""
    target_path = DATA_PATH_NEW if path is None else Path(path)
    if not isinstance(target_path, Path):
        target_path = Path(target_path)

    if target_path.exists():
        df = pd.read_csv(target_path)
    else:
        LOGGER.warning("Processed products dataset not found at %s", target_path)
        df = _empty_reviews_dataframe()

    if DATA_PATH_OLD.exists():
        df1 = pd.read_csv(DATA_PATH_OLD)
    else:
        LOGGER.warning("Processed combined dataset not found at %s", DATA_PATH_OLD)
        df1 = _empty_reviews_dataframe()

    if "product" in df.columns:
        df = df[~df["product"].isin(["Удалено", "Удален", "Удалена", "Удалены"])]

    df["date"] = pd.to_datetime(df.get("posted_at"), errors="coerce")
    df1["date"] = pd.to_datetime(df1.get("posted_at"), errors="coerce")

    if "strengths" in df.columns:
        strengths = df["strengths"].apply(_safe_literal_eval)
    else:
        strengths = pd.Series([[] for _ in range(len(df))], index=df.index)
    if "weaknesses" in df.columns:
        weaknesses = df["weaknesses"].apply(_safe_literal_eval)
    else:
        weaknesses = pd.Series([[] for _ in range(len(df))], index=df.index)

    df["product_strengths"] = strengths
    df["product_weaknesses"] = weaknesses

    if "grade_extracted" not in df.columns:
        df["grade_extracted"] = pd.NA
    if "grade_extracted" not in df1.columns:
        df1["grade_extracted"] = pd.NA

    return df, df1

def get_base_full() -> tuple[int, int]:
    """Количество уникальных продуктов в base и разница с full."""
    with open(DATA_FILE_MODEL / "taxonomy_base.json", "r", encoding="utf-8") as f:
        base = json.load(f)

    with open(DATA_FILE_MODEL / "taxonomy_full.json", "r", encoding="utf-8") as f:
        full = json.load(f)

    base_keys = set(base)
    full_keys = set(full.keys())
    # print(base_keys, full_keys)
    return len(base_keys), len(full_keys - base_keys)


def calc_data(df: pd.DataFrame, df1: pd.DataFrame) -> dict[str, Any]:
    """Агрегированные метрики по отзывам."""
    count_days = df["date"].dt.date.nunique()
    sentiment_value_counts = df["sentiment"].value_counts().to_dict()
    procent_sentiment = df["sentiment"].value_counts(normalize=True).to_dict()

    popular_products = df["product"].value_counts().sort_values(ascending=True).to_dict()

    count_reviews_per_day = df.groupby(df["date"].dt.date).size()
    count_reviews_per_day.index = count_reviews_per_day.index.astype(str)
    count_reviews_per_day = count_reviews_per_day.to_dict()

    mean_rating_per_day = df.groupby(df["date"].dt.date)["grade_extracted"].mean()
    mean_rating_per_day.index = mean_rating_per_day.index.astype(str)
    mean_rating_per_day = mean_rating_per_day.to_dict()

    mean_rating_all_data = df["grade_extracted"].mean() if "grade_extracted" in df else float("nan")

    return {
        "total_reviews": len(df1),
        "count_days": count_days,
        "mean_rating_all_data": df1["grade_extracted"].mean(),
        "sentiment_value_counts": sentiment_value_counts,
        "procent_sentiment": procent_sentiment,
        "popular_products": popular_products,
        "count_reviews_per_day": count_reviews_per_day,
        "mean_rating_per_day": mean_rating_per_day,
    }

# === TIME SERIES AGGREGATION ===
def build_series_by_mode(
    data_map: dict[str, float],
    mode: str,        # 'year' | 'month' | 'week' | 'day'
    agg: str = "sum", # 'sum' для счётчиков, 'mean' для средних
) -> dict[str, float]:
    """Превращает дневные данные в серию нужной дискретизации."""
    if not data_map:
        return {}

    s = (
        pd.Series(data_map, dtype=float)
        .rename_axis("date")
        .reset_index(name="value")
    )
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").set_index("date")["value"]

    # дневная сетка
    start_day, end_day = s.index.min().normalize(), s.index.max().normalize()
    daily_idx = pd.date_range(start_day, end_day, freq="D")
    daily = s.reindex(daily_idx)
    if agg == "sum":
        daily = daily.fillna(0.0)

    if mode == "year":
        last_year = daily.index.max().year
        y_start, y_end = pd.Timestamp(year=last_year, month=1, day=1), pd.Timestamp(year=last_year, month=12, day=31)
        year_slice = daily.loc[y_start:y_end] if not daily.empty else daily
        monthly = year_slice.resample("MS").sum() if agg == "sum" else year_slice.resample("MS").mean()
        monthly = monthly.reindex(pd.date_range(y_start, y_end, freq="MS"))
        if agg == "sum":
            monthly = monthly.fillna(0.0)
        return monthly.to_dict()

    if mode == "week":
        week_start = (start_day - pd.offsets.Week(weekday=0)).normalize()
        week_end = (end_day + pd.offsets.Week(weekday=6)).normalize()
        idx = pd.date_range(week_start, week_end, freq="D")
        series = daily.reindex(idx)
        if agg == "sum":
            series = series.fillna(0.0)
        return series.to_dict()

    if mode == "month":
        m_start = daily.index.min().to_period("M").start_time
        m_end = daily.index.max().to_period("M").start_time
        month_starts = pd.date_range(m_start, m_end, freq="MS")

        out_idx, out_vals = [], []
        for ms in month_starts:
            me = ms + pd.offsets.MonthBegin(1)
            bounds = pd.date_range(ms, me, periods=5)  # 4 интервала
            for i in range(4):
                left, right = bounds[i], bounds[i + 1] - pd.Timedelta(days=1)
                chunk = daily.loc[left.normalize():right.normalize()]
                val = float(chunk.sum()) if agg == "sum" else float(chunk.mean()) if not chunk.empty else float("nan")
                out_idx.append(left + (bounds[i + 1] - left) / 2)
                out_vals.append(val)
        return dict(zip(out_idx, out_vals))

    if mode == "day":
        out_idx, out_vals = [], []
        for d in pd.date_range(start_day, end_day, freq="D"):
            day_val = daily.loc[d]
            edges = pd.date_range(d, d + pd.Timedelta(days=1), periods=7)
            for i in range(6):
                mid = edges[i] + (edges[i+1] - edges[i]) / 2
                val = float(day_val) if agg == "mean" else float(day_val)
                out_idx.append(mid)
                out_vals.append(val)
        return dict(zip(out_idx, out_vals))

    return daily.to_dict()
