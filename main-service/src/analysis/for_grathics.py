from ast import literal_eval
import ast
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Tuple
import plotly.graph_objects as go
import pandas as pd
from wordcloud import WordCloud
from src.configs.config import settings
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE_MODEL = BASE_DIR.parents[2] / "hack-LCT" / "data"

DATA_PATH_NEW = DATA_FILE_MODEL / "enriched_products_combined.csv"
DATA_PATH_OLD = DATA_FILE_MODEL / "enriched_combined.csv"

CONFIG_FIG = {
    "displayModeBar": True,
    "responsive": True,
    "scrollZoom": True,
    "modeBarButtonsToRemove": [
        "lasso2d", 
        "select2d", 
        "zoomIn2d", 
        "zoomOut2d",
        "zoom2d",
        "pan2d",
        "autoScale2d",
        "resetScale2d"
    ],
    "displaylogo": False,
    "watermark": False,
}

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
    data_map: dict[str, float],
    title: str,
    y_title: str,
    uirevision: str,
    hovertemplate: str,
) -> go.Figure:
    """Линейный график временного ряда."""
    if not data_map:
        return _empty_figure("No data available")

    df = (
        pd.Series(data_map)
        .rename_axis("date")
        .reset_index(name="value")
        .assign(date=lambda frame: pd.to_datetime(frame["date"]))
        .sort_values("date")
        
    )

    fig = px.line(df, x="date", y="value", markers=True, title=title)
    fig.update_traces(mode="lines+markers", hovertemplate=hovertemplate)
    fig.update_layout(
        margin=dict(l=24, r=16, t=40, b=32),
        xaxis_title="Дата",
        yaxis_title=y_title,
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(255,255,255)",
        font=dict(color="#1f2937"),
        uirevision=uirevision,
        title=dict(
            text=title,
            x=0.5,
            y=0.98,
            xanchor="center",
            font=dict(size=25)
        ),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(15, 23, 42, 0.08)")
    return fig

def calc_data(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегированные метрики по отзывам."""
    count_days = df["date"].dt.date.nunique()
    sentiment_value_counts = df["sentiment"].value_counts().to_dict()
    procent_sentiment = df["sentiment"].value_counts(normalize=True).to_dict()

    #popular_products = df["product"].value_counts().head(10).sort_values(ascending=True).to_dict()

    count_reviews_per_day = df.groupby(df["date"].dt.date).size()
    count_reviews_per_day.index = count_reviews_per_day.index.astype(str)
    count_reviews_per_day = count_reviews_per_day.to_dict()

    mean_rating_per_day = df.groupby(df["date"].dt.date)["grade_extracted"].mean()
    mean_rating_per_day.index = mean_rating_per_day.index.astype(str)
    mean_rating_per_day = mean_rating_per_day.to_dict()

    mean_rating_all_data = df["grade_extracted"].mean() if "grade_extracted" in df else float("nan")

    return {
        "total_reviews": len(df),
        "count_days": count_days,
        "mean_rating_all_data": mean_rating_all_data,
        "sentiment_value_counts": sentiment_value_counts,
        "procent_sentiment": procent_sentiment,
        #"popular_products": popular_products,
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

def sentiment_pie_chart(procent_sentiment: dict[str, float]) -> go.Figure:
    if not procent_sentiment:
        return _empty_figure("Нет данных по тональностям")

    labels = list(procent_sentiment.keys())
    values = [round(v * 100, 1) for v in procent_sentiment.values()]

    fig = px.pie(
        names=labels,
        values=values,
        title="Распределение тональности",
        color=labels,
        color_discrete_map={
            "positive": "#16a34a",
            "negative": "#ef4444",
            "neutral": "#6b7280"
        }
    )
    fig.update_traces(
        textinfo="label+percent", 
        pull=[0.05, 0.05, 0.05], 
        showlegend=False
    )
    fig.update_layout(
        margin=dict(l=24, r=16, t=50, b=32),
        paper_bgcolor="rgba(255,255,255)",
        font=dict(color="#1f2937"),
        title=dict(
            text="Распределение тональности",
            x=0.5,
            y=1,
            xanchor="center",
            font=dict(size=25)
        ),
    )
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
        background_color="white"
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
            font=dict(size=25)
        ),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        margin=dict(l=0, r=0, t=30, b=0),  # сверху оставляем место под заголовок
        hovermode=False,
        dragmode=False,
    )
    return fig

def load_reviews_dataframe(path: str | None = None) -> pd.DataFrame:
    """Загружает отзывы из Excel и приводит поля к нужному виду."""
    target_path = DATA_PATH_NEW if path is None else path
    df = pd.read_csv(target_path)

    # фильтруем "другой"
    df = df[~df["product"].isin(["другой", "другое", "Другой", "Другое"])]

    df["date"] = pd.to_datetime(df["posted_at"], errors="coerce")
    df["product_strengths"] = df["strengths"].apply(_safe_literal_eval)
    df["product_weaknesses"] = df["weaknesses"].apply(_safe_literal_eval)
    return df