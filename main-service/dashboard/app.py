# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime, date
import json
import os
import random
from itertools import cycle

from urllib.parse import parse_qs

import httpx
import pandas as pd

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from utils import (
    _empty_figure,
    _horizontal_bar,
    _sentiment_item,
    _time_series_figure,
    generate_cloud,
    get_base_full,
    load_reviews_dataframe,
    calc_data,
    pick_share,
    to_pct,
    build_series_by_mode,
    INTERVAL,
    ASSETS_FOLDER,
    CONFIG_FIG,
    ALL_PRODUCT_SUMMARY,
    ALL_PRODUCT_DESCRIPTION,
    EXTRACT,
    UNICODE_SPACE,
)

MAIN_SERVICE_BASE_URL = os.getenv("MAIN_SERVICE_BASE_URL", "http://localhost:8003").rstrip("/")

OVERALL_SERIES_NAME = "Общий тренд"
OVERALL_SERIES_COLOR = "#1f2937"
PRODUCT_SERIES_COLORS: tuple[str, ...] = (
    "rgba(244, 189, 199, 0.99999999999999999999)",
    "rgba(190, 227, 248, 0.999999999999999999999)",
    "rgba(255, 223, 186, 0.999999999999999999999)",
    "rgba(186, 230, 201, 0.999999999999999999999)",
    "rgba(221, 214, 254, 0.999999999999999999999)",
    "rgba(255, 214, 220, 0.999999999999999999999)",
    "rgba(254, 249, 195, 0.999999999999999999999)",
    "rgba(187, 247, 208, 0.999999999999999999999)",
    "rgba(224, 231, 255, 0.999999999999999999999)",
    "rgba(209, 250, 229, 0.999999999999999999999)",
    "rgba(252, 216, 244, 0.999999999999999999999)",
    "rgba(214, 226, 251, 0.999999999999999999999)",
    "rgba(255, 240, 219, 0.999999999999999999999)",
    "rgba(219, 234, 254, 0.999999999999999999999)",
)

# === APP INIT ===
app = Dash(__name__, assets_folder=ASSETS_FOLDER)

app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>Общий анализ</title>
    {%favicon%}
    {%css%}
  </head>
  <body class="sidebar-open">
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""

# === LAYOUT ===
app.layout = html.Div(
    [
        # Router (для чтения query-параметров)
        dcc.Location(id="url", refresh=False),

        # Sidebar toggle
        html.Button(
            html.Img(
                src="/assets/icons/logo.svg",
                style={"width": "25px", "height": "25px"},
            ),
            id="sidebar-toggle",
            className="icon-btn",
            **{
                "data-tooltip": "Открыть боковую панель",
                "aria-label": "Открыть боковую панель",
                "aria-expanded": "false",
                "aria-controls": "left-sidebar",
            },
        ),

        # Sidebar
        html.Div(
            html.Nav(
                [
                    html.Div("CloseAI", className="title"),
                    html.A("Общий анализ", href="#", className="dashboard-link"),
                    html.A("Парсинг", href="http://main-service:8003#parsing", **{"data-target": "parsing"}),
                    html.A("Таксономия", href="http://main-service:8003#taxonomy", **{"data-target": "taxonomy"}),
                    html.A("Экспорт", href="http://main-service:8003#export", **{"data-target": "export"}),
                    html.A("Анализ", href="http://main-service:8003#analysis", **{"data-target": "analysis"}),
                    html.A("Продукты", href="#", id="products-link", **{"data-target": "products"}),
                ],
                className="vnav",
            ),
            id="left-sidebar",
            className="sidebar open",
            **{"aria-label": "Вертикальная навигация"},
        ),

        # Stores
        html.Div(id="current-dates-store", style={"display": "none"}),
        dcc.Store(id="selected-product-store"),

        # Background overlay
        html.Div(className="bg"),

        # === MAIN CONTENT ===
        html.Main(
            [
                # Title + last update
                html.Div(
                    [
                        html.H1("Общий анализ", className="title"),
                        html.Div(id="last-update", className="hint"),
                    ],
                    className="tabs",
                ),
                
                # Controls
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Период", style={"fontWeight": "600"}),
                                        dcc.Dropdown(
                                            id="time-range",
                                            options=[
                                                {"label": "День", "value": "day"},
                                                {"label": "Неделя", "value": "week"},
                                                {"label": "Месяц", "value": "month"},
                                                {"label": "Год", "value": "year"},
                                            ],
                                            value="month",
                                            clearable=False,
                                            style={"width": "200px"},
                                            className="dash-dropdown",
                                            searchable=False,
                                        ),
                                    ],
                                    className="timerange-div",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [html.Label("Начальная дата", style={"fontWeight": "600"}),
                                             dcc.DatePickerSingle(
                                                 id="start-date",
                                                 date=date(2024, 1, 1),
                                                 display_format='DD.MM.YYYY',
                                                 className="dash-datepicker",
                                             )],
                                            className="datepicker-div",
                                        ),
                                        html.Div(
                                            [html.Label("Конечная дата", style={"fontWeight": "600"}),
                                             dcc.DatePickerSingle(
                                                 id="end-date",
                                                 date=date(2025, 5, 31),
                                                 display_format='DD.MM.YYYY',
                                                 className="dash-datepicker",
                                             )],
                                            className="datepicker-div",
                                        ),
                                    ],
                                    className="datepickers-row",
                                ),
                            ],
                            className="controls-row",
                        ),
                    ],
                    className="tile controls-card",
                ),

                # KPI Row 1
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.Div("Кол-во отзывов", className="kpi-label"),
                                     html.Div(id="total-reviews", className="kpi-value")],
                                    className="kpi-wrapper kpi kpi--small",
                                ),
                                html.Div(
                                    [html.Div("Средний рейтинг", className="kpi-label"),
                                     html.Div(id="average-rating", className="kpi-value")],
                                    className="kpi-wrapper kpi kpi--small",
                                ),
                            ],
                            className="kpi-column",
                        ),
                        html.Div(
                            [html.Div("Процент тональности", className="kpi-label"),
                             html.Div(id="sentiment-breakdown", className="sentiment-breakdown")],
                            className="kpi-wrapper sentiment-card",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [html.Div("Кол-во дней", className="kpi-label"),
                                     html.Div(id="count-days", className="kpi-value")],
                                    className="kpi-wrapper kpi kpi--small",
                                ),
                                html.Div(
                                    [html.Div("Самый популярный продукт или услуга", className="kpi-label"),
                                     html.Div(id="popular-product", className="kpi-value")],
                                    className="kpi-wrapper kpi kpi--small",
                                ),
                            ],
                            className="kpi-column",
                        ),
                    ],
                    className="kpi-grid",
                ),

                # KPI Row 2
                html.Div(
                    [
                        html.Div(
                            [html.Div("Кол-во уникальных продуктов показаных", className="kpi-label"),
                             html.Div(id="count-base", className="kpi-value")],
                            className="kpi-wrapper kpi kpi--small",
                        ),
                        html.Div(
                            [html.Div("Кол-во уникальных продуктов скрытых", className="kpi-label"),
                             html.Div(id="count-full", className="kpi-value")],
                            className="kpi-wrapper kpi kpi--small",
                        ),
                    ],
                    className="kpi-grid",
                ),
                # Graphs
                html.Div(
                    [
                        html.Div(
                            dcc.Graph(
                                id="word_cloud", 
                                className="dash-graph", 
                                config={
                                    **CONFIG_FIG,
                                    "displaylogo": False,
                                    "watermark": False,
                                    "scrollZoom": False,
                                    "modeBarButtonsToRemove": list(set(CONFIG_FIG["modeBarButtonsToRemove"] + [
                                        "autoScale2d",
                                        "resetScale2d",
                                        "zoom2d",
                                        "pan2d",
                                    ]))
                                }
                            ),
                            className="tile graph-card"
                        ),
                        html.Div(dcc.Graph(id="reviews-per-day", className="dash-graph", config=CONFIG_FIG),
                                 className="tile graph-card"),
                        html.Div(dcc.Graph(id="mean-rating-per-day", className="dash-graph", config=CONFIG_FIG),
                                 className="tile graph-card"),
                        html.Div(dcc.Graph(
                            id="popular-products",
                            className="dash-graph",
                            config={**CONFIG_FIG, "displaylogo": False, "watermark": False, "scrollZoom": False},
                            responsive=False,
                            style={"height": 350, "overflowY": "scroll"},
                        ),
                        className="tile graph-card"),
                    ],
                    className="graph-section",
                ),

                # Auto-refresh
                # dcc.Interval(id="data-refresh", interval=INTERVAL, n_intervals=0),
            ],
            className="container",
        ),

        # Error status
        html.Div(id="status_erorrs", className="status_erorrs", style={"display": "none"}),
    ],
className="page",
)

# === Sync dates from URL ===
@app.callback(
    Output("start-date", "date"),
    Output("end-date", "date"),
    Input("url", "search"),
)
def sync_dates_from_query(search: str | None):
    if not search:
        return no_update, no_update

    query = parse_qs(search.lstrip("?"), keep_blank_values=False)
    start_param = (query.get("start_date") or [None])[0]
    end_param = (query.get("end_date") or [None])[0]

    def normalize(value: str | None):
        if not value:
            return no_update
        try:
            return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
        except (TypeError, ValueError):
            return no_update

    return normalize(start_param), normalize(end_param)


# === CALLBACKS ===
@app.callback(
    Output("total-reviews", "children"),
    Output("average-rating", "children"),
    Output("sentiment-breakdown", "children"),
    Output("count-days", "children"),
    Output("popular-product", "children"),
    Output("count-base", "children"),
    Output("count-full", "children"),
    Output("word_cloud", "figure"),
    Output("reviews-per-day", "figure"),
    Output("mean-rating-per-day", "figure"),
    Output("popular-products", "figure"),
    Output("last-update", "children"),
    Output("status_erorrs", "children"),
    Output("selected-product-store", "data"),

    # Input("data-refresh", "n_intervals"),
    Input("time-range", "value"),
    Input("start-date", "date"),
    Input("end-date", "date"),
    Input("url", "search"),
    Input("popular-products", "clickData"),
    State("selected-product-store", "data"),
)
def refresh_dashboard(time_range, start_date, end_date, url_search, click_data, selected_product_state):

    warning = ""
    triggered_id = ctx.triggered_id

    # === Parse selected products ===
    selected_products: list[str] = []
    if isinstance(selected_product_state, list):
        selected_products = [str(item) for item in selected_product_state if isinstance(item, str) and item.strip()]
    elif isinstance(selected_product_state, str) and selected_product_state.strip():
        selected_products = [selected_product_state]

    # === URL-driven filter (filter_product) overrides selection ===
    has_filter = False
    if isinstance(url_search, str) and url_search:
        try:
            query = parse_qs(url_search.lstrip("?"), keep_blank_values=False)
            filter_values = query.get("filter_product") or []
            if filter_values:
                selected_products = [v.strip().lower() for v in filter_values if v and v.strip()]
                selected_norms = selected_products[:]
                has_filter = True
            else:
                selected_norms = []
        except Exception:
            pass

    # === Deduplicate selection preserving order ===
    cleaned_products: list[str] = []
    selected_norms: list[str] = []
    seen_norms: set[str] = set()
    for item in selected_products:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        norm = cleaned.lower()
        if norm in seen_norms:
            continue
        seen_norms.add(norm)
        cleaned_products.append(cleaned)
        selected_norms.append(norm)
    selected_products = cleaned_products

    # === API Requests ===
    try:
        payload = {"date_from": start_date, "date_to": end_date}
        r1 = httpx.post(EXTRACT, json=payload, timeout=30.0)

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        if (end_date - start_date).days < 30:
            raise ValueError()

        payload["was_processed"] = r1.json()["was_processed"]
        payload["was_date_changed"] = triggered_id in ["start-date", "end-date"]
        httpx.post(ALL_PRODUCT_DESCRIPTION, json=payload, timeout=30.0)
        httpx.post(ALL_PRODUCT_SUMMARY, json=payload, timeout=30.0)
    except ValueError:
        return _return_empty(warning="Диапазон дат должен быть не меньше 1 месяца" + random.choice(UNICODE_SPACE),
                             selected_products=selected_products)

    except TypeError:
        return _return_empty(warning="Не удалось обработать даты" + random.choice(UNICODE_SPACE),
                             selected_products=selected_products)

    # === Data preparation ===
    try:
        df, df1 = load_reviews_dataframe()

        df_all = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
        df1_all = df1[(df1["date"] >= start_date) & (df1["date"] <= end_date)].copy()
        # df_all.dropna(subset=["date"], inplace=True)

        df_all["product_norm"] = df_all["product"].astype(str).str.strip().str.lower()
        df_all["product_display"] = df_all["product"].astype(str).str.strip()
        df_all["date_only"] = df_all["date"].dt.date

        metrics_all = calc_data(df_all, df1_all)

        selected_norms_set = set(selected_norms)
        if selected_norms_set:
            df_filtered = df_all[df_all["product_norm"].isin(selected_norms_set)].copy()
        else:
            df_filtered = df_all.copy()

        metrics = calc_data(df_filtered, df1_all)

        product_display_map: dict[str, str] = {}
        for norm, display in zip(df_all["product_norm"], df_all["product_display"]):
            if not norm or norm in product_display_map:
                continue
            product_display_map[norm] = display

        selected_display: list[str] = []
        for norm, original_label in zip(selected_norms, selected_products):
            selected_display.append(product_display_map.get(norm, original_label))
        selected_products = selected_display

        # KPIs
        total_reviews = str(int(metrics.get("total_reviews", 0)))
        count_days = (end_date - start_date).days
        mean_rating = metrics.get("mean_rating_all_data")
        average_rating = f"{float(mean_rating):.2f}" if pd.notna(mean_rating) else "-"

        popular_products_dict = metrics.get("popular_products", {})
        popular_product = str(list(popular_products_dict.keys())[-1]).capitalize() if popular_products_dict else ""

        sentiment_breakdown_children = _build_sentiment(metrics)

        overall_reviews_map: dict[str, float] = {}
        if not df_all.empty:
            overall_reviews_map = {
                date.isoformat(): float(value)
                for date, value in df_all.groupby("date_only").size().items()
            }

        overall_mean_map: dict[str, float] = {}
        if not df_all.empty and "grade_extracted" in df_all.columns:
            overall_mean_map = {
                date.isoformat(): float(value)
                for date, value in df_all.groupby("date_only")["grade_extracted"].mean().items()
                if pd.notna(value)
            }

        product_counts_map: dict[str, dict[str, float]] = {}
        product_mean_map: dict[str, dict[str, float]] = {}
        if not df_all.empty:
            counts_series = df_all.groupby(["product_norm", "date_only"]).size()
            for (norm, date_obj), value in counts_series.items():
                if not norm:
                    continue
                product_counts_map.setdefault(norm, {})[date_obj.isoformat()] = float(value)

            if "grade_extracted" in df_all.columns:
                mean_series = df_all.groupby(["product_norm", "date_only"])["grade_extracted"].mean()
                for (norm, date_obj), value in mean_series.items():
                    if not norm or pd.isna(value):
                        continue
                    product_mean_map.setdefault(norm, {})[date_obj.isoformat()] = float(value)

        display_lookup: dict[str, str] = {}
        for norm_key, display_value in product_display_map.items():
            if not isinstance(display_value, str):
                continue
            display_clean = display_value.strip()
            if not display_clean:
                continue
            display_lookup.setdefault(display_clean.lower(), norm_key)

        if selected_norms:
            active_series = list(zip(selected_norms, selected_products))
        else:
            popular_products_all = metrics_all.get("popular_products", {}) or {}
            ordered_labels = list(popular_products_all.keys())
            active_series = []
            seen_auto_norms: set[str] = set()
            for label in reversed(ordered_labels):
                label_str = str(label).strip()
                if not label_str:
                    continue
                norm_value = display_lookup.get(label_str.lower(), label_str.lower())
                if norm_value in seen_auto_norms:
                    continue
                if not product_counts_map.get(norm_value) and not product_mean_map.get(norm_value):
                    continue
                display_label = product_display_map.get(norm_value, label_str)
                active_series.append((norm_value, display_label))
                seen_auto_norms.add(norm_value)

        reviews_series_map: dict[str, dict[str, float]] = {}
        mean_series_map: dict[str, dict[str, float]] = {}

        color_map: dict[str, str] = {}
        used_colors: set[str] = set()
        palette_cycle = cycle(PRODUCT_SERIES_COLORS)
        palette_size = len(PRODUCT_SERIES_COLORS)

        for norm, label in active_series:
            counts_dict = product_counts_map.get(norm, {})
            mean_dict = product_mean_map.get(norm, {})
            if not counts_dict and not mean_dict:
                continue

            color = next(palette_cycle)
            attempts = 0
            while color in used_colors and attempts < palette_size:
                color = next(palette_cycle)
                attempts += 1
            used_colors.add(color)
            color_map[label] = color

            if counts_dict:
                reviews_series_map[label] = build_series_by_mode(counts_dict, mode=time_range, agg="sum")
            if mean_dict:
                mean_series_map[label] = build_series_by_mode(mean_dict, mode=time_range, agg="mean")

        color_map[OVERALL_SERIES_NAME] = OVERALL_SERIES_COLOR
        reviews_series_map[OVERALL_SERIES_NAME] = build_series_by_mode(overall_reviews_map, mode=time_range, agg="sum")
        mean_series_map[OVERALL_SERIES_NAME] = build_series_by_mode(overall_mean_map, mode=time_range, agg="mean")

        trend_color_map = {
            label: color
            for label, color in color_map.items()
            if label != OVERALL_SERIES_NAME
        }

        reviews_title_by_range = {
            "year": "Кол-во упоминаний по датам за год",
            "month": "Кол-во упоминаний по датам за месяц",
            "week": "Кол-во упоминаний по датам за неделю",
            "day": "Кол-во упоминаний по датам за день",
        }
        mean_title_by_range = {
            "year": "Средняя оценка по датам за год",
            "month": "Средняя оценка по датам за месяц",
            "week": "Средняя оценка по датам за неделю",
            "day": "Средняя оценка по датам за день",
        }

        reviews_per_day = _time_series_figure(
            reviews_series_map,
            reviews_title_by_range.get(time_range, "Кол-во упоминаний"),
            "Кол-во упоминаний",
            "reviews-per-day",
            "Кол-во упоминаний: %{y}<br>Дата: %{x}",
            color_map=color_map,
        )
        mean_rating_per_day = _time_series_figure(
            mean_series_map,
            mean_title_by_range.get(time_range, "Средняя оценка"),
            "Оценка",
            "mean-rating-per-day",
            "Средняя оценка: %{y}<br>Дата: %{x}",
            color_map=color_map,
        )

        popular_products_fig = _horizontal_bar(
            metrics_all.get("popular_products", {}),
            "ТОП продуктов",
            "Отзывы",
            "popular-products",
            selected_labels=selected_products,
            series_colors=trend_color_map,
        )

        wordcloud_fig = generate_cloud(df_all)

        base, full = get_base_full()
        last_update = f"Last update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    except Exception as e:
        return _return_empty(warning=str(e), selected_products=selected_products)

    # === Return ===
    status_node = html.Span(warning, **{"data-stamp": datetime.now().isoformat()}) if warning else "success"
    return (
        total_reviews,
        average_rating,
        sentiment_breakdown_children,
        count_days,
        popular_product,
        base,
        full,
        wordcloud_fig,
        reviews_per_day,
        mean_rating_per_day,
        popular_products_fig,
        last_update,
        status_node,
        selected_products,
    )
def _return_empty(warning: str, selected_products: list[str]):
    placeholder_text = ""
    empty_fig = _empty_figure("Данные не найдены")
    sentiment_placeholder = html.Div(placeholder_text, className="sentiment-empty")
    return (
        placeholder_text, placeholder_text, sentiment_placeholder, placeholder_text,
        placeholder_text, placeholder_text, placeholder_text, 
        empty_fig, empty_fig, empty_fig, empty_fig,
        placeholder_text, warning, selected_products,
    )


def _build_sentiment(metrics: dict) -> list | html.Div:
    sentiment_shares_raw = metrics.get("procent_sentiment", {})
    normalized_shares: dict[str, float] = {}
    for key, value in sentiment_shares_raw.items():
        try:
            if isinstance(value, str) and value.endswith("%"):
                value = float(value.rstrip("%"))
            normalized_shares[key.lower()] = float(value)
        except (TypeError, ValueError):
            continue

    if not normalized_shares:
        return html.Div("", className="sentiment-empty")

    positive = to_pct(pick_share(normalized_shares, "positive", "pos"))
    negative = to_pct(pick_share(normalized_shares, "negative", "neg"))
    neutral = to_pct(pick_share(normalized_shares, "neutral", "neu"))

    dict_shares = [
        {"title": "Negative", "procent": negative, "color": "#ef4444"},
        {"title": "Neutral", "procent": neutral, "color": "#6b7280"},
        {"title": "Positive", "procent": positive, "color": "#16a34a"},
    ]
    dict_normalized_shares = sorted(dict_shares, key=lambda x: x["procent"])
    return [_sentiment_item(item["title"], item["procent"], item["color"]) for item in dict_normalized_shares]


@app.callback(
    Output("products-link", "href"),
    Input("start-date", "date"),
    Input("end-date", "date"),
)
def update_products_link(start_date: str, end_date: str):
    base_url = MAIN_SERVICE_BASE_URL
    if start_date and end_date:
        return f"{base_url}/?start_date={start_date}&end_date={end_date}#products"
    return f"{base_url}/#products"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
