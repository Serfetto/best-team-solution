import json
from typing import Any
import time
import plotly.utils
from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

from src.analysis.utils import _get_from_service, _post_to_service
from src.analysis.for_grathics import CONFIG_FIG, generate_cloud, load_reviews_dataframe, calc_data, build_series_by_mode, _time_series_figure, sentiment_pie_chart
from src.configs.config import settings
from src.models.database import db_dependency
from src.models.schemas import (
    AnalysisRequest,
    BatchAnalysisRequest,
    InsightsItem,
    PredictModel,
    PredictResponse,
    ProductAllDescriptionResponse,
    ProductDescriptionSummaryResponse,
    ProductSummaryRequest,
    ProductSummaryResponse,
    ProductDescriptionRequest,
    ProductDescriptionResponse,
    ProductAllDescriptionRequest,
    ProductAllDescriptionSummaryResponse,
)

router = APIRouter()

#json ответ оргам
@router.post("/predict/", tags=["analysis"])
async def predict(data_predict: BatchAnalysisRequest):
    payload = data_predict.model_dump(mode="json")
    raw_result = await _post_to_service(settings.ANALIZE_BATCH_URL_MODEL, payload)
    result = PredictResponse.model_validate(raw_result)
    return result

@router.post("/insights/product-summary/", tags=["analysis"])
async def product_summary(data_predict: ProductSummaryRequest):
    payload = data_predict.model_dump(mode="json")
    raw_result = await _post_to_service(settings.INSIGHTS_PRODUCT_SUMMARY_URL_MODEL, payload)
    result = ProductSummaryResponse.model_validate(raw_result)
    return result

@router.post("/insights/product-description/", tags=["analysis"])
async def product_description(data_predict: ProductDescriptionRequest):
    payload = data_predict.model_dump(mode="json")
    raw_result = await _post_to_service(settings.INSIGHTS_PRODUCTS_DESCRIPTION_URL_MODEL, payload)
    result = ProductDescriptionResponse.model_validate(raw_result)
    return result

@router.get("/insights/products/", tags=["analysis"])
async def all_product_description():
    for item in ["base", "full"]:
        with open(settings.PROCESSED_DATA_DIR / f"taxonomy_{item}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if item == "base":
            base = data
        else:
            full = data


    items = []
    for product_name, data in full.items():
        if product_name in base:
            item = ProductDescriptionSummaryResponse(
                product_name=product_name.capitalize(),
                description=data.get("description", None),
                examples=data.get("description_examples", None),
                strengths=data.get("strengths_examples", None),
                weaknesses=data.get("weaknesses_examples", None),
                strengths_summary=data.get("strengths_summary", None),
                weaknesses_summary=data.get("weaknesses_summary", None),
            )
            items.append(item)

    result = ProductAllDescriptionSummaryResponse(root=items)
    
    return result

@router.post("/insights/product-graphs/")
def product_graphs(payload: ProductDescriptionRequest):
    product_name = payload.product_name
    date_from = pd.to_datetime(payload.date_from)
    date_to = pd.to_datetime(payload.date_to)
    
    df = load_reviews_dataframe()
    df = df[(df["date"] >= date_from) & (df["date"] <= date_to)]
    if product_name:
        df = df[df["product"].str.lower() == product_name.lower()]

    metrics = calc_data(df)

    reviews_series = build_series_by_mode(metrics["count_reviews_per_day"], "month", agg="sum")
    mean_series = build_series_by_mode(metrics["mean_rating_per_day"], "month", agg="mean")

    fig1 = _time_series_figure(
        reviews_series,
        "Отзывы за выбранный период", "Кол-во отзывов",
        "reviews-per-day",
        "Отзывы: %{y}<br>Дата: %{x}"
    )

    fig2 = _time_series_figure(
        mean_series,
        "Средний рейтинг за выбранный период", "Рейтинг",
        "mean-rating-per-day",
        "Рейтинг: %{y}<br>Дата: %{x}"
    )

    fig_sentiment = sentiment_pie_chart(metrics["procent_sentiment"])

    fig_cloud = generate_cloud(df)

    return {
        "total_reviews": int(metrics["total_reviews"]),
        "average_rating": metrics["mean_rating_all_data"],
        "fig_reviews": json.loads(json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)),
        "fig_mean": json.loads(json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)),
        "fig_sentiment": json.loads(json.dumps(fig_sentiment, cls=plotly.utils.PlotlyJSONEncoder)),
        "fig_cloud": json.loads(json.dumps(fig_cloud, cls=plotly.utils.PlotlyJSONEncoder)) if fig_cloud is not None else fig_cloud,
        "config": CONFIG_FIG, 
    }

#фулл ответ для сервиса
@router.post("/", tags=["analysis"])
async def analyze(request: AnalysisRequest):
    payload = request.model_dump(mode="json")
    return await _post_to_service(settings.ANALIZE_URL_MODEL, payload)

@router.get("/report/", tags=["analysis"])
async def get_latest_report():
    return await _get_from_service(settings.REPORT_URL_MODEL)



