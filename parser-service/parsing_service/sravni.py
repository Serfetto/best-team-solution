from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests

from .state import StateStore
from .utils import append_rows, ensure_csv, format_datetime, parse_date

BASE_URL = "https://www.sravni.ru/proxy-reviews/reviews/"
COMMON_PARAMS: Dict[str, str] = {
    "FilterBy": "withRates",
    "LocationGarId": "",
    "NewIds": "true",
    "OrderBy": "byDate",
    "PageSize": "1000",
    "ReviewObjectId": "5bb4f768245bc22a520a6115",
    "ReviewObjectType": "banks",
    "SqueezesVectorIds": "",
    "Tag": "",
    "WithVotes": "true",
    "fingerPrint": "aaf2486069572ec691eb9a410d2c04b7",
}
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Referer": "https://www.sravni.ru/bank/gazprombank/otzyvy/",
}
FIELDS = ["id", "d_id", "date", "rating", "text", "title", "userName"]
PAGE_SIZE = int(COMMON_PARAMS["PageSize"])


@dataclass
class SravniFullResult:
    processed_pages: int
    added_reviews: int
    last_page: int
    has_more: bool


@dataclass
class SravniIncrementalResult:
    added_reviews: int
    reached_existing: bool


class SravniRuParser:
    def __init__(
        self,
        data_path: Path,
        state_store: StateStore,
    ) -> None:
        self._data_path = data_path
        self._state = state_store
        ensure_csv(self._data_path, FIELDS)

    def full_parse(
        self,
        pages_per_run: Optional[int] = None,
        start_page: Optional[int] = None,
    ) -> SravniFullResult:
        state_snapshot = self._state.load()
        progress = state_snapshot["sravni"].get("full", {})
        page_index = start_page if start_page is not None else max(0, int(progress.get("next_page", 0)))

        existing_ids = self._load_existing_ids()
        latest_date, latest_ids = self._extract_latest_meta(state_snapshot)

        processed_pages = 0
        added_reviews = 0
        has_more = True
        last_page = page_index

        session = self._build_session()


        while True:
            if pages_per_run is not None and processed_pages >= pages_per_run:
                break

            items = self._fetch_page(session, page_index)
            if not items:
                has_more = False
                self._state.update(
                    lambda s, next_page=page_index, ldt=latest_date, lids=list(latest_ids): self._update_state_full(
                        s, has_more, next_page, ldt, lids
                    )
                )
                break

            rows, latest_date, latest_ids = self._prepare_rows(items, existing_ids, latest_date, latest_ids)

            if rows:
                added_reviews += append_rows(self._data_path, FIELDS, rows)

            processed_pages += 1
            last_page = page_index
            page_index += 1
            has_more = len(items) == PAGE_SIZE

            self._state.update(
                lambda s, hm=has_more, next_page=page_index, ldt=latest_date, lids=list(latest_ids): self._update_state_full(
                    s, hm, next_page, ldt, lids
                )
            )

            if not has_more:
                break

            self._sleep()

        return SravniFullResult(
            processed_pages=processed_pages,
            added_reviews=added_reviews,
            last_page=last_page,
            has_more=has_more,
        )

    def incremental(self, max_pages: Optional[int] = None) -> SravniIncrementalResult:
        state_snapshot = self._state.load()
        latest_date, latest_ids = self._extract_latest_meta(state_snapshot)
        if latest_date is None:
            full_result = self.full_parse(pages_per_run=max_pages, start_page=0)
            return SravniIncrementalResult(added_reviews=full_result.added_reviews, reached_existing=not full_result.has_more)

        existing_ids = self._load_existing_ids()
        added_reviews = 0
        reached_existing = False
        page_index = 0
        processed_pages = 0
        session = self._build_session()
        cutoff_date = latest_date
        cutoff_ids = set(latest_ids)

        while True:
            if max_pages is not None and processed_pages >= max_pages:
                break

            items = self._fetch_page(session, page_index)
            if not items:
                break

            new_rows: List[dict] = []
            for item in items:
                rid = item.get("id")
                if not rid:
                    continue
                item_date = parse_date(item.get("date"))
                if item_date is None:
                    continue
                if cutoff_date is not None:
                    if item_date < cutoff_date:
                        reached_existing = True
                        break
                    if item_date == cutoff_date and rid in cutoff_ids:
                        reached_existing = True
                        break
                if rid in existing_ids:
                    continue
                row = self._to_row(item)
                new_rows.append(row)
                existing_ids.add(rid)
                added_reviews += 1
                if item_date > latest_date:
                    latest_date = item_date
                    latest_ids = {rid}
                elif item_date == latest_date:
                    latest_ids.add(rid)

            if new_rows:
                append_rows(self._data_path, FIELDS, new_rows)
                self._state.update(
                    lambda s, ldt=latest_date, lids=list(latest_ids): self._update_latest_meta(s, ldt, lids)
                )

            if reached_existing or len(items) < PAGE_SIZE:
                break

            page_index += 1
            processed_pages += 1
            self._sleep()

        return SravniIncrementalResult(added_reviews=added_reviews, reached_existing=reached_existing)

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(HEADERS)
        return session

    def _fetch_page(self, session: requests.Session, page_index: int) -> List[dict]:
        params = COMMON_PARAMS.copy()
        params["PageIndex"] = str(page_index)

        backoff = 1.0
        for attempt in range(4):
            resp = session.get(BASE_URL, params=params, timeout=30)
            if resp.status_code == 200:
                payload = resp.json()
                return payload.get("items", []) or []
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff + random.uniform(0, 0.5))
                backoff = min(backoff * 2, 8.0)
                continue
            break
        return []

    def _load_existing_ids(self) -> Set[str]:
        if not self._data_path.exists():
            return set()
        ids: Set[str] = set()
        with self._data_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rid = row.get("id")
                if rid:
                    ids.add(str(rid))
        return ids

    def _prepare_rows(
        self,
        items: Iterable[dict],
        existing_ids: Set[str],
        latest_date: Optional[datetime],
        latest_ids: Set[str],
    ) -> Tuple[List[dict], Optional[datetime], Set[str]]:
        rows: List[dict] = []
        for item in items:
            rid = item.get("id")
            if not rid or rid in existing_ids:
                continue
            existing_ids.add(rid)
            row = self._to_row(item)
            rows.append(row)
            item_date = parse_date(row.get("date"))
            if item_date is None:
                continue
            if latest_date is None or item_date > latest_date:
                latest_date = item_date
                latest_ids = {rid}
            elif item_date == latest_date:
                latest_ids.add(rid)
        return rows, latest_date, latest_ids

    def _to_row(self, item: dict) -> dict:
        first = (item.get("authorName") or "").strip()
        last = (item.get("authorLastName") or "").strip()
        name = (first + (" " + last if last else "")).strip() or None
        review_id = item.get("id")
        raw_date = item.get("date")
        normalized_date = format_datetime(raw_date)
        return {
            "id": review_id,
            "d_id": f"s{review_id}" if review_id is not None else None,
            "date": normalized_date or raw_date,
            "rating": item.get("rating"),
            "text": item.get("text"),
            "title": item.get("title"),
            "userName": name,
        }

    def _extract_latest_meta(self, state: dict) -> Tuple[Optional[datetime], Set[str]]:
        sravni_state = state.get("sravni", {})
        latest_date_str = sravni_state.get("latest_date")
        latest_date = parse_date(latest_date_str) if latest_date_str else None
        latest_ids = set(str(x) for x in (sravni_state.get("latest_date_ids") or []) if x)
        return latest_date, latest_ids

    def _update_state_full(
        self,
        state: dict,
        has_more: bool,
        next_page: int,
        latest_date: Optional[datetime],
        latest_ids: List[str],
    ) -> None:
        sravni_state = state.setdefault("sravni", {})
        sravni_state["full"] = {"has_more": has_more, "next_page": next_page}
        self._update_latest_meta(state, latest_date, latest_ids)

    def _update_latest_meta(
        self,
        state: dict,
        latest_date: Optional[datetime],
        latest_ids: Iterable[str],
    ) -> None:
        sravni_state = state.setdefault("sravni", {})
        if latest_date is not None:
            formatted = format_datetime(latest_date)
            if formatted:
                sravni_state["latest_date"] = formatted
                normalized = sorted({str(x) for x in latest_ids if x})
                sravni_state["latest_date_ids"] = normalized

    @staticmethod
    def _sleep(a: float = 0.6, b: float = 1.6) -> None:
        time.sleep(random.uniform(a, b))

