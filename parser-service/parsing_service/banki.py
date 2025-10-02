from __future__ import annotations

import asyncio
import csv
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

from playwright.async_api import APIRequestContext, Error, async_playwright

from .state import StateStore
from .utils import append_rows, ensure_csv, format_datetime, parse_date


MAIN_URL = "https://www.banki.ru/services/responses/bank/gazprombank/"
API_TMPL = "https://www.banki.ru/services/responses/list/ajax/?page={page}&is_countable=on&bank=gazprombank"

FIELDS = ["id", "d_id", "agentAnswerText", "date", "rating", "text", "title", "userName"]

@dataclass
class BankiFullResult:
    processed_pages: int
    added_reviews: int
    last_page: int
    has_more: bool


@dataclass
class BankiIncrementalResult:
    added_reviews: int
    reached_existing: bool


class BankiRuParser:
    def __init__(
        self,
        data_path: Path,
        state_store: StateStore,
        headless: bool = True,
    ) -> None:
        self._data_path = data_path
        self._state = state_store
        self._headless = headless
        ensure_csv(self._data_path, FIELDS)

    def full_parse(
        self,
        pages_per_run: Optional[int] = None,
        start_page: Optional[int] = None,
    ) -> BankiFullResult:
        return asyncio.run(self.full_parse_async(pages_per_run=pages_per_run, start_page=start_page))

    async def full_parse_async(
        self,
        pages_per_run: Optional[int] = None,
        start_page: Optional[int] = None,
    ) -> BankiFullResult:
        state_snapshot = self._state.load()
        progress = state_snapshot["banki"].get("full", {})
        page_num = start_page if start_page is not None else max(1, int(progress.get("next_page", 1)))
        has_more = bool(progress.get("has_more", True))

        existing_ids = self._load_existing_ids()
        latest_date, latest_ids = self._extract_latest_meta(state_snapshot)

        processed_pages = 0
        added_reviews = 0
        last_success_page = page_num
        has_more_pages = has_more

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self._headless)
            try:
                context = await browser.new_context(locale="ru-RU")
                try:
                    page = await context.new_page()
                    await page.goto(MAIN_URL, wait_until="domcontentloaded")
                    await self._jitter(1.0, 2.0)

                    while True:
                        if pages_per_run is not None and processed_pages >= pages_per_run:
                            break

                        data = await self._fetch_page(context.request, page_num)
                        if data is None:
                            break

                        items = data.get("data", []) or []
                        has_more_pages = bool(data.get("hasMorePages", False))
                        rows, latest_date, latest_ids = self._prepare_rows(
                            items, existing_ids, latest_date, latest_ids
                        )

                        if rows:
                            added_reviews += append_rows(self._data_path, FIELDS, rows)

                        processed_pages += 1
                        last_success_page = page_num
                        page_num += 1

                        self._state.update(
                            lambda s, hm=has_more_pages, next_page=page_num, ldt=latest_date, lids=list(latest_ids): self._update_state_full(
                                s, hm, next_page, ldt, lids
                            )
                        )

                        if not has_more_pages:
                            break

                        await self._jitter()
                finally:
                    await context.close()
            finally:
                await browser.close()

        return BankiFullResult(
            processed_pages=processed_pages,
            added_reviews=added_reviews,
            last_page=last_success_page,
            has_more=has_more_pages,
        )

    def incremental(self, max_pages: Optional[int] = None) -> BankiIncrementalResult:
        return asyncio.run(self.incremental_async(max_pages=max_pages))

    async def incremental_async(self, max_pages: Optional[int] = None) -> BankiIncrementalResult:
        state_snapshot = self._state.load()
        latest_date, latest_ids = self._extract_latest_meta(state_snapshot)
        if latest_date is None:
            full_result = await self.full_parse_async(pages_per_run=max_pages, start_page=1)
            return BankiIncrementalResult(
                added_reviews=full_result.added_reviews,
                reached_existing=not full_result.has_more,
            )

        existing_ids = self._load_existing_ids()
        added_reviews = 0
        reached_existing = False
        page_num = 1
        processed_pages = 0

        cutoff_date = latest_date
        cutoff_ids = set(latest_ids)

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self._headless)
            try:
                context = await browser.new_context(locale="ru-RU")
                try:
                    api_page = await context.new_page()
                    await api_page.goto(MAIN_URL, wait_until="domcontentloaded")
                    await self._jitter(1.0, 2.0)

                    while True:
                        if max_pages is not None and processed_pages >= max_pages:
                            break

                        data = await self._fetch_page(context.request, page_num)
                        if data is None:
                            break

                        items = data.get("data", []) or []
                        new_rows: List[dict] = []
                        for item in items:
                            rid = self._normalize_id(item.get("id"))
                            if rid is None:
                                continue

                            item_date = parse_date(item.get("dateCreate"))
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

                            row = self._build_row(item, rid)
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

                        if reached_existing or not data.get("hasMorePages", False):
                            break

                        page_num += 1
                        processed_pages += 1
                        await self._jitter()
                finally:
                    await context.close()
            finally:
                await browser.close()

        return BankiIncrementalResult(added_reviews=added_reviews, reached_existing=reached_existing)

    async def _fetch_page(
        self,
        ctx_request: APIRequestContext,
        page_num: int,
        attempts: int = 5,
    ) -> Optional[dict]:
        url = API_TMPL.format(page=page_num)
        headers = {
            "Referer": MAIN_URL,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
        }
        attempt = 0
        while attempt < attempts:
            attempt += 1
            try:
                response = await ctx_request.get(url, headers=headers, timeout=15000)
            except Error:
                response = None
            if response and 200 <= response.status < 300:
                try:
                    return await response.json()
                except Exception:
                    return None
            if response and response.status in (429, 500, 502, 503, 504):
                backoff = min(8, 0.8 * (2 ** (attempt - 1)))
                await asyncio.sleep(backoff + random.uniform(0, 0.7))
                continue
            break
        return None

    def _load_existing_ids(self) -> Set[int]:
        if not self._data_path.exists():
            return set()
        ids: Set[int] = set()
        with self._data_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rid = self._normalize_id(row.get("id"))
                if rid is not None:
                    ids.add(rid)
        return ids

    def _build_row(self, item: dict, rid: Optional[int]) -> dict:
        row = {
            "id": item.get("id"),
            "d_id": f"b{rid}" if rid is not None else None,
            "agentAnswerText": item.get("agentAnswerText"),
            "date": item.get("dateCreate"),
            "rating": item.get("grade"),
            "text": item.get("text"),
            "title": item.get("title"),
            "userName": item.get("userName"),
        }
        normalized_date = format_datetime(row.get("date"))
        if normalized_date:
            row["date"] = normalized_date
        return row

    def _prepare_rows(
        self,
        items: Iterable[dict],
        existing_ids: Set[int],
        latest_date: Optional[datetime],
        latest_ids: Set[int],
    ) -> Tuple[List[dict], Optional[datetime], Set[int]]:
        rows: List[dict] = []
        for item in items:
            rid = self._normalize_id(item.get("id"))
            if rid is None or rid in existing_ids:
                continue
            existing_ids.add(rid)
            row = self._build_row(item, rid)
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

    def _extract_latest_meta(self, state: dict) -> Tuple[Optional[datetime], Set[int]]:
        banki_state = state.get("banki", {})
        latest_date_str = banki_state.get("latest_date")
        latest_date = parse_date(latest_date_str) if latest_date_str else None
        raw_ids = banki_state.get("latest_date_ids") or []
        latest_ids: Set[int] = set()
        for raw in raw_ids:
            try:
                latest_ids.add(int(raw))
            except (TypeError, ValueError):
                continue
        return latest_date, latest_ids

    def _update_state_full(
        self,
        state: dict,
        has_more: bool,
        next_page: int,
        latest_date: Optional[datetime],
        latest_ids: List[int],
    ) -> None:
        banki_state = state.setdefault("banki", {})
        banki_state["full"] = {"has_more": has_more, "next_page": next_page}
        self._update_latest_meta(state, latest_date, latest_ids)

    def _update_latest_meta(
        self,
        state: dict,
        latest_date: Optional[datetime],
        latest_ids: Iterable[int],
    ) -> None:
        banki_state = state.setdefault("banki", {})
        if latest_date is not None:
            formatted = format_datetime(latest_date)
            if formatted:
                banki_state["latest_date"] = formatted
                normalized_ids = sorted({int(i) for i in latest_ids})
                banki_state["latest_date_ids"] = normalized_ids

    @staticmethod
    def _normalize_id(value) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    async def _jitter(a: float = 0.6, b: float = 1.6) -> None:
        await asyncio.sleep(random.uniform(a, b))


