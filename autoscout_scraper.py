#!/usr/bin/env python3
"""
Scrape listing data from AutoScout24 (.de / .ch) via the embedded Next.js payload.

Important:
  - AutoScout24 Terms of Use and robots.txt restrict automated extraction. Verify you have
    the right to use this data before large-scale scraping. This script is intended for small
    research exports and respecting rate limits.
  - There is no public AutoScout24 *read API* for search/listings (the official Listing Creation
    API is for dealers publishing inventory). Paid keys apply only if you route traffic through
    a third-party scraping/proxy vendor or automotive data reseller.
  - The site uses bot protection (Akamai). Many requests / datacenter IPs can trigger 403. For
    .ch listings, geo / residential IPs often work better when plain requests fail.

No API key is required for the approach implemented here beyond what your HTTP stack needs.

Coverage: this reads the structured JSON packaged in each HTML page (__NEXT_DATA__). It captures
many listing and vehicle attributes (specs, equipment groups, seller type, prices, etc.) but not
literally everything on AutoScout24 (full image URL lists at max resolution,
leasing/finance calculators, arbitrary UI-only widgets). Extend the parsers if you need more keys
from listingDetails — they are nested in JSON.

Germany inventory on www.autoscout24.de defaults to `--language en` (`culture=en-GB`): enumerations and
equipment names come back in English. CSV columns are fixed to `SCRAPE_OUTPUT_COLUMNS`; use
`--concurrency` for parallel listing-page and detail-page HTTP (raises 429/403 risk if set too high).
Use `--checkpoint-every` (default 2000): while collecting URLs (--catalog-all / search pages), snippet rows
(no doors/seats) go to `<out_stem>_catalog_snippet.csv` beside `--out`; during listing-detail scraping,
merged rows (including doors/seats) are written to `--out` every time that many detail rows finish.
Ctrl+C saves progress. Saves use the relative `--out`; the script resolves it to an absolute path and prints cwd on startup.
Shell pitfall: nothing may follow `\` except a newline — if there are spaces after `\`, `--out`
can be dropped and defaults to autoscout_dataset.csv.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import re
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse, urlunsplit

import pandas as pd
import requests
from requests import RequestException
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
]

SCRAPE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "listing_url",
    "image_url",
    "make_model",
    "price",
    "km",
    "Fuel",
    "transmission_card",
    "first_registration_iso",
    "power_hp_raw",
    "body_type",
    "vehicle_type_card",
    "doors",
    "seats",
    "co2_g_per_km",
    "cons_comb",
    "seller_type_detail",
    "city_detail",
)


@dataclass(frozen=True)
class SiteProfile:
    key: str
    base_netloc: str
    list_prefix: str
    cy_param: str
    query_suffix: str = ""
    # e.g. "en-GB" — appended as &culture=… so JSON uses English labels on .de/.ch.
    culture: str | None = None

    @property
    def listings_base_url(self) -> str:
        base = (
            f"https://{self.base_netloc}{self.list_prefix}"
            f"?atype=C&cy={self.cy_param}&sort=standard&ustate=N%2CU"
        )
        if self.query_suffix:
            q = self.query_suffix.lstrip("&")
            base = f"{base}&{q}"
        if self.culture:
            base = f"{base}&culture={self.culture}"
        return base

    @property
    def region_iso(self) -> str:
        """ISO-style region for the marketplace (column scrape_region)."""
        return {"D": "DE", "CH": "CH", "A": "AT", "I": "IT", "F": "FR", "NL": "NL"}.get(
            self.cy_param, self.cy_param
        )


PROFILES = {
    "de": SiteProfile(key="de", base_netloc="www.autoscout24.de", list_prefix="/lst", cy_param="D"),
    # Swiss marketplace is often crawled via the /de/ locale path.
    "ch": SiteProfile(
        key="ch",
        base_netloc="www.autoscout24.ch",
        list_prefix="/de/lst",
        cy_param="CH",
    ),
}


def create_session(proxy: str | None, *, accept_language: str) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=1.2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": accept_language,
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    )
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
    return session


class ThreadLocalSessionFactory:
    """requests.Session is not thread-safe; use one session per worker thread."""

    __slots__ = ("_accept_language", "_local", "_proxy")

    def __init__(self, proxy: str | None, *, accept_language: str) -> None:
        self._proxy = proxy
        self._accept_language = accept_language
        self._local = threading.local()

    def __call__(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = create_session(self._proxy, accept_language=self._accept_language)
            self._local.session = s
        return s


def extract_next_data(html: str) -> dict[str, Any] | None:
    """Parse embedded Next.js JSON; BeautifulSoup can miss/trim rare script layouts — regex fallback."""
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", id="__NEXT_DATA__")
    raw: str | None = None
    if tag:
        if tag.string:
            raw = str(tag.string)
        elif tag.contents:
            raw = "".join(
                c if isinstance(c, str) else (c.string or "")
                for c in tag.contents
            )
    if not raw:
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
        if not m:
            return None
        raw = m.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def polite_sleep(seconds_min: float, seconds_max: float) -> None:
    time.sleep(random.uniform(seconds_min, seconds_max))


def absolutize(profile: SiteProfile, path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    return urlunsplit(("https", profile.base_netloc, path_or_url, "", ""))


def normalize_listing_path(profile: SiteProfile, path: str) -> str:
    """English locale search uses /offers/; German detail URLs use /angebote/ on the .de host."""
    if profile.base_netloc == "www.autoscout24.de" and path.startswith("/offers/"):
        return "/angebote/" + path.removeprefix("/offers/")
    return path


def with_culture_query(url: str, culture: str | None) -> str:
    if not culture:
        return url
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "culture" in qs:
        return url
    qs["culture"] = [culture]
    new_query = urlencode({k: v[-1] for k, v in qs.items()}, doseq=True)
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)
    )


def fmt_maybe_dict(value: Any) -> Any:
    if isinstance(value, dict):
        fmt = value.get("formatted")
        if fmt is not None:
            return fmt
        raw = value.get("raw")
        if raw is not None:
            return raw
        return None
    return value


def snippet_price_euro_int_maybe(price_value: Any) -> int | None:
    """Best-effort EUR integer from search-snippet `price` strings (e.g. '€ 12.345', '€ 1.234,56')."""
    if price_value is None or (isinstance(price_value, float) and math.isnan(price_value)):
        return None
    s = str(price_value).strip()
    if not s:
        return None
    sl = s.lower()
    if "request" in sl or "auction" in sl or sl in {"-", "n/a"}:
        return None
    s_only = re.sub(r"[^\d.,]", "", s)
    if not s_only:
        return None
    if "," in s_only and "." in s_only:
        if s_only.rfind(",") > s_only.rfind("."):
            main, frac = s_only.rsplit(",", 1)
            main = main.replace(".", "")
            try:
                return int(float(main + "." + frac))
            except ValueError:
                return None
        main, frac = s_only.rsplit(".", 1)
        main = main.replace(",", "")
        try:
            return int(float(main + "." + frac))
        except ValueError:
            return None
    if "," in s_only:
        parts = s_only.split(",")
        if len(parts[-1]) <= 2 and len(parts) > 1:
            main = ",".join(parts[:-1]).replace(",", "")
            try:
                return int(float(main + "." + parts[-1]))
            except ValueError:
                return None
        try:
            return int("".join(parts))
        except ValueError:
            return None
    if "." in s_only:
        parts = s_only.split(".")
        if len(parts[-1]) <= 2 and len(parts) > 1:
            main = ".".join(parts[:-1]).replace(".", "")
            try:
                return int(float(main + "." + parts[-1]))
            except ValueError:
                return None
        try:
            return int("".join(parts))
        except ValueError:
            return None
    try:
        return int(s_only)
    except ValueError:
        return None


def max_snippet_price_euro_from_card_rows(card_rows: dict[str, dict[str, Any]]) -> int | None:
    best: int | None = None
    for row in card_rows.values():
        n = snippet_price_euro_int_maybe(row.get("price"))
        if n is not None and (best is None or n > best):
            best = n
    return best


def _door_seat_scalar(value: Any) -> int | float | None:
    """Normalize doors/seats from int, GraphQL-ish dict ({raw, formatted}), or digit string."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        fv = float(value)
        return int(fv) if fv.is_integer() else fv
    if isinstance(value, dict):
        raw = value.get("raw")
        if isinstance(raw, (int, float)) and not (isinstance(raw, float) and math.isnan(raw)):
            fv = float(raw)
            return int(fv) if fv.is_integer() else fv
        formatted = value.get("formatted")
        if isinstance(formatted, str):
            m = re.search(r"(\d+)", formatted.replace(",", ""))
            if m:
                return int(m.group(1))
        return None
    if isinstance(value, str):
        m = re.search(r"(\d+)", value.replace(",", ""))
        return int(m.group(1)) if m else None
    return None


def _first_door_seat_scalar(*candidates: Any) -> int | float | None:
    for c in candidates:
        s = _door_seat_scalar(c)
        if s is not None:
            return s
    return None


def doors_seats_from_listing(ld: dict[str, Any]) -> tuple[int | float | None, int | float | None]:
    """
    Prefer vehicle.numberOfDoors / numberOfSeats; some payloads only set rawData.interior &
    classified_* tracking fields.
    """
    v = ld.get("vehicle") if isinstance(ld.get("vehicle"), dict) else {}
    tp = ld.get("trackingParams") if isinstance(ld.get("trackingParams"), dict) else {}
    rd = v.get("rawData") if isinstance(v.get("rawData"), dict) else {}
    interior = v.get("interior") if isinstance(v.get("interior"), dict) else {}
    interior_rd = (
        rd.get("interior") if isinstance(rd.get("interior"), dict) else {}
    )

    doors = _first_door_seat_scalar(
        v.get("numberOfDoors"),
        rd.get("numberOfDoors"),
        tp.get("classified_doors"),
    )
    seats = _first_door_seat_scalar(
        v.get("numberOfSeats"),
        interior.get("numberOfSeats"),
        interior_rd.get("numberOfSeats"),
        tp.get("classified_seats"),
    )
    return doors, seats


def doors_seats_regex_from_next_data_html(html: str) -> tuple[int | float | None, int | float | None]:
    """Last-resort parse when structured listingDetails is incomplete but __NEXT_DATA__ is present."""
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return None, None
    chunk = m.group(1)
    md = re.search(r'"numberOfDoors"\s*:\s*(\d+)', chunk)
    ms = re.search(r'"numberOfSeats"\s*:\s*(\d+)', chunk)
    d = int(md.group(1)) if md else None
    s = int(ms.group(1)) if ms else None
    return d, s


def first_image_url(images: Any) -> str | None:
    """First picture URL from listing `images` (strings or dicts with url-like keys)."""
    seq = images if isinstance(images, list) else []
    for x in seq:
        u: str | None = None
        if isinstance(x, str):
            s = x.strip()
            if s:
                u = s
        elif isinstance(x, dict):
            for key in ("url", "uri", "src", "path", "defaultUrl", "fullUrl"):
                v = x.get(key)
                if isinstance(v, str) and v.strip():
                    u = v.strip()
                    break
            if u is None and isinstance(x.get("sizes"), dict):
                vals = [vv for vv in x["sizes"].values() if isinstance(vv, str) and vv.strip()]
                if vals:
                    u = vals[-1]
        if u:
            return u
    return None


AS24_MAX_SEARCH_PAGES = 200
# Max listing-detail futures submitted at once (avoids ~600k Task objects OOM / IDE freeze).
DETAIL_SUBMIT_CHUNK = 2000
# Used when a single EUR price still has 200+ search pages (more listings than one query can return).
CATALOG_KM_UPPER_DEFAULT = 2_500_000
CATALOG_FREG_YEAR_MIN = 1985


def _freg_year_hi() -> int:
    return max(2030, datetime.datetime.now().year + 1)


def search_list_url(
    profile: SiteProfile,
    page: int,
    *,
    price_lo: int | None = None,
    price_hi: int | None = None,
    km_lo: int | None = None,
    km_hi: int | None = None,
    freg_from: int | None = None,
    freg_to: int | None = None,
) -> str:
    u = profile.listings_base_url + f"&page={page}"
    if price_lo is not None and price_hi is not None:
        u += f"&pricefrom={price_lo}&priceto={price_hi}"
    if km_lo is not None and km_hi is not None:
        u += f"&kmfrom={km_lo}&kmto={km_hi}"
    if freg_from is not None and freg_to is not None:
        u += f"&fregfrom={freg_from}&fregto={freg_to}"
    return u


def _fetch_search_page_listings(
    session_factory: Callable[[], requests.Session],
    profile: SiteProfile,
    page: int,
    delay: tuple[float, float],
    *,
    price_lo: int | None = None,
    price_hi: int | None = None,
    km_lo: int | None = None,
    km_hi: int | None = None,
    freg_from: int | None = None,
    freg_to: int | None = None,
    referer_profile: SiteProfile | None = None,
) -> list[dict[str, Any]]:
    polite_sleep(*delay)
    sess = session_factory()
    url = search_list_url(
        profile,
        page,
        price_lo=price_lo,
        price_hi=price_hi,
        km_lo=km_lo,
        km_hi=km_hi,
        freg_from=freg_from,
        freg_to=freg_to,
    )
    headers: dict[str, str] = {}
    if page > 1:
        if price_lo is not None and price_hi is not None:
            headers["Referer"] = search_list_url(
                profile,
                page - 1,
                price_lo=price_lo,
                price_hi=price_hi,
                km_lo=km_lo,
                km_hi=km_hi,
                freg_from=freg_from,
                freg_to=freg_to,
            )
        else:
            hp = referer_profile or profile
            headers["Referer"] = hp.listings_base_url + f"&page={page - 1}"
    rr = sess.get(url, headers=headers, timeout=35)
    if rr.status_code == 403:
        raise RuntimeError("HTTP 403 while fetching a search results page.")
    rr.raise_for_status()
    pblob = extract_next_data(rr.text)
    if not pblob:
        return []
    return pblob["props"]["pageProps"].get("listings") or []


def first_registration_from_search_snippet(
    vehicle: dict[str, Any], tracking: dict[str, Any], vehicle_details: Any
) -> str | None:
    """ISO date YYYY-MM-01 from detail page field, tracking (MM-YYYY), or vehicleDetails calendar row."""
    raw = vehicle.get("firstRegistrationDateRaw")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()[:10] if len(raw.strip()) >= 10 else raw.strip()
    fr = tracking.get("firstRegistration")
    if isinstance(fr, str) and fr.strip():
        m = re.match(r"^(\d{1,2})-(\d{4})$", fr.strip())
        if m:
            mm, yyyy = int(m.group(1)), int(m.group(2))
            return f"{yyyy:04d}-{mm:02d}-01"
    if isinstance(vehicle_details, list):
        for row in vehicle_details:
            if not isinstance(row, dict):
                continue
            if row.get("iconName") != "calendar":
                continue
            data = (row.get("data") or "").strip()
            m2 = re.match(r"^(\d{1,2})/(\d{4})$", data)
            if m2:
                mm2, yyyy2 = int(m2.group(1)), int(m2.group(2))
                return f"{yyyy2:04d}-{mm2:02d}-01"
    return None


def hp_raw_from_vehicle_details(vehicle_details: Any) -> int | float | None:
    """Parse '103 kW (140 hp)' from search card vehicleDetails."""
    if not isinstance(vehicle_details, list):
        return None
    for row in vehicle_details:
        if not isinstance(row, dict):
            continue
        if row.get("iconName") != "speedometer" and (
            row.get("ariaLabel") or ""
        ).lower() != "power":
            continue
        data = str(row.get("data") or "")
        m = re.search(r"\((\d+)\s*hp\)", data, re.I) or re.search(r"(\d+)\s*hp\b", data, re.I)
        if m:
            v = float(m.group(1))
            return int(v) if v.is_integer() else v
    return None


def cons_and_co2_from_wltp(wltp_values: Any) -> tuple[str | None, str | None]:
    """wltpValues like ['5.2 l/100 km (comb.)', '135 g/km (comb.)'] on search cards."""
    if not isinstance(wltp_values, list):
        return None, None
    cons = str(wltp_values[0]).strip() if len(wltp_values) >= 1 and wltp_values[0] else None
    co2 = str(wltp_values[1]).strip() if len(wltp_values) >= 2 and wltp_values[1] else None
    return cons, co2


def add_card_row_from_listing_item(
    rows: dict[str, dict[str, Any]],
    profile: SiteProfile,
    item: dict[str, Any],
) -> None:
    rel = item.get("url")
    if not rel:
        return
    rel_norm = normalize_listing_path(profile, rel)
    listing_url = with_culture_query(absolutize(profile, rel_norm), profile.culture)
    vehicle = item.get("vehicle") or {}
    v_raw = vehicle.get("rawData") if isinstance(vehicle.get("rawData"), dict) else {}
    v_int = vehicle.get("interior") if isinstance(vehicle.get("interior"), dict) else {}
    v_int_rd = (
        v_raw.get("interior") if isinstance(v_raw.get("interior"), dict) else {}
    )
    loc = item.get("location") or {}
    seller = item.get("seller") or {}
    price = item.get("price") or {}
    tracking = item.get("tracking") or {}
    vd = item.get("vehicleDetails")
    w_cons, w_co2 = cons_and_co2_from_wltp(item.get("wltpValues"))

    vco2 = vehicle.get("co2emissionInGramPerKmWithFallback")
    if isinstance(vco2, dict):
        co2_snippet = fmt_maybe_dict(vco2)
    elif vco2 is not None:
        co2_snippet = vco2
    else:
        co2_snippet = w_co2

    vcons = vehicle.get("fuelConsumptionCombined")
    if vcons is not None:
        cons_snippet = fmt_maybe_dict(vcons)
    else:
        cons_snippet = w_cons

    rows[listing_url] = {
        "listing_url": listing_url,
        "image_url": first_image_url(item.get("images")),
        "make_model": (
            f"{vehicle.get('make', '')} {vehicle.get('model', '')} {vehicle.get('modelVersionInput', '')}"
        ).strip(),
        "price": price.get("priceFormatted") or fmt_maybe_dict(price.get("price")),
        "km": fmt_maybe_dict(vehicle.get("mileageInKm")),
        "Fuel": fmt_maybe_dict(vehicle.get("fuel")),
        "transmission_card": fmt_maybe_dict(
            vehicle.get("transmission") or vehicle.get("transmissionType")
        ),
        "first_registration_iso": first_registration_from_search_snippet(vehicle, tracking, vd),
        "power_hp_raw": vehicle.get("rawPowerInHp") if vehicle.get("rawPowerInHp") is not None else hp_raw_from_vehicle_details(vd),
        "body_type": fmt_maybe_dict(vehicle.get("bodyType")),
        "vehicle_type_card": vehicle.get("type"),
        "doors": _first_door_seat_scalar(
            vehicle.get("numberOfDoors"), v_raw.get("numberOfDoors")
        ),
        "seats": _first_door_seat_scalar(
            vehicle.get("numberOfSeats"),
            v_int.get("numberOfSeats"),
            v_int_rd.get("numberOfSeats"),
        ),
        "co2_g_per_km": co2_snippet,
        "cons_comb": cons_snippet,
        "seller_type_detail": seller.get("type"),
        "city_detail": loc.get("city"),
    }


def catalog_partition_collect_rows(
    session: requests.Session,
    profile: SiteProfile,
    session_factory: Callable[[], requests.Session],
    *,
    rows: dict[str, dict[str, Any]],
    price_min: int,
    price_max: int,
    delay: tuple[float, float],
    depth: int,
    max_depth: int,
    concurrency: int,
    after_bulk_ingest: Callable[[dict[str, dict[str, Any]]], None] | None = None,
    km_lo: int | None = None,
    km_hi: int | None = None,
    freg_from: int | None = None,
    freg_to: int | None = None,
) -> None:
    """Recurse price (then mileage, then first-registration year) until each slice is < 200 search pages."""
    if depth > max_depth:
        raise RuntimeError(
            f"Partition depth exceeded ({max_depth}) for EUR {price_min}–{price_max}. "
            "Increase --partition-max-depth or narrow with --query-suffix."
        )

    polite_sleep(*delay)
    url = search_list_url(
        profile,
        1,
        price_lo=price_min,
        price_hi=price_max,
        km_lo=km_lo,
        km_hi=km_hi,
        freg_from=freg_from,
        freg_to=freg_to,
    )
    r = session.get(url, timeout=35)
    if r.status_code == 403:
        raise RuntimeError(
            "HTTP 403 during catalog partition (blocked). Use slower delays or a residential proxy."
        )
    r.raise_for_status()
    blob = extract_next_data(r.text)
    if not blob:
        return
    page_props = blob["props"]["pageProps"]
    nr = int(page_props.get("numberOfResults") or 0)
    np_pages = int(page_props.get("numberOfPages") or 0)

    if nr == 0 or np_pages == 0:
        return

    if np_pages >= AS24_MAX_SEARCH_PAGES and price_min < price_max:
        mid = (price_min + price_max) // 2
        print(
            f"[catalog] EUR {price_min}–{price_max}: {nr} hits, pages={np_pages} → split "
            f"→ {price_min}–{mid} | {mid + 1}–{price_max}",
            flush=True,
        )
        catalog_partition_collect_rows(
            session,
            profile,
            session_factory,
            rows=rows,
            price_min=price_min,
            price_max=mid,
            delay=delay,
            depth=depth + 1,
            max_depth=max_depth,
            concurrency=concurrency,
            after_bulk_ingest=after_bulk_ingest,
            km_lo=km_lo,
            km_hi=km_hi,
            freg_from=freg_from,
            freg_to=freg_to,
        )
        catalog_partition_collect_rows(
            session,
            profile,
            session_factory,
            rows=rows,
            price_min=mid + 1,
            price_max=price_max,
            delay=delay,
            depth=depth + 1,
            max_depth=max_depth,
            concurrency=concurrency,
            after_bulk_ingest=after_bulk_ingest,
            km_lo=km_lo,
            km_hi=km_hi,
            freg_from=freg_from,
            freg_to=freg_to,
        )
        return

    if np_pages >= AS24_MAX_SEARCH_PAGES and price_min >= price_max:
        if km_lo is None:
            mid = CATALOG_KM_UPPER_DEFAULT // 2
            print(
                f"[catalog] EUR {price_min} single-price: {nr} hits, {np_pages} pages → "
                f"split mileage 0–{mid} km | {mid + 1}–{CATALOG_KM_UPPER_DEFAULT} km",
                flush=True,
            )
            catalog_partition_collect_rows(
                session,
                profile,
                session_factory,
                rows=rows,
                price_min=price_min,
                price_max=price_max,
                delay=delay,
                depth=depth + 1,
                max_depth=max_depth,
                concurrency=concurrency,
                after_bulk_ingest=after_bulk_ingest,
                km_lo=0,
                km_hi=mid,
                freg_from=freg_from,
                freg_to=freg_to,
            )
            catalog_partition_collect_rows(
                session,
                profile,
                session_factory,
                rows=rows,
                price_min=price_min,
                price_max=price_max,
                delay=delay,
                depth=depth + 1,
                max_depth=max_depth,
                concurrency=concurrency,
                after_bulk_ingest=after_bulk_ingest,
                km_lo=mid + 1,
                km_hi=CATALOG_KM_UPPER_DEFAULT,
                freg_from=freg_from,
                freg_to=freg_to,
            )
            return
        if km_lo < km_hi:
            mid_k = (km_lo + km_hi) // 2
            print(
                f"[catalog] EUR {price_min} km {km_lo}–{km_hi}: {nr} hits, pages={np_pages} → split "
                f"→ km {km_lo}–{mid_k} | {mid_k + 1}–{km_hi}",
                flush=True,
            )
            catalog_partition_collect_rows(
                session,
                profile,
                session_factory,
                rows=rows,
                price_min=price_min,
                price_max=price_max,
                delay=delay,
                depth=depth + 1,
                max_depth=max_depth,
                concurrency=concurrency,
                after_bulk_ingest=after_bulk_ingest,
                km_lo=km_lo,
                km_hi=mid_k,
                freg_from=freg_from,
                freg_to=freg_to,
            )
            catalog_partition_collect_rows(
                session,
                profile,
                session_factory,
                rows=rows,
                price_min=price_min,
                price_max=price_max,
                delay=delay,
                depth=depth + 1,
                max_depth=max_depth,
                concurrency=concurrency,
                after_bulk_ingest=after_bulk_ingest,
                km_lo=mid_k + 1,
                km_hi=km_hi,
                freg_from=freg_from,
                freg_to=freg_to,
            )
            return
        if freg_from is None or freg_to is None:
            y0, y1 = CATALOG_FREG_YEAR_MIN, _freg_year_hi()
            mid_y = (y0 + y1) // 2
            print(
                f"[catalog] EUR {price_min} km {km_lo}–{km_hi}: {nr} hits, {np_pages} pages → "
                f"split first registration {y0}–{mid_y} | {mid_y + 1}–{y1}",
                flush=True,
            )
            catalog_partition_collect_rows(
                session,
                profile,
                session_factory,
                rows=rows,
                price_min=price_min,
                price_max=price_max,
                delay=delay,
                depth=depth + 1,
                max_depth=max_depth,
                concurrency=concurrency,
                after_bulk_ingest=after_bulk_ingest,
                km_lo=km_lo,
                km_hi=km_hi,
                freg_from=y0,
                freg_to=mid_y,
            )
            catalog_partition_collect_rows(
                session,
                profile,
                session_factory,
                rows=rows,
                price_min=price_min,
                price_max=price_max,
                delay=delay,
                depth=depth + 1,
                max_depth=max_depth,
                concurrency=concurrency,
                after_bulk_ingest=after_bulk_ingest,
                km_lo=km_lo,
                km_hi=km_hi,
                freg_from=mid_y + 1,
                freg_to=y1,
            )
            return
        if freg_from < freg_to:
            mid_y = (freg_from + freg_to) // 2
            print(
                f"[catalog] EUR {price_min} km {km_lo}–{km_hi} freg {freg_from}–{freg_to}: "
                f"{nr} hits, pages={np_pages} → split years {freg_from}–{mid_y} | {mid_y + 1}–{freg_to}",
                flush=True,
            )
            catalog_partition_collect_rows(
                session,
                profile,
                session_factory,
                rows=rows,
                price_min=price_min,
                price_max=price_max,
                delay=delay,
                depth=depth + 1,
                max_depth=max_depth,
                concurrency=concurrency,
                after_bulk_ingest=after_bulk_ingest,
                km_lo=km_lo,
                km_hi=km_hi,
                freg_from=freg_from,
                freg_to=mid_y,
            )
            catalog_partition_collect_rows(
                session,
                profile,
                session_factory,
                rows=rows,
                price_min=price_min,
                price_max=price_max,
                delay=delay,
                depth=depth + 1,
                max_depth=max_depth,
                concurrency=concurrency,
                after_bulk_ingest=after_bulk_ingest,
                km_lo=km_lo,
                km_hi=km_hi,
                freg_from=mid_y + 1,
                freg_to=freg_to,
            )
            return
        raise RuntimeError(
            "AutoScout24 still returns "
            f"{np_pages} pages for EUR {price_min}, km {km_lo}–{km_hi}, years {freg_from}–{freg_to}. "
            "Add --query-suffix filters (e.g. body type, model) manually."
        )

    extra = ""
    if km_lo is not None and km_hi is not None:
        extra += f", km {km_lo}–{km_hi}"
    if freg_from is not None and freg_to is not None:
        extra += f", freg {freg_from}–{freg_to}"

    for item in page_props.get("listings") or []:
        add_card_row_from_listing_item(rows, profile, item)
    if after_bulk_ingest is not None:
        after_bulk_ingest(rows)

    print(
        f"[catalog] EUR {price_min}–{price_max}{extra}: {nr} listings, pages 1–{np_pages} "
        f"({len(rows)} unique urls total after this slice’s page 1)",
        flush=True,
    )

    tail_pages = list(range(2, np_pages + 1))
    if not tail_pages:
        return

    if concurrency <= 1:
        for page in tail_pages:
            listings = _fetch_search_page_listings(
                session_factory,
                profile,
                page,
                delay,
                price_lo=price_min,
                price_hi=price_max,
                km_lo=km_lo,
                km_hi=km_hi,
                freg_from=freg_from,
                freg_to=freg_to,
            )
            for item in listings:
                add_card_row_from_listing_item(rows, profile, item)
            if after_bulk_ingest is not None:
                after_bulk_ingest(rows)
        return

    workers = max(1, min(concurrency, len(tail_pages)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                _fetch_search_page_listings,
                session_factory,
                profile,
                page,
                delay,
                price_lo=price_min,
                price_hi=price_max,
                km_lo=km_lo,
                km_hi=km_hi,
                freg_from=freg_from,
                freg_to=freg_to,
            )
            for page in tail_pages
        ]
        for fut in as_completed(futures):
            listings = fut.result()
            for item in listings:
                add_card_row_from_listing_item(rows, profile, item)
            if after_bulk_ingest is not None:
                after_bulk_ingest(rows)


def scrape_search_urls(
    profile: SiteProfile,
    session_factory: Callable[[], requests.Session],
    *,
    pages: Iterable[int],
    delay: tuple[float, float],
    concurrency: int,
    referer_profile: SiteProfile | None = None,
    after_bulk_ingest: Callable[[dict[str, dict[str, Any]]], None] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """
    Collect absolute listing URLs from search result pages plus basic card fields.
    """
    rows: dict[str, dict[str, Any]] = {}
    page_list = list(pages)
    if not page_list:
        return [], rows

    def ingest_listings(listings: list[dict[str, Any]]) -> None:
        for item in listings:
            add_card_row_from_listing_item(rows, profile, item)
        if after_bulk_ingest is not None:
            after_bulk_ingest(rows)

    if concurrency <= 1:
        for page in page_list:
            ingest_listings(
                _fetch_search_page_listings(
                    session_factory, profile, page, delay, referer_profile=referer_profile
                )
            )
        return list(rows.keys()), rows

    workers = max(1, min(concurrency, len(page_list)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _fetch_search_page_listings,
                session_factory,
                profile,
                page,
                delay,
                referer_profile=referer_profile,
            )
            for page in page_list
        ]
        for fut in as_completed(futs):
            ingest_listings(fut.result())
    return list(rows.keys()), rows


def _detail_fallback_row(listing_url: str, search_row: dict[str, Any]) -> dict[str, Any]:
    """Keep search-snippet fields when the listing page is gone or unreadable."""
    fallback = dict.fromkeys(SCRAPE_OUTPUT_COLUMNS, None)
    fallback.update({k: v for k, v in search_row.items() if k in SCRAPE_OUTPUT_COLUMNS})
    fallback["listing_url"] = listing_url
    return fallback


def scrape_listing_detail(
    session: requests.Session,
    profile: SiteProfile,
    listing_url: str,
    search_row: dict[str, Any],
    *,
    delay: tuple[float, float],
) -> dict[str, Any]:
    pu = urlparse(listing_url)
    path_fixed = normalize_listing_path(profile, pu.path)
    listing_url = with_culture_query(
        urlunparse((pu.scheme, pu.netloc, path_fixed, pu.params, pu.query, pu.fragment)),
        profile.culture,
    )
    polite_sleep(*delay)
    try:
        r = session.get(listing_url, timeout=40)
    except RequestException:
        return _detail_fallback_row(listing_url, search_row)
    if r.status_code == 403:
        raise RuntimeError(f"Blocked on detail page: {listing_url}")
    if r.status_code in (404, 410, 451):
        return _detail_fallback_row(listing_url, search_row)
    try:
        r.raise_for_status()
    except RequestException:
        return _detail_fallback_row(listing_url, search_row)
    blob = extract_next_data(r.text)
    if not blob:
        return _detail_fallback_row(listing_url, search_row)

    ld = blob["props"]["pageProps"].get("listingDetails") or {}
    v = ld.get("vehicle") or {}
    seller = ld.get("seller") or {}
    location = ld.get("location") or {}
    doors_cnt, seats_cnt = doors_seats_from_listing(ld if isinstance(ld, dict) else {})
    if doors_cnt is None or seats_cnt is None:
        rd, rs = doors_seats_regex_from_next_data_html(r.text)
        if doors_cnt is None:
            doors_cnt = rd
        if seats_cnt is None:
            seats_cnt = rs

    fuel_detail = fmt_maybe_dict(v.get("fuelCategory")) or fmt_maybe_dict(v.get("primaryFuel"))
    prices = ld.get("prices") or {}
    pub = prices.get("public") or ld.get("price") or {}

    cons = v.get("fuelConsumptionCombined")
    cons_txt = fmt_maybe_dict(cons) if cons is not None else None

    co2 = v.get("co2emissionInGramPerKmWithFallback")
    co2_out = fmt_maybe_dict(co2) if isinstance(co2, dict) else co2

    mm = f"{v.get('make','')} {v.get('model','')} {v.get('modelVersionInput','')}".strip()

    row: dict[str, Any] = {
        "listing_url": listing_url,
        "image_url": first_image_url(ld.get("images")) or search_row.get("image_url"),
        "make_model": mm or search_row.get("make_model"),
        "price": fmt_maybe_dict(pub.get("price")),
        "km": fmt_maybe_dict(v.get("mileageInKm")),
        "Fuel": fuel_detail or search_row.get("Fuel"),
        "transmission_card": fmt_maybe_dict(v.get("transmissionType")) or search_row.get("transmission_card"),
        "first_registration_iso": v.get("firstRegistrationDateRaw"),
        "power_hp_raw": v.get("rawPowerInHp"),
        "body_type": fmt_maybe_dict(v.get("bodyType")),
        "vehicle_type_card": fmt_maybe_dict(v.get("type")) or search_row.get("vehicle_type_card"),
        "doors": doors_cnt,
        "seats": seats_cnt,
        "co2_g_per_km": co2_out,
        "cons_comb": cons_txt or fmt_maybe_dict(v.get("wltp")),
        "seller_type_detail": seller.get("type") or search_row.get("seller_type_detail"),
        "city_detail": location.get("city") or search_row.get("city_detail"),
    }

    return row


def merge_card_and_detail(card: dict[str, Any], slot: dict[str, Any]) -> dict[str, Any]:
    """
    Overlay listing-detail on search-snippet without wiping card fields with explicit None.

    Search cards carry WLTP strings etc.; detail JSON often omits the same keys or sets them
    null — a plain {**card, **slot} would erase snippet values and look like a 'partial' scrape.
    """
    merged = dict(card)
    for k, v in slot.items():
        if v is None or k == "listing_url":
            continue
        merged[k] = v
    lu = slot.get("listing_url")
    if isinstance(lu, str) and lu.strip():
        merged["listing_url"] = lu.strip()
    return merged


def merged_rows_for_output(
    urls_slice: list[str],
    card_rows: dict[str, dict[str, Any]],
    detail_slots: list[Any],
) -> list[dict[str, Any]]:
    """One row per URL: search-snippet card plus listing-detail fields when that slot is filled."""
    out: list[dict[str, Any]] = []
    for i, url in enumerate(urls_slice):
        card = dict(card_rows.get(url, {}))
        slot: Any = None
        if i < len(detail_slots):
            slot = detail_slots[i]
        if isinstance(slot, dict):
            merged = merge_card_and_detail(card, slot)
        else:
            merged = card
        out.append(merged)
    return out


def write_scrape_output(rows: list[dict[str, Any]], dest: Path) -> pd.DataFrame:
    """Writes CSV or Parquet with fixed columns."""
    frame = pd.DataFrame(rows).reindex(columns=list(SCRAPE_OUTPUT_COLUMNS))
    if dest.suffix.lower() == ".parquet":
        try:
            frame.to_parquet(dest, index=False)
        except ImportError as exc:
            raise SystemExit(
                "Writing Parquet needs pyarrow (pip install pyarrow). "
                f"Underlying error: {exc}"
            ) from exc
    else:
        frame.to_csv(dest, index=False)
    return frame


def catalog_snippet_checkpoint_path(dest: Path) -> Path:
    """Separate file so `--out` is not overwritten with snippet-only rows before the detail phase."""
    # If `--out` already is the snippet file, do not append `_catalog_snippet` again.
    if dest.stem.endswith("_catalog_snippet"):
        return dest
    return dest.parent / f"{dest.stem}_catalog_snippet{dest.suffix}"


def canonical_listing_url(profile: SiteProfile, raw_url: str) -> str | None:
    """Normalize path (/offers/→/angebote/ on .de) and culture query to match in-memory card keys."""
    s = (raw_url or "").strip()
    if not s:
        return None
    pu = urlparse(s)
    if pu.netloc:
        path = normalize_listing_path(profile, pu.path)
        scheme = pu.scheme or "https"
        u = urlunparse((scheme, pu.netloc, path, pu.params, pu.query, pu.fragment))
    else:
        u = absolutize(profile, pu.path if pu.path.startswith("/") else f"/{pu.path}")
    return with_culture_query(u, profile.culture)


def load_card_rows_from_csv(path: Path, profile: SiteProfile) -> dict[str, dict[str, Any]]:
    """Build listing_url → row dict for SCRAPE_OUTPUT_COLUMNS from a prior export."""
    df = pd.read_csv(path)
    if "listing_url" not in df.columns:
        raise SystemExit(f"{path}: CSV has no listing_url column.")
    out: dict[str, dict[str, Any]] = {}
    for _, rec in df.iterrows():
        key = canonical_listing_url(profile, str(rec.get("listing_url") or ""))
        if not key:
            continue
        row: dict[str, Any] = {}
        for col in SCRAPE_OUTPUT_COLUMNS:
            if col not in df.columns:
                row[col] = None
                continue
            v = rec[col]
            row[col] = None if pd.isna(v) else v
        out[key] = row
    return out


def load_existing_listing_urls(path: Path, profile: SiteProfile) -> set[str]:
    """Read listing_url values from an existing output file and canonicalize for fast skip checks."""
    if not path.is_file():
        return set()
    try:
        df = pd.read_csv(path, usecols=["listing_url"])
    except ValueError:
        return set()
    except Exception:
        return set()
    done: set[str] = set()
    for raw in df.get("listing_url", []):
        key = canonical_listing_url(profile, str(raw or ""))
        if key:
            done.add(key)
    return done


def save_card_rows_checkpoint(
    card_rows: dict[str, dict[str, Any]],
    dest: Path,
    *,
    label: str = "checkpoint",
) -> None:
    """URL-collection phase: search-snippet fields only (doors/seats need listing-detail pages)."""
    ordered = list(card_rows.values())
    if not ordered:
        return
    write_scrape_output(ordered, dest)
    try:
        sz = dest.stat().st_size
    except OSError:
        sz = -1
    print(
        f"Saved {len(ordered)} rows (search-snippet only; doors/seats after detail phase) → {dest} "
        f"({sz} bytes) [{label}]",
        flush=True,
    )


def build_page_range(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    parts = sorted({int(x) for x in spec.replace(",", " ").split() if x.strip()})
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect AutoScout24 listing rows into CSV/Parquet.")
    parser.add_argument(
        "--country",
        choices=sorted(PROFILES.keys()),
        default="de",
        help="Marketplace to crawl (host + cy). Use de for AutoScout24 Germany.",
    )
    parser.add_argument(
        "--language",
        choices=("en", "de"),
        default="en",
        help="en adds culture=en-GB so listing JSON uses English strings on autoscout24.de.",
    )
    parser.add_argument(
        "--pages",
        default="1-3",
        help='Comma list or hyphen range for search pages (e.g. "1-5", "1,3,10"). '
        'Ignored when --catalog-all is set.',
    )
    parser.add_argument(
        "--catalog-all",
        action="store_true",
        help=(
            "Recover listings beyond AutoScout's ~200-page cap by recursively splitting "
            "price brackets; if one EUR price still hits 200 pages, splits by mileage then "
            "first-registration year (fregfrom/fregto) automatically."
        ),
    )
    parser.add_argument(
        "--catalog-price-min",
        type=int,
        default=0,
        help=(
            "With --catalog-all: minimum EUR for search pricefrom (default 0). "
            "When input rows are loaded, pricefrom is also raised to the max parsed snippet EUR unless disabled "
            "with --no-catalog-floor-from-input-max-price."
        ),
    )
    parser.add_argument("--catalog-price-max", type=int, default=9_999_999, help="With --catalog-all: priceto max.")
    parser.add_argument(
        "--catalog-floor-from-input-max-price",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --catalog-all (default: on): if any rows are already loaded from --resume-from-csv / existing --out, "
            "raise pricefrom to at least the highest EUR parsed from the 'price' column so catalog restarts near the "
            "top of the market. You can miss new listings listed below that max. "
            "Use --no-catalog-floor-from-input-max-price to always start at --catalog-price-min only."
        ),
    )
    parser.add_argument(
        "--partition-max-depth",
        type=int,
        default=160,
        help="Safety cap for --catalog-all recursion (price + mileage + year splits share this depth).",
    )
    parser.add_argument(
        "--max-details",
        type=int,
        default=0,
        help="Maximum detail pages after search (0 = unlimited for collected URLs).",
    )
    parser.add_argument("--no-details", action="store_true", help="Skip detail pages (card data only).")
    parser.add_argument("--delay-min", type=float, default=1.0)
    parser.add_argument("--delay-max", type=float, default=3.0)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help=(
            "Number of concurrent HTTP threads for listing result pages (catalog/manual pages) "
            "and listing detail pages. Higher values reduce wall time but increase 429/403 risk."
        ),
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="Optional HTTPS proxy URL, e.g. http://user:pass@host:8888 "
        "(useful when .de/.ch replies with 403).",
    )
    parser.add_argument(
        "--referer-country",
        default=None,
        choices=sorted(PROFILES.keys()),
        help="Optional separate profile whose URL is sent as Referer (rarely needed).",
    )
    parser.add_argument(
        "--out",
        default="autoscout_dataset.csv",
        help="Destination file (.csv recommended; .parquet also supported via pandas).",
    )
    parser.add_argument(
        "--query-suffix",
        default=None,
        help='Extra GET parameters copied from the AutoScout24 filter bar, '
        'e.g. "&mmmv=49%%7CBMW". Must start with & if omitting leading ampersand is handled internally.',
    )
    parser.add_argument(
        "--resume-from-csv",
        metavar="PATH",
        default=None,
        help="Load existing rows (listing_url + same columns as --out) before catalog/search; "
        "URLs already in the file stay in the dict and are updated if seen again.",
    )
    parser.add_argument(
        "--fresh-out",
        action="store_true",
        help="Do not auto-load rows from an existing --out file. Default: if --out already exists as a CSV, "
        "it is merged into memory first so new runs add/update URLs instead of starting from an empty dataset.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Skip catalog and --pages; only listing-detail (or --no-details) runs. "
        "Requires prior rows: use --resume-from-csv and/or an existing --out CSV (auto-loaded unless --fresh-out).",
    )
    parser.add_argument(
        "--skip-existing-out",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before listing-detail fetch, skip URLs already present in --out by matching listing_url "
            "(default: enabled). Use --no-skip-existing-out to re-fetch all details."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=2000,
        help=(
            "During URL collection (--catalog-all / search): save card-only CSV every time this many *unique listing URLs* "
            "have been accumulated. During listing-detail phase: save every this many completed detail rows. "
            "0 disables these periodic saves (Ctrl+C save still applies). Saves overwrite --out (relative paths use cwd)."
        ),
    )
    args = parser.parse_args()

    delay = (max(0.1, args.delay_min), max(args.delay_min, args.delay_max))
    concurrency = max(1, args.concurrency)
    lang_culture_accept: dict[str, tuple[str | None, str]] = {
        "en": ("en-GB", "en-US,en;q=0.9,de-DE;q=0.4,de;q=0.3"),
        "de": (None, "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7"),
    }
    culture, accept_language = lang_culture_accept[args.language]
    base_profile = PROFILES[args.country]
    suffix = (args.query_suffix or "").strip()
    profile = SiteProfile(
        key=base_profile.key,
        base_netloc=base_profile.base_netloc,
        list_prefix=base_profile.list_prefix,
        cy_param=base_profile.cy_param,
        query_suffix=suffix.lstrip("&") if suffix else "",
        culture=culture,
    )
    ref_profile = None
    if args.referer_country:
        rb = PROFILES[args.referer_country]
        ref_profile = SiteProfile(
            key=rb.key,
            base_netloc=rb.base_netloc,
            list_prefix=rb.list_prefix,
            cy_param=rb.cy_param,
            query_suffix="",
            culture=culture,
        )
    session = create_session(args.proxy, accept_language=accept_language)
    session_factory: Callable[[], requests.Session]
    if concurrency <= 1:
        session_factory = lambda s=session: s
    else:
        session_factory = ThreadLocalSessionFactory(args.proxy, accept_language=accept_language)

    if args.catalog_price_min > args.catalog_price_max:
        raise SystemExit("--catalog-price-min cannot exceed --catalog-price-max.")

    outp = Path(args.out).expanduser()
    dest = outp if outp.is_absolute() else (Path.cwd() / outp).resolve()
    snippet_ck = catalog_snippet_checkpoint_path(dest)

    print(f"Working directory: {os.getcwd()}", flush=True)
    print(f"--out as given: {args.out!r}", flush=True)

    checkpoint_every = max(0, args.checkpoint_every)
    ck_lock = threading.Lock()
    if checkpoint_every:
        print(
            f"Output file (absolute): {dest}  (checkpoint every {checkpoint_every} URLs in catalog, "
            f"then every {checkpoint_every} listing-detail rows)",
            flush=True,
        )
        print(
            f"Catalog-phase checkpoints (snippet rows; doors/seats empty until listing pages are fetched): "
            f"{snippet_ck}",
            flush=True,
        )
    else:
        print(f"Output file (absolute): {dest}  (periodic checkpoints off; Ctrl+C still saves)", flush=True)

    catalog_next_milestone = {"n": checkpoint_every if checkpoint_every > 0 else None}
    catalog_ck_lock = threading.Lock()

    def catalog_bulk_hook(rows: dict[str, dict[str, Any]]) -> None:
        """Save search-snippet rows while crawling catalog (listing-detail pass fills/refines fields)."""
        if checkpoint_every <= 0:
            return
        with catalog_ck_lock:
            target_n = catalog_next_milestone["n"]
            if target_n is None:
                return
            ln = len(rows)
            while ln >= target_n:
                snap = dict(rows)
                save_card_rows_checkpoint(snap, snippet_ck, label="catalog checkpoint")
                print(
                    f"catalog checkpoint: {ln} URLs (threshold {target_n}) → {snippet_ck} "
                    f"(doors/seats appear in {dest} after the listing-detail phase)",
                    flush=True,
                )
                target_n += checkpoint_every
            catalog_next_milestone["n"] = target_n

    def flush_detail_checkpoint(slot_list: list[Any], label: str) -> None:
        merged = merged_rows_for_output(urls_slice, card_rows, slot_list)
        if not merged:
            return
        n_detail = sum(1 for x in slot_list if isinstance(x, dict))
        with ck_lock:
            write_scrape_output(merged, dest)
        print(
            f"checkpoint: {label} ({len(merged)} rows, {n_detail} listing-detail pages merged) → {dest}",
            flush=True,
        )

    card_rows: dict[str, dict[str, Any]] = {}
    urls: list[str] = []
    loaded_resume_resolved: Path | None = None

    if args.resume_from_csv:
        rp_csv = Path(args.resume_from_csv).expanduser()
        rp_csv = rp_csv if rp_csv.is_absolute() else (Path.cwd() / rp_csv).resolve()
        if not rp_csv.is_file():
            raise SystemExit(f"--resume-from-csv: not a file: {rp_csv}")
        seeded = load_card_rows_from_csv(rp_csv, profile)
        card_rows.update(seeded)
        loaded_resume_resolved = rp_csv.resolve()
        print(f"Preloaded {len(seeded)} rows from {rp_csv}", flush=True)

    # Merge existing --out so reruns add new URLs instead of starting from an empty dict (unless --fresh-out).
    if (
        not args.fresh_out
        and dest.is_file()
        and dest.suffix.lower() == ".csv"
        and loaded_resume_resolved != dest.resolve()
    ):
        from_out = load_card_rows_from_csv(dest, profile)
        if from_out:
            card_rows.update(from_out)
            print(
                f"Merged {len(from_out)} rows from existing --out ({dest}); "
                f"in-memory URLs now {len(card_rows)} (same listing_url: values from --out win).",
                flush=True,
            )

    if args.csv_only:
        if not card_rows:
            raise SystemExit(
                "csv-only: no rows in memory. Use --resume-from-csv PATH and/or keep your prior --out CSV "
                "(it is auto-loaded unless --fresh-out)."
            )
        urls = list(card_rows.keys())
    else:
        try:
            catalog_price_min_effective = args.catalog_price_min
            if args.catalog_all and args.catalog_floor_from_input_max_price and card_rows:
                mx = max_snippet_price_euro_from_card_rows(card_rows)
                if mx is not None:
                    catalog_price_min_effective = max(catalog_price_min_effective, mx)
                    if catalog_price_min_effective > args.catalog_price_max:
                        raise SystemExit(
                            f"Effective catalog pricefrom EUR {catalog_price_min_effective} (from input max EUR {mx}) "
                            f"exceeds --catalog-price-max {args.catalog_price_max}."
                        )
                    print(
                        f"catalog pricefrom raised to EUR {catalog_price_min_effective} "
                        f"(CLI --catalog-price-min was {args.catalog_price_min}; max parsed snippet price EUR {mx}). "
                        f"Search brackets entirely below this floor are skipped.",
                        flush=True,
                    )
                else:
                    print(
                        "catalog-floor-from-input-max-price: no parseable EUR prices in loaded rows; "
                        f"using --catalog-price-min {args.catalog_price_min}.",
                        flush=True,
                    )
            if args.catalog_all:
                catalog_partition_collect_rows(
                    session,
                    profile,
                    session_factory,
                    rows=card_rows,
                    price_min=catalog_price_min_effective,
                    price_max=args.catalog_price_max,
                    delay=delay,
                    depth=0,
                    max_depth=args.partition_max_depth,
                    concurrency=concurrency,
                    after_bulk_ingest=catalog_bulk_hook if checkpoint_every > 0 else None,
                )
            else:
                pages = build_page_range(args.pages)
                _, collected = scrape_search_urls(
                    profile,
                    session_factory,
                    pages=pages,
                    delay=delay,
                    concurrency=concurrency,
                    referer_profile=ref_profile,
                    after_bulk_ingest=catalog_bulk_hook if checkpoint_every > 0 else None,
                )
                card_rows.update(collected)
            urls = list(card_rows.keys())
        except KeyboardInterrupt:
            print("\nStopped during URL/catalog collection — saving card data collected so far…", flush=True)
            save_card_rows_checkpoint(card_rows, snippet_ck, label="interrupt (catalog)")
            sys.exit(130)

    if not urls:
        raise SystemExit("No listings found; check filters or whether the host returned a block page.")

    queued_urls = urls
    if args.skip_existing_out and not args.no_details:
        existing = load_existing_listing_urls(dest, profile)
        if existing:
            before = len(queued_urls)
            queued_urls = [u for u in queued_urls if u not in existing]
            skipped = before - len(queued_urls)
            print(
                f"Skip-existing-out: found {len(existing)} URLs in {dest}; "
                f"skipping {skipped}, remaining {len(queued_urls)}.",
                flush=True,
            )
        elif dest.is_file():
            print(
                f"Skip-existing-out: {dest} exists but has no usable listing_url column; processing all URLs.",
                flush=True,
            )

    cap = args.max_details or len(queued_urls)
    urls_slice = queued_urls[:cap]

    print(
        f"Listing-detail phase: fetching {len(urls_slice)} listing HTML pages; "
        f"doors/seats (and any missing specs) merge into {dest}",
        flush=True,
    )

    if args.no_details:
        try:
            frame = write_scrape_output([card_rows[u] for u in urls_slice], dest)
        except KeyboardInterrupt:
            print("\nInterrupted — saving collected card rows…", flush=True)
            save_card_rows_checkpoint({u: card_rows[u] for u in urls_slice if u in card_rows}, dest)
            sys.exit(130)
        print(f"Saved {len(frame)} rows (card only) to {dest}")
        return

    def run_detail(ix_ul: tuple[int, str]) -> tuple[int, dict[str, Any]]:
        ix, listing_url = ix_ul
        sess = session_factory()
        rec = scrape_listing_detail(sess, profile, listing_url, card_rows[listing_url], delay=delay)
        return ix, rec

    records_sp: list[Any]
    interrupted = False

    if concurrency <= 1:
        records_sp = []
        try:
            for idx, u in enumerate(urls_slice, start=1):
                records_sp.append(scrape_listing_detail(session, profile, u, card_rows[u], delay=delay))
                if checkpoint_every and len(records_sp) % checkpoint_every == 0:
                    flush_detail_checkpoint(records_sp, f"every {checkpoint_every} details")
                if idx % 25 == 0:
                    print(f"details {idx}/{len(urls_slice)}", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted — saving completed listing-detail rows…", flush=True)
            flush_detail_checkpoint(records_sp, "interrupt (Ctrl+C)")
            sys.exit(130)
        frame = write_scrape_output(merged_rows_for_output(urls_slice, card_rows, records_sp), dest)

    else:
        records_sp = [None] * len(urls_slice)
        workers = max(1, min(concurrency, len(urls_slice), DETAIL_SUBMIT_CHUNK))
        ex = ThreadPoolExecutor(max_workers=workers)
        done = 0
        try:
            for chunk_lo in range(0, len(urls_slice), DETAIL_SUBMIT_CHUNK):
                chunk_hi = min(chunk_lo + DETAIL_SUBMIT_CHUNK, len(urls_slice))
                futs = [
                    ex.submit(run_detail, (i, urls_slice[i])) for i in range(chunk_lo, chunk_hi)
                ]
                for fut in as_completed(futs):
                    ix, rec = fut.result()
                    records_sp[ix] = rec
                    done += 1
                    if checkpoint_every and done % checkpoint_every == 0:
                        flush_detail_checkpoint(records_sp, f"every {checkpoint_every} details")
                    if done % 50 == 0:
                        print(f"details {done}/{len(urls_slice)}", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted — cancelling in-flight listing pages and saving…", flush=True)
            ex.shutdown(wait=False, cancel_futures=True)
            flush_detail_checkpoint(records_sp, "interrupt (Ctrl+C)")
            sys.exit(130)
        finally:
            if not interrupted:
                ex.shutdown(wait=True, cancel_futures=False)

        frame = write_scrape_output(merged_rows_for_output(urls_slice, card_rows, records_sp), dest)

    print(f"Saved {len(frame)} rows to {dest}")


if __name__ == "__main__":
    main()
