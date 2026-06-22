import argparse
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime

import requests


ENDPOINT = "https://apis.data.go.kr/6260000/BusanCommercialHistoryService/getCommercialHistoryList"
DEFAULT_OUTPUT = os.path.join("data", "busan_commercial_cache.json")
DEFAULT_NUM_OF_ROWS = 1000
DEFAULT_SEARCH_TERM = ""


def load_local_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_point_geom(geom):
    if not geom:
        return None
    match = re.search(r"POINT\(([-\d.]+)\s+([-\d.]+)\)", str(geom))
    if not match:
        return None
    try:
        return {"lon": float(match.group(1)), "lat": float(match.group(2))}
    except ValueError:
        return None


def normalize_items(item):
    if not item:
        return []
    return item if isinstance(item, list) else [item]


def get_nested(data, *keys):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def normalize_cache_item(item):
    point = parse_point_geom(item.get("geom"))
    if not point:
        return None

    open_date = item.get("apvperymd") or item.get("apvpermymd") or "-"
    close_date = item.get("dcbyymd") or item.get("dcbymd") or "-"
    raw_id = "|".join([
        str(item.get("bplcnm") or "").strip(),
        str(item.get("rdnwhladdr") or "").strip(),
        str(item.get("geom") or "").strip(),
        str(open_date or "").strip(),
    ])

    return {
        "id": hashlib.sha1(raw_id.encode("utf-8")).hexdigest(),
        "name": item.get("bplcnm") or "상점명 없음",
        "status": item.get("trdstatenm") or "-",
        "major": item.get("majornm") or "-",
        "minor": item.get("minornm") or "-",
        "businessType": item.get("upjongnm") or "-",
        "address": item.get("rdnwhladdr") or "-",
        "openDate": open_date,
        "closeDate": close_date,
        "lat": point["lat"],
        "lon": point["lon"],
    }


def fetch_page(service_key, search_term, page_no, num_of_rows, timeout):
    params = {
        "serviceKey": service_key,
        "pageNo": page_no,
        "numOfRows": num_of_rows,
        "resultType": "json",
    }
    if search_term:
        params["rdnwhladdr"] = search_term

    response = requests.get(
        ENDPOINT,
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    header = get_nested(payload, "response", "header") or {}
    if header.get("resultCode") not in (None, "00"):
        raise RuntimeError(header.get("resultMsg") or "Busan commercial API returned an error.")

    body = get_nested(payload, "response", "body") or {}
    items = [
        item
        for item in normalize_items(get_nested(body, "items", "item"))
        if isinstance(item, dict)
    ]
    return {
        "totalCount": int(body.get("totalCount") or 0),
        "items": items,
    }


def build_cache(service_key, search_term, output, num_of_rows, max_pages, timeout):
    start_time = time.time()
    first_page = fetch_page(service_key, search_term, 1, num_of_rows, timeout)
    total_count = first_page["totalCount"]
    total_pages = math.ceil(total_count / num_of_rows) if total_count else 1
    if max_pages > 0:
        total_pages = min(total_pages, max_pages)

    seen_ids = set()
    cache_items = []
    raw_items_count = 0
    with_geom_count = 0

    for page_no in range(1, total_pages + 1):
        page = first_page if page_no == 1 else fetch_page(service_key, search_term, page_no, num_of_rows, timeout)
        raw_items_count += len(page["items"])
        for item in page["items"]:
            cached = normalize_cache_item(item)
            if not cached:
                continue
            with_geom_count += 1
            if cached["id"] in seen_ids:
                continue
            seen_ids.add(cached["id"])
            cache_items.append(cached)

        print(f"page {page_no}/{total_pages} fetched={raw_items_count} cached={len(cache_items)}", flush=True)

    cache_data = {
        "meta": {
            "exists": True,
            "updatedAt": datetime.now().isoformat(timespec="seconds"),
            "source": "BusanCommercialHistoryService",
            "endpoint": ENDPOINT,
            "searchTerm": search_term or "ALL",
            "totalCount": total_count,
            "totalPages": total_pages,
            "requestedPages": total_pages,
            "rawItemsCount": raw_items_count,
            "withGeomCount": with_geom_count,
            "dedupedCount": len(cache_items),
            "elapsedSeconds": round(time.time() - start_time, 2),
            "numOfRows": num_of_rows,
        },
        "items": cache_items,
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    temp_path = f"{output}.tmp"
    with open(temp_path, "w", encoding="utf-8") as cache_file:
        json.dump(cache_data, cache_file, ensure_ascii=False, separators=(",", ":"))
    os.replace(temp_path, output)
    return cache_data


def main():
    parser = argparse.ArgumentParser(description="Build Busan commercial history cache for fast nearby-store lookup.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"cache output path. default: {DEFAULT_OUTPUT}")
    parser.add_argument("--search-term", default=DEFAULT_SEARCH_TERM, help="rdnwhladdr search term. default: empty, fetch all")
    parser.add_argument("--num-of-rows", type=int, default=DEFAULT_NUM_OF_ROWS, help="rows per API page. default: 1000")
    parser.add_argument("--max-pages", type=int, default=0, help="limit pages for testing. 0 means all pages.")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds. default: 20")
    args = parser.parse_args()

    load_local_env()
    service_key = os.getenv("BUSAN_COMMERCIAL_SERVICE_KEY")
    if not service_key:
        raise RuntimeError("BUSAN_COMMERCIAL_SERVICE_KEY is missing. Add it to .env or your environment.")

    cache_data = build_cache(
        service_key=service_key,
        search_term=args.search_term,
        output=args.output,
        num_of_rows=args.num_of_rows,
        max_pages=args.max_pages,
        timeout=args.timeout,
    )
    meta = cache_data["meta"]
    print("cache saved:", args.output)
    print("raw items:", meta["rawItemsCount"])
    print("with geom:", meta["withGeomCount"])
    print("deduped:", meta["dedupedCount"])
    print("elapsed seconds:", meta["elapsedSeconds"])


if __name__ == "__main__":
    main()
