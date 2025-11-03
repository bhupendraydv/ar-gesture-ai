from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

APP_TITLE = "Countries API Proxy"
APP_DESC = (
    "Async FastAPI backend exposing country data endpoints powered by the REST Countries API.\n"
    "Endpoints:\n"
    "- GET /countries\n"
    "- GET /countries/search?q={query}\n"
    "- GET /countries/region/{region}\n"
    "- GET /countries/code/{code}\n"
)

app = FastAPI(title=APP_TITLE, description=APP_DESC, version="1.0.0")

# CORS (allow all by default; adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REST_COUNTRIES_BASE = "https://restcountries.com/v3.1"
DEFAULT_FIELDS = (
    "name,cca2,cca3,ccn3,cioc,capital,region,subregion,languages,"
    "borders,area,population,flags,maps,timezones,idd,latlng"
)


def _fields_param(custom_fields: Optional[str]) -> str:
    if custom_fields is None or not custom_fields.strip():
        return DEFAULT_FIELDS
    # sanitize: allow only alphanum, commas, and underscores
    safe = ",".join(
        f.strip() for f in custom_fields.split(",") if f.strip()
    )
    return safe or DEFAULT_FIELDS


async def _fetch_json(url: str) -> List[dict]:
    timeout = httpx.Timeout(15.0, read=15.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url)
        if resp.status_code == 404:
            # REST Countries returns 404 when no match
            raise HTTPException(status_code=404, detail="No countries found")
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e)) from e
        try:
            data = resp.json()
        except ValueError as e:
            raise HTTPException(status_code=502, detail="Invalid response from upstream API") from e
        # REST Countries returns object for alpha code endpoints; normalize to list
        if isinstance(data, dict):
            return [data]
        return data


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/countries")
async def get_all_countries(fields: Optional[str] = Query(None, description="Comma-separated fields to include")) -> List[dict]:
    url = f"{REST_COUNTRIES_BASE}/all?fields={_fields_param(fields)}"
    return await _fetch_json(url)


@app.get("/countries/search")
async def search_countries(
    q: str = Query(..., min_length=1, description="Search by full or partial country name"),
    fullText: bool = Query(False, description="Use full-text search (exact name match)"),
    fields: Optional[str] = Query(None, description="Comma-separated fields to include"),
) -> List[dict]:
    # REST Countries expects fullText=true|false
    full = "true" if fullText else "false"
    url = f"{REST_COUNTRIES_BASE}/name/{httpx.utils.quote(q, safe='')}?fullText={full}&fields={_fields_param(fields)}"
    return await _fetch_json(url)


@app.get("/countries/region/{region}")
async def countries_by_region(region: str, fields: Optional[str] = Query(None)) -> List[dict]:
    url = f"{REST_COUNTRIES_BASE}/region/{httpx.utils.quote(region, safe='')}?fields={_fields_param(fields)}"
    return await _fetch_json(url)


@app.get("/countries/code/{code}")
async def country_by_code(code: str, fields: Optional[str] = Query(None)) -> List[dict]:
    # alpha endpoint supports alpha-2 or alpha-3 codes
    url = f"{REST_COUNTRIES_BASE}/alpha/{httpx.utils.quote(code, safe='')}?fields={_fields_param(fields)}"
    return await _fetch_json(url)


# Helpful root endpoint
@app.get("/")
async def root() -> dict:
    return {
        "name": APP_TITLE,
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/countries",
            "/countries/search?q=...&fullText=false",
            "/countries/region/{region}",
            "/countries/code/{code}",
        ],
    }
