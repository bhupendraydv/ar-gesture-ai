from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from .countries_service import CountriesService

router = APIRouter(prefix="/countries", tags=["countries"])

@router.get("/", summary="Fetch all countries")
async def get_all_countries() -> List[Dict[str, Any]]:
    try:
        return await CountriesService.fetch_all_countries()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@router.get("/search", summary="Search countries by name")
async def search_countries(name: str = Query(..., min_length=1)) -> Dict[str, Any]:
    country = await CountriesService.fetch_country_by_name(name)
    if not country:
        raise HTTPException(status_code=404, detail=f"Country not found for name: {name}")
    return country

@router.get("/region/{region}", summary="Fetch countries by region")
async def get_countries_by_region(region: str) -> List[Dict[str, Any]]:
    countries = await CountriesService.fetch_countries_by_region(region)
    return countries

@router.get("/code/{code}", summary="Fetch country by country code")
async def get_country_by_code(code: str) -> Dict[str, Any]:
    country = await CountriesService.fetch_countries_by_code(code)
    if not country:
        raise HTTPException(status_code=404, detail=f"Country not found for code: {code}")
    return country
