import httpx
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class CountriesService:
    """
    Service class to interact with the REST Countries API
    Provides methods to fetch country information including flags, capitals, and other details
    """
    
    BASE_URL = "https://restcountries.com/v3.1"
    
    @staticmethod
    async def fetch_all_countries() -> List[Dict[str, Any]]:
        """
        Fetch all countries from the REST Countries API
        
        Returns:
            List of countries with their information (name, flag, capital, region, etc.)
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{CountriesService.BASE_URL}/all")
                response.raise_for_status()
                countries = response.json()
                
                # Format the response to extract relevant fields
                formatted_countries = []
                for country in countries:
                    formatted_country = {
                        "name": country.get("name", {}).get("common", "N/A"),
                        "official_name": country.get("name", {}).get("official", "N/A"),
                        "flag": country.get("flag", "N/A"),
                        "capital": country.get("capital", ["N/A"])[0] if country.get("capital") else "N/A",
                        "region": country.get("region", "N/A"),
                        "subregion": country.get("subregion", "N/A"),
                        "population": country.get("population", 0),
                        "area": country.get("area", 0),
                        "timezones": country.get("timezones", []),
                        "languages": list(country.get("languages", {}).values()) if country.get("languages") else [],
                        "cca2": country.get("cca2", "N/A"),
                        "cca3": country.get("cca3", "N/A")
                    }
                    formatted_countries.append(formatted_country)
                
                logger.info(f"Successfully fetched {len(formatted_countries)} countries from REST Countries API")
                return formatted_countries
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP Error fetching countries: {str(e)}")
            raise Exception(f"Failed to fetch countries from REST Countries API: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching countries: {str(e)}")
            raise
    
    @staticmethod
    async def fetch_country_by_name(name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a specific country by name
        
        Args:
            name: Country name to search for
            
        Returns:
            Country information if found, None otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{CountriesService.BASE_URL}/name/{name}")
                response.raise_for_status()
                countries = response.json()
                
                if countries:
                    country = countries[0]  # Get the first match
                    formatted_country = {
                        "name": country.get("name", {}).get("common", "N/A"),
                        "official_name": country.get("name", {}).get("official", "N/A"),
                        "flag": country.get("flag", "N/A"),
                        "capital": country.get("capital", ["N/A"])[0] if country.get("capital") else "N/A",
                        "region": country.get("region", "N/A"),
                        "subregion": country.get("subregion", "N/A"),
                        "population": country.get("population", 0),
                        "area": country.get("area", 0),
                        "timezones": country.get("timezones", []),
                        "languages": list(country.get("languages", {}).values()) if country.get("languages") else [],
                        "cca2": country.get("cca2", "N/A"),
                        "cca3": country.get("cca3", "N/A")
                    }
                    logger.info(f"Successfully fetched country: {formatted_country['name']}")
                    return formatted_country
                return None
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP Error fetching country {name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error fetching country {name}: {str(e)}")
            return None
    
    @staticmethod
    async def fetch_countries_by_region(region: str) -> List[Dict[str, Any]]:
        """
        Fetch countries by region
        
        Args:
            region: Region name (e.g., 'Europe', 'Asia', 'Americas', 'Africa', 'Oceania')
            
        Returns:
            List of countries in the specified region
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{CountriesService.BASE_URL}/region/{region}")
                response.raise_for_status()
                countries = response.json()
                
                formatted_countries = []
                for country in countries:
                    formatted_country = {
                        "name": country.get("name", {}).get("common", "N/A"),
                        "official_name": country.get("name", {}).get("official", "N/A"),
                        "flag": country.get("flag", "N/A"),
                        "capital": country.get("capital", ["N/A"])[0] if country.get("capital") else "N/A",
                        "region": country.get("region", "N/A"),
                        "subregion": country.get("subregion", "N/A"),
                        "population": country.get("population", 0),
                        "area": country.get("area", 0),
                    }
                    formatted_countries.append(formatted_country)
                
                logger.info(f"Successfully fetched {len(formatted_countries)} countries from region: {region}")
                return formatted_countries
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP Error fetching countries by region {region}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error fetching countries by region {region}: {str(e)}")
            return []
    
    @staticmethod
    async def fetch_countries_by_code(code: str) -> Optional[Dict[str, Any]]:
        """
        Fetch country by country code (ISO 3166-1 alpha-2 or alpha-3)
        
        Args:
            code: Country code (e.g., 'US', 'IN', 'USA', 'IND')
            
        Returns:
            Country information if found, None otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{CountriesService.BASE_URL}/alpha/{code}")
                response.raise_for_status()
                country = response.json()
                
                formatted_country = {
                    "name": country.get("name", {}).get("common", "N/A"),
                    "official_name": country.get("name", {}).get("official", "N/A"),
                    "flag": country.get("flag", "N/A"),
                    "capital": country.get("capital", ["N/A"])[0] if country.get("capital") else "N/A",
                    "region": country.get("region", "N/A"),
                    "subregion": country.get("subregion", "N/A"),
                    "population": country.get("population", 0),
                    "area": country.get("area", 0),
                    "timezones": country.get("timezones", []),
                    "currencies": list(country.get("currencies", {}).keys()) if country.get("currencies") else [],
                    "cca2": country.get("cca2", "N/A"),
                    "cca3": country.get("cca3", "N/A")
                }
                logger.info(f"Successfully fetched country by code: {formatted_country['name']}")
                return formatted_country
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP Error fetching country by code {code}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error fetching country by code {code}: {str(e)}")
            return None
