from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.models.search import PersonSearchRequest, DeepDiveRequest, PersonSearchResponse
from app.services.search_service import SearchService
from app.utils.cache import cache
from app.core.config import settings
from typing import Dict, List, Optional

# Create router
search_router = APIRouter(tags=["search"])

# Dependency to get search service
def get_search_service():
    return SearchService()

@search_router.post("/search", response_model=PersonSearchResponse)
async def search_person(
    request: PersonSearchRequest,
    search_service: SearchService = Depends(get_search_service)
):
    """
    Search for information about a person.

    This endpoint uses LangGraph to determine the best sources to search
    and OpenAI to generate a summary of the retrieved information.
    """
    try:
        result = await search_service.search_person(
            person=request.person,
            sources=request.sources
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching for person: {str(e)}"
        )

@search_router.post("/search/deep_dive", response_model=PersonSearchResponse)
async def deep_dive(
    request: DeepDiveRequest,
    search_service: SearchService = Depends(get_search_service)
):
    """
    Perform a deep dive search on a specific topic related to a person.

    This endpoint allows for more focused searches on specific aspects
    of a person's life, career, or achievements.
    """
    try:
        result = await search_service.deep_dive(
            person=request.person,
            topic=request.topic,
            sources=request.sources
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing deep dive: {str(e)}"
        )

@search_router.delete("/cache", tags=["cache"])
async def clear_cache(source: Optional[str] = None):
    """
    Clear the cache.

    Args:
        source: Optional source name to clear only that source's cache.

    Returns:
        Dict: Status message.
    """
    try:
        cache.clear(source)
        if source:
            return {"status": f"Cache cleared for source: {source}"}
        else:
            return {"status": "All cache cleared"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
        )