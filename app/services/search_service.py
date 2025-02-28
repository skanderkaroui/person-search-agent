from typing import Dict, List, Optional
from app.models.search import PersonSearchResponse, SourceInfo
from app.graphs.person_search_graph import PersonSearchGraph
from app.core.config import settings

class SearchService:
    """Service for searching information about people."""

    def __init__(self):
        """Initialize the search service."""
        self.search_graph = PersonSearchGraph()

    async def search_person(
        self,
        person: str,
        sources: Optional[List[str]] = None
    ) -> PersonSearchResponse:
        """
        Search for information about a person.

        Args:
            person: Name of the person to search for.
            sources: Optional list of sources to search.

        Returns:
            PersonSearchResponse: Response containing summary and sources.
        """
        # Execute the search graph
        result = await self.search_graph.execute(
            person=person,
            sources=sources,
            is_deep_dive=False,
            topic=None
        )

        # Format the response
        return PersonSearchResponse(
            summary=result["summary"],
            sources=result["sources"]
        )

    async def deep_dive(
        self,
        person: str,
        topic: str,
        sources: Optional[List[str]] = None
    ) -> PersonSearchResponse:
        """
        Perform a deep dive search on a specific topic related to a person.

        Args:
            person: Name of the person to search for.
            topic: Specific topic to explore in depth.
            sources: Optional list of sources to search.

        Returns:
            PersonSearchResponse: Response containing summary and sources.
        """
        # Execute the search graph with deep dive parameters
        result = await self.search_graph.execute(
            person=person,
            sources=sources,
            is_deep_dive=True,
            topic=topic
        )

        # Format the response
        return PersonSearchResponse(
            summary=result["summary"],
            sources=result["sources"]
        )