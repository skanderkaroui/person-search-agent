from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class PersonSearchRequest(BaseModel):
    """Request model for person search."""

    person: str = Field(..., description="Name of the person to search for")
    sources: Optional[List[str]] = Field(
        default=None,
        description="List of sources to search (e.g., 'Wikipedia', 'Twitter', 'Google'). If not provided, the system will determine the best sources."
    )

    class Config:
        schema_extra = {
            "example": {
                "person": "Elon Musk",
                "sources": ["Wikipedia", "Twitter"]
            }
        }

class DeepDiveRequest(BaseModel):
    """Request model for deep dive search."""

    person: str = Field(..., description="Name of the person to search for")
    topic: str = Field(..., description="Specific topic to explore in depth")
    sources: Optional[List[str]] = Field(
        default=None,
        description="List of sources to search (e.g., 'Wikipedia', 'Twitter', 'Google'). If not provided, the system will determine the best sources."
    )

    class Config:
        schema_extra = {
            "example": {
                "person": "Elon Musk",
                "topic": "SpaceX achievements",
                "sources": ["Wikipedia", "Google"]
            }
        }

class SourceInfo(BaseModel):
    """Model for source information."""

    url: str = Field(..., description="URL of the source")
    title: Optional[str] = Field(None, description="Title of the source")
    snippet: Optional[str] = Field(None, description="Snippet or preview of the source content")

class PersonSearchResponse(BaseModel):
    """Response model for person search."""

    summary: str = Field(..., description="AI-generated summary of the person")
    sources: Dict[str, List[SourceInfo]] = Field(
        ...,
        description="Dictionary mapping source names to lists of source information"
    )

    class Config:
        schema_extra = {
            "example": {
                "summary": "Elon Musk is a billionaire entrepreneur, CEO of Tesla and SpaceX...",
                "sources": {
                    "Wikipedia": [
                        {
                            "url": "https://en.wikipedia.org/wiki/Elon_Musk",
                            "title": "Elon Musk - Wikipedia",
                            "snippet": "Elon Reeve Musk is a business magnate and investor..."
                        }
                    ],
                    "Twitter": [
                        {
                            "url": "https://twitter.com/elonmusk",
                            "title": "Elon Musk (@elonmusk) / X",
                            "snippet": "Elon Musk's official Twitter account"
                        }
                    ]
                }
            }
        }