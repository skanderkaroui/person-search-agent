from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional

from app.graphs.simple_analyst_graph import run_research

# Create router
simple_analyst_router = APIRouter()

# Define request model
class ResearchRequest(BaseModel):
    topic: str
    max_analysts: Optional[int] = 2
    max_turns: Optional[int] = 2

# Define response model
class ResearchResponse(BaseModel):
    report: str

@simple_analyst_router.post("/research", response_model=ResearchResponse)
async def create_research(request: ResearchRequest):
    """
    Generate a research report on a given topic using AI analysts.

    This endpoint:
    1. Creates AI analyst personas based on the topic
    2. Conducts interviews between the analysts and AI experts
    3. Generates a comprehensive research report

    Parameters:
    - topic: The research topic to analyze
    - max_analysts: Maximum number of analysts to create (default: 2)
    - max_turns: Maximum number of conversation turns per interview (default: 2)
    """
    try:
        # Run the research process
        report = run_research(
            topic=request.topic,
            max_analysts=request.max_analysts,
            max_turns=request.max_turns
        )

        # Return the report
        return ResearchResponse(report=report)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating research report: {str(e)}"
        )