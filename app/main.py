from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

# Import the analyst research function
import sys
sys.path.append('.')  # Add the current directory to the path
from analyst_test import run_analyst_research

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Analyst Research API",
    description="API for generating research reports using AI analysts",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class ResearchRequest(BaseModel):
    topic: str
    max_analysts: Optional[int] = 2

# Define response model
class ResearchResponse(BaseModel):
    report: str

@app.post("/research", response_model=ResearchResponse)
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
    """
    try:
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."
            )

        # Run the research process
        report = run_analyst_research(
            topic=request.topic,
            max_analysts=request.max_analysts
        )

        # Return the report
        return ResearchResponse(report=report)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating research report: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {
        "message": "Welcome to the Analyst Research API",
        "version": "0.1.0",
        "docs_url": "/docs",
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )