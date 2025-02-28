# Analyst Research API

A simplified implementation of a research system using LangGraph and OpenAI.

## Overview

This project demonstrates how to build an AI-powered research system using LangGraph. It creates AI analyst personas based on a topic, conducts interviews between the analysts and AI experts, and generates a comprehensive research report.

## Features

- **AI Analyst Personas**: Automatically creates analyst personas based on the research topic
- **Simulated Interviews**: Conducts interviews between analysts and AI experts
- **Report Generation**: Compiles insights into a comprehensive research report
- **FastAPI Interface**: Exposes the functionality through a simple API

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/skanderkaroui/analyst-research-api.git
cd analyst-research-api
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key
```

## Version Compatibility

We've tested and confirmed that the following versions work well together:

```
fastapi==0.115.9
uvicorn==0.34.0
pydantic==2.10.6
pydantic-core==2.27.2
python-dotenv==1.0.1
langchain-core==0.3.40
langchain-openai==0.3.7
langgraph==0.3.1
openai==1.65.1
typing-extensions==4.12.2
langsmith==0.3.11
```

If you encounter dependency conflicts or errors like `ForwardRef._evaluate() missing 1 required keyword-only argument: 'recursive_guard'`, try using these specific versions.

## Running the Application

### Option 1: Run the FastAPI Server

```bash
# Activate the virtual environment
venv\Scripts\activate  # On Windows

# Start the server
uvicorn app.main:app --reload
```

Then access the API at http://localhost:8000/docs

### Option 2: Run the Simple Test Script

```bash
python simple_test.py
```

### Option 3: Run the Analyst Test Script

```bash
python analyst_test.py
```

## API Endpoints

### Generate Research Report

```
POST /research
```

Request body:
```json
{
    "topic": "The impact of artificial intelligence on healthcare",
    "max_analysts": 2
}
```

Response:
```json
{
    "report": "# AI in Healthcare: A Comprehensive Analysis\n\n## Introduction\n..."
}
```

## Project Structure

- `app.py`: Original FastAPI application
- `app/main.py`: Modular FastAPI application
- `simple_test.py`: Simple test script for LangGraph
- `analyst_test.py`: Implementation of the analyst research system
- `requirements.txt`: Project dependencies

## Troubleshooting

If you encounter any issues:

1. Make sure your OpenAI API key is correctly set in the `.env` file
2. Check that all dependencies are installed with the correct versions
3. Ensure you're using Python 3.8 or higher
4. If you see warnings about `json_schema` with GPT-3.5-turbo, these can be safely ignored

## Next Steps

Once this simplified version is working, you can:

1. Add more advanced features like web search integration
2. Implement caching to reduce API calls
3. Add more sophisticated analyst personas and interview techniques
4. Integrate with other data sources

## Author

[Skander Karoui](https://github.com/skanderkaroui)