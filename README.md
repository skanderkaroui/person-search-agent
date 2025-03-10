# Analyst Research API

An AI-powered API built with FastAPI for generating research reports using LangGraph for multi-step reasoning and OpenAI for response generation.

## Features

- **FastAPI Framework**: Modern, high-performance web framework for building APIs
- **AI Analyst Personas**: Automatically creates analyst personas based on the research topic
- **Simulated Interviews**: Conducts interviews between analysts and AI experts
- **Report Generation**: Compiles insights into a comprehensive research report
- **Multi-Step Reasoning**: Uses LangGraph for structured reasoning workflows
- **AI-Generated Summaries**: OpenAI summarizes the retrieved data

## Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) - Fast, modern Python web framework
- **API Documentation**: Automatic Swagger/OpenAPI documentation
- **AI Model**: OpenAI (GPT-3.5-turbo)
- **Agentic Mechanism**: LangGraph
- **Dependency Management**: Python virtual environments

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

4. Create a `.env` file and add your OpenAI API key:
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

## Usage

1. Start the FastAPI server:
```bash
# Activate the virtual environment
venv\Scripts\activate  # On Windows

# Start the server
uvicorn app.main:app --reload
```

2. Access the interactive API documentation at `http://localhost:8000/docs`

3. Example API request:
```bash
curl -X POST "http://localhost:8000/research" \
     -H "Content-Type: application/json" \
     -d '{"topic": "The impact of artificial intelligence on healthcare", "max_analysts": 2}'
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

## Running Test Scripts

### Option 1: Run the Simple Test Script

```bash
python simple_test.py
```

### Option 2: Run the Analyst Test Script

```bash
python analyst_test.py
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

## License

MIT

## Author

[Skander Karoui](https://github.com/skanderkaroui)
