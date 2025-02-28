# Person Search API

An AI-powered API built with FastAPI for searching information about people using LangGraph for multi-step reasoning and OpenAI for response generation.

## Features

- **FastAPI Framework**: Modern, high-performance web framework for building APIs
- **Person Search**: Retrieve publicly available data on a person
- **Source Filtering**: Specify which sources to search (Twitter, Google)
- **Multi-Step Reasoning**: AI dynamically chooses the best sources for retrieval
- **AI-Generated Summaries**: OpenAI summarizes the retrieved data
- **Deep Dives**: Request further searches for more details on specific topics
- **Web Scraping**: Uses web scraping instead of APIs for data retrieval
- **Caching**: Implements both Redis and in-memory caching for faster responses
- **Rate Limiting**: Protects against abuse with configurable rate limits

## Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) - Fast, modern Python web framework
- **API Documentation**: Automatic Swagger/OpenAPI documentation
- **Dependency Injection**: FastAPI's built-in dependency injection system
- **AI Model**: OpenAI (GPT-4)
- **Agentic Mechanism**: LangGraph
- **Search Sources**: Twitter (via scraping), Google (via scraping)
- **Web Scraping**: BeautifulSoup, Requests
- **Caching**: Redis with in-memory fallback
- **Rate Limiting**: Redis-based with in-memory fallback

## Installation

1. Clone the repository:
```bash
git clone https://github.com/skanderkaroui/person-search-api.git
cd person-search-api
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file based on `.env.example` and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
# Rate limiting configuration
RATE_LIMIT_MAX=60
RATE_LIMIT_WINDOW=60
```

## Usage

1. Start the FastAPI server:
```bash
# First, create a clean virtual environment
python -m venv fresh_venv
source fresh_venv/bin/activate  # On Windows: fresh_venv\Scripts\activate

# Install the minimized dependencies (Wikipedia has been removed)
pip install -r requirements.txt

# If you encounter any dependency conflicts, try:
pip install -r requirements.txt --force-reinstall

# Then start the server:
uvicorn app.main:app --reload
```

2. Access the interactive API documentation at `http://localhost:8000/docs`

3. Example API request:
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
     -H "Content-Type: application/json" \
     -d '{"person": "Elon Musk", "sources": ["Twitter", "Google"]}'
```

## API Endpoints

### Search for a Person

```
POST /api/v1/search
```

Request body:
```json
{
    "person": "Elon Musk",
    "sources": ["Twitter", "Google"]
}
```

Response:
```json
{
    "summary": "Elon Musk is a billionaire entrepreneur, CEO of Tesla and SpaceX...",
    "sources": {
        "Twitter": [
            {
                "url": "https://twitter.com/elonmusk",
                "title": "Tweet by Elon Musk (@elonmusk)",
                "snippet": "Elon Musk's tweet content..."
            }
        ],
        "Google": [
            {
                "url": "https://www.tesla.com/elon-musk",
                "title": "Elon Musk | Tesla",
                "snippet": "Elon Musk co-founded and leads Tesla, SpaceX, Neuralink and The Boring Company."
            }
        ]
    }
}
```

### Deep Dive on a Topic

```
POST /api/v1/search/deep_dive
```

Request body:
```json
{
    "person": "Elon Musk",
    "topic": "SpaceX achievements",
    "sources": ["Google"]
}
```

### Cache Management

```
DELETE /api/v1/cache
```

Optional query parameters:
- `source`: Clear cache for a specific source (e.g., "Twitter", "Google")

Example:
```bash
# Clear all cache
curl -X DELETE "http://localhost:8000/api/v1/cache"

# Clear only Twitter cache
curl -X DELETE "http://localhost:8000/api/v1/cache?source=Twitter"
```

## FastAPI Benefits

This project leverages several key FastAPI features:

1. **Automatic API Documentation**: Interactive documentation at `/docs` using Swagger UI
2. **Request Validation**: Automatic validation using Pydantic models
3. **Dependency Injection**: Clean service integration with `Depends`
4. **Async Support**: Fully asynchronous API endpoints for high performance
5. **Middleware**: Custom rate limiting middleware for request throttling
6. **Type Hints**: Full type annotation for better code quality and IDE support

## Web Scraping Approach

This API uses web scraping instead of relying on external APIs:

- **Twitter**: Scrapes Twitter search results using BeautifulSoup
- **Google**: Scrapes Google search results with fallback to DuckDuckGo

This approach eliminates the need for API keys and rate limits, but may be subject to changes in website structures.

## Caching Mechanism

The API implements a dual-layer caching system:

1. **Redis Cache**: Primary caching mechanism for distributed deployments
2. **In-Memory Cache**: Fallback when Redis is unavailable

Cache entries expire after 1 hour by default to ensure fresh data. The cache significantly reduces the number of web scraping requests, improving performance and reducing the risk of being rate-limited.

## Rate Limiting

To prevent abuse and ensure fair usage, the API implements rate limiting:

- Default: 60 requests per minute per IP address
- Configurable via environment variables (`RATE_LIMIT_MAX` and `RATE_LIMIT_WINDOW`)
- Uses Redis for distributed deployments with in-memory fallback
- Rate limit headers included in responses:
  - `X-RateLimit-Limit`: Maximum number of requests allowed
  - `X-RateLimit-Remaining`: Number of requests remaining in the current window

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t person-search-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 --env-file .env person-search-api
```

## License

MIT

## Author

[Skander Karoui](https://github.com/skanderkaroui)#   p e r s o n - s e a r c h - a g e n t  
 