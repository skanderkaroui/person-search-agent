import os
from pydantic import Field, BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseModel):
    """Application settings."""

    # API Configuration
    API_HOST: str = Field(default=os.getenv("API_HOST", "0.0.0.0"))
    API_PORT: int = Field(default=int(os.getenv("API_PORT", "8000")))
    DEBUG: bool = Field(default=os.getenv("DEBUG", "False").lower() == "true")

    # OpenAI Configuration
    OPENAI_API_KEY: str = Field(default=os.getenv("OPENAI_API_KEY", ""))

    # Redis Configuration
    REDIS_HOST: str = Field(default=os.getenv("REDIS_HOST", "localhost"))
    REDIS_PORT: int = Field(default=int(os.getenv("REDIS_PORT", "6379")))
    REDIS_PASSWORD: str = Field(default=os.getenv("REDIS_PASSWORD", ""))
    REDIS_DB: int = Field(default=int(os.getenv("REDIS_DB", "0")))

    # Rate Limiting Configuration
    RATE_LIMIT_MAX: int = Field(default=int(os.getenv("RATE_LIMIT_MAX", "60")))
    RATE_LIMIT_WINDOW: int = Field(default=int(os.getenv("RATE_LIMIT_WINDOW", "60")))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()