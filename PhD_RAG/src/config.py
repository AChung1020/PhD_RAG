# Global configurations
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    model_name: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 300

    class Config:
        env_file = ".env"

settings = Settings()