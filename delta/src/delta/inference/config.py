"""
Configuration class for LLM inference package settings.
"""

import os


class Config:
    """Configuration class for LLM inference package settings."""

    MAX_WORKERS = 32

    # Environment variable names (override these to use different env vars)
    OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
    GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
    OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

    _openai_client = None
    _gemini_client = None

    @classmethod
    def set_max_workers(cls, workers: int):
        """Set the maximum number of worker threads.

        Args:
            workers: Number of worker threads (must be >= 1)
        """
        if not isinstance(workers, int) or workers < 1:
            raise ValueError("MAX_WORKERS must be an integer >= 1")
        cls.MAX_WORKERS = workers

    @classmethod
    def get_max_workers(cls):
        """Get the current maximum number of workers."""
        return cls.MAX_WORKERS

    @classmethod
    def set_api_key_env_vars(cls, *, openai: str = None, gemini: str = None, openrouter: str = None):
        """Override the environment variable names used for API keys.

        Args:
            openai: Env var name for the OpenAI API key
            gemini: Env var name for the Gemini API key
            openrouter: Env var name for the OpenRouter API key
        """
        if openai is not None:
            cls.OPENAI_API_KEY_ENV = openai
            cls._openai_client = None
        if gemini is not None:
            cls.GEMINI_API_KEY_ENV = gemini
            cls._gemini_client = None
        if openrouter is not None:
            cls.OPENROUTER_API_KEY_ENV = openrouter

    @classmethod
    def OPENAI_CLIENT(cls):
        if cls._openai_client is None:
            from openai import OpenAI
            cls._openai_client = OpenAI(api_key=os.getenv(cls.OPENAI_API_KEY_ENV))
        return cls._openai_client

    @classmethod
    def GEMINI_CLIENT(cls):
        if cls._gemini_client is None:
            from google import genai
            cls._gemini_client = genai.Client(api_key=os.getenv(cls.GEMINI_API_KEY_ENV))
        return cls._gemini_client

    @classmethod
    def OPENROUTER_API_KEY(cls):
        return os.getenv(cls.OPENROUTER_API_KEY_ENV)
