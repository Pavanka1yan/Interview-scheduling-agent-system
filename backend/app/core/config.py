"""Application configuration utilities."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic import BaseModel


class Settings(BaseModel):
    DATABASE_URL: str = ""
    PROVIDERS: List[str] = []
    LOG_LEVEL: str = "INFO"
    TZ: str = "Asia/Kolkata"


@lru_cache
def get_settings() -> Settings:
    providers = os.getenv("PROVIDERS", "")
    return Settings(
        DATABASE_URL=os.getenv("DATABASE_URL", ""),
        PROVIDERS=[p.strip() for p in providers.split(",") if p.strip()],
        LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
        TZ=os.getenv("TZ", "Asia/Kolkata"),
    )


settings = get_settings()
