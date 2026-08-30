"""
core/database.py
==================

PatientTriage.ai — Database Engine & Session
-----------------------------------------------
SQLAlchemy engine/session setup for the SQLite-backed clinical audit and
staging database. Alembic (see `backend/alembic/`) owns schema migrations
against this same `Base` metadata.
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

# `check_same_thread=False` is required for SQLite when the connection is
# shared across FastAPI's threadpool-executed request handlers.
_connect_args = {"check_same_thread": False, "timeout": 60} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=_connect_args,
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and guarantees it is
    closed after the request, even if an exception is raised.

    Usage
    -----
    ```python
    @router.get("/example")
    def example(db: Session = Depends(get_db)) -> ...:
        ...
    ```
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
