"""
Database connection shared by all scripts.

Credentials come from environment variables (or a local .env file), never
from source code. Copy .env.example to .env and fill in your values:

    PGHOST=localhost
    PGPORT=5432
    PGDATABASE=credit_card_db
    PGUSER=postgres
    PGPASSWORD=your_password
"""
import os

import psycopg2

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_connection():
    password = os.environ.get("PGPASSWORD")
    if not password:
        raise RuntimeError("Set PGPASSWORD (see .env.example) before running the scripts.")
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "credit_card_db"),
        user=os.environ.get("PGUSER", "postgres"),
        password=password,
    )
